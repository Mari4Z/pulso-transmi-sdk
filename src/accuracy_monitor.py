"""Hourly job: for predictions whose target time has now passed, look up
the real demand collected for that timestamp, fill it in, and score
accuracy per forecast horizon (15/30/45/60 min — the "4 tiempos").

This only scores what pulso_transmi.pipeline recorded in `predictions`
when it submitted to the competition API (see record_predictions there);
it never talks to that API itself.
"""
import os
from collections import defaultdict
from datetime import datetime, timezone

from supabase import Client, create_client

PAGE_SIZE = 1000


def _supabase_client() -> Client | None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _latest_observed_at(supabase: Client) -> str | None:
    result = (
        supabase.table("observations")
        .select("observed_at")
        .order("observed_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0]["observed_at"] if rows else None


def _pending_predictions(supabase: Client, up_to: str) -> list[dict]:
    """Predictions whose target_at has real data available and whose
    actual_demand hasn't been filled in yet."""
    rows: list[dict] = []
    offset = 0
    while True:
        page = (
            supabase.table("predictions")
            .select("prediction_id,station_id,target_at,horizon_steps,predicted_demand,model_version_id")
            .is_("actual_demand", "null")
            .lte("target_at", up_to)
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
            .data
        )
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def _actual_demand_by_key(supabase: Client, station_ids: list[str], min_at: str, max_at: str) -> dict:
    lookup: dict[tuple[str, str], int] = {}
    offset = 0
    while True:
        page = (
            supabase.table("observations")
            .select("station_id,observed_at,demand")
            .in_("station_id", station_ids)
            .gte("observed_at", min_at)
            .lte("observed_at", max_at)
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
            .data
        )
        if not page:
            break
        for row in page:
            lookup[(row["station_id"], row["observed_at"])] = row["demand"]
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return lookup


def main() -> None:
    print("Iniciando monitor de accuracy...")
    supabase = _supabase_client()
    if supabase is None:
        print("Variables de Supabase no definidas; nada que hacer.")
        return

    latest_observed_at = _latest_observed_at(supabase)
    if latest_observed_at is None:
        print("Aún no hay observaciones en Supabase; nada que evaluar.")
        return

    pending = _pending_predictions(supabase, latest_observed_at)
    if not pending:
        print("No hay predicciones pendientes de evaluar (target_at ya con datos reales).")
        return

    station_ids = sorted({row["station_id"] for row in pending})
    target_ats = [row["target_at"] for row in pending]
    actuals = _actual_demand_by_key(supabase, station_ids, min(target_ats), max(target_ats))

    resolved: list[dict] = []
    for row in pending:
        key = (row["station_id"], row["target_at"])
        if key not in actuals:
            continue  # data for this timestamp hasn't been collected yet
        actual = actuals[key]
        resolved.append({**row, "actual_demand": actual})

    if not resolved:
        print(f"{len(pending)} predicciones vencidas, pero aún sin observación real correspondiente.")
        return

    print(f"Resolviendo {len(resolved)} de {len(pending)} predicciones vencidas...")
    for row in resolved:
        supabase.table("predictions").update({
            "actual_demand": row["actual_demand"],
            "submission_status": "accepted",
        }).eq("prediction_id", row["prediction_id"]).execute()

    # Accuracy por horizonte ("4 tiempos": 15/30/45/60 min), estilo
    # docs/experimentos-modelos.md: WAPE = sum|error| / sum(actual).
    by_horizon: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in resolved:
        by_horizon[(row["model_version_id"], row["horizon_steps"])].append(row)

    measured_at = datetime.now(timezone.utc)
    metric_rows = []
    for (model_version_id, horizon_steps), rows in by_horizon.items():
        actual_sum = sum(r["actual_demand"] for r in rows)
        if actual_sum <= 0:
            continue
        abs_error_sum = sum(abs(r["actual_demand"] - r["predicted_demand"]) for r in rows)
        wape = abs_error_sum / actual_sum
        accuracy = max(0.0, 1 - wape)
        window_start = min(r["target_at"] for r in rows)
        window_end = max(r["target_at"] for r in rows)
        horizon_minutes = horizon_steps * 15
        print(
            f"  horizonte {horizon_minutes}min: accuracy={accuracy * 100:.2f}% "
            f"wape={wape * 100:.2f}% (n={len(rows)})"
        )
        for metric_name, value in (("accuracy", accuracy), ("wape", wape)):
            metric_rows.append(
                {
                    "model_version_id": model_version_id,
                    "metric_name": f"{metric_name}_{horizon_minutes}min",
                    "window_start": window_start,
                    "window_end": window_end,
                    "metric_value": value,
                    "calculated_at": measured_at.isoformat(),
                }
            )

    if metric_rows:
        supabase.table("metrics").insert(metric_rows).execute()
        print(f"{len(metric_rows)} métricas guardadas en Supabase (tabla 'metrics').")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Advertencia: el monitor de accuracy falló. Error: {e}")
