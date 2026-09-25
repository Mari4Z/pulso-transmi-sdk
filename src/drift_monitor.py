import os
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import ks_2samp
from supabase import Client, create_client

from pulso_transmi import PulsoTransmiClient


# A drift signal below this magnitude is logged and shown on the dashboard
# but doesn't justify an automatic retrain by itself (KS is very sensitive
# at these sample sizes, so small shifts trigger drift_detected constantly).
SEVERE_MEAN_CHANGE_PCT = 15.0


def _pct_change(historical: float, recent: float) -> float | None:
    if historical == 0:
        return None
    return (recent - historical) / abs(historical) * 100.0


def _write_severe_output(severe: bool) -> None:
    """Expose the retrain decision to the calling workflow via $GITHUB_OUTPUT."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"severe_drift={'true' if severe else 'false'}\n")


def main() -> None:
    print("Iniciando monitor de drift...")
    print("Descargando observaciones...")
    with PulsoTransmiClient(timeout=60) as client:
        # page_size es por página, no un límite total: el cliente pagina
        # automáticamente hasta traer todo el histórico disponible.
        observations = client.observations_dataframe(page_size=5000)

    if observations.empty:
        print("No hay suficientes datos para drift.")
        _write_severe_output(False)
        return

    # Los datos más recientes (último día disponible) vs. todo lo anterior como baseline
    cutoff = observations["observed_at"].max() - pd.Timedelta(1, unit="D")
    recent = observations[observations["observed_at"] >= cutoff]
    historical = observations[observations["observed_at"] < cutoff]

    if recent.empty or historical.empty:
        print("No hay suficientes datos particionados para calcular drift (necesitamos datos de más de 1 día).")
        _write_severe_output(False)
        return

    features_to_monitor = ["demand"]
    drift_logs = []

    for feature in features_to_monitor:
        if feature not in recent.columns or feature not in historical.columns:
            continue
        recent_vals = recent[feature].dropna()
        hist_vals = historical[feature].dropna()
        if recent_vals.empty or hist_vals.empty:
            continue

        stat, p_value = ks_2samp(hist_vals, recent_vals)
        drift_detected = bool(p_value < 0.05)

        historical_mean = float(hist_vals.mean())
        recent_mean = float(recent_vals.mean())
        historical_std = float(hist_vals.std())
        recent_std = float(recent_vals.std())

        drift_logs.append(
            {
                "feature": feature,
                "ks_stat": float(stat),
                "p_value": float(p_value),
                "drift_detected": drift_detected,
                "measured_at": str(datetime.now(timezone.utc)),
                "historical_mean": historical_mean,
                "recent_mean": recent_mean,
                "mean_change_pct": _pct_change(historical_mean, recent_mean),
                "historical_std": historical_std,
                "recent_std": recent_std,
                "std_change_pct": _pct_change(historical_std, recent_std),
                "recent_samples": int(len(recent_vals)),
                "historical_samples": int(len(hist_vals)),
            }
        )

    if not drift_logs:
        print("No se pudo calcular drift para ninguna feature.")
        _write_severe_output(False)
        return

    print("Resultados de Drift:")
    for log in drift_logs:
        mean_change = log["mean_change_pct"]
        change_str = f"{mean_change:+.1f}%" if mean_change is not None else "n/a"
        print(
            f"  {log['feature']}: Drift={'Si' if log['drift_detected'] else 'No'} "
            f"(p={log['p_value']:.4f}, ks={log['ks_stat']:.4f}, cambio_media={change_str})"
        )

    severe = any(
        log["drift_detected"] and abs(log["mean_change_pct"] or 0) >= SEVERE_MEAN_CHANGE_PCT
        for log in drift_logs
    )
    print(f"¿Drift severo (retrain automático)? {'Sí' if severe else 'No'}")
    _write_severe_output(severe)

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("Variables Supabase no definidas, drift no guardado.")
        return

    print("Guardando drift en Supabase (tabla 'model_drift_log')...")
    try:
        supabase: Client = create_client(supabase_url, supabase_key)
        supabase.table("model_drift_log").insert(drift_logs).execute()
        print("Drift guardado con éxito.")
    except Exception as e:
        print(f"Error al guardar drift en Supabase: {e}")
        # El script sigue funcionando a pesar del error


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Advertencia: El monitor de drift falló pero no detendrá el flujo principal. Error: {e}")
