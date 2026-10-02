from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd

from .client import DEFAULT_BASE_URL, PulsoTransmiClient

MODEL_PATH = Path("models/pulso_hgb_poisson.joblib")
METADATA_PATH = Path("models/pulso_hgb_poisson.json")
HORIZONS = (15, 30, 45, 60)
# Must match ALGORITHM in src/pipeline.py — it's what registers the active
# model_versions row this reads back.
ALGORITHM = "hgb-poisson"

# exp-20260929-shock-blend (docs/experimentos-modelos.md): the trained
# model lags a genuine regime shift by however long it takes for enough
# post-shift rows to accumulate and get retrained on, while a naive
# persistence forecast (last observed value) has zero such lag — it just
# *is* the new level. Blending a modest slice of the model's own lag_1
# feature into its prediction costs ~0.1pp on calm holdout data but wins
# ~5pp during a live regime shift, which is when it matters most (see
# docs). Must match the constants below in src/pipeline.py — that's what
# train_and_evaluate() scores against, so the reported holdout metrics
# match what actually gets submitted here.
# exp-20260930-seasonal-naive: base dropped 0.25->0.05 together with
# upgrading the naive reference itself from plain lag_1 to naive_seasonal
# (see build_features) — same time yesterday, scaled by how the recent
# ~4h level compares to that same window a day ago. It's good enough
# that calm stations do better trusting it *less* toward the model: the
# model already handles calm stations well, so the naive component's
# main job becomes covering stations actually in drift. Beat plain lag_1
# in all 12 stations on the live holdout (+0.2 to +4.7pp each, +2.93pp on
# Banderas specifically) — see docs/experimentos-modelos.md.
NAIVE_BLEND_BASE = 0.05

# exp-20260930-adaptive-blend: a flat 0.25 wasn't enough for a station in
# a *severe* collapse (Banderas stayed at 0% accuracy — predictions still
# ~2-3x the real level) — 75% weight on a model still anchored to the old
# level dominates. Scale the naive weight up with how severe that
# station's own recent drift is (same recent-vs-historical % change
# drift_monitor.py computes), capped at NAIVE_BLEND_SEVERE. Tested: +9.7pp
# in the shock window, +3.6pp on Banderas specifically, for -1.2pp on the
# full holdout (which right now is itself mostly drift-affected rows —
# a calm station never leaves NAIVE_BLEND_BASE).
#
# exp-20260930-extended-ceiling: a station in extreme, still-ongoing
# collapse (Banderas past 80% change) kept scoring near 0% even at 0.65 —
# it was already saturating the old 15-50% scale, so no matter how much
# worse it got the weight couldn't follow. Since a station already at ~0%
# accuracy can't be made worse by this metric, there was no real downside
# to testing higher: raised the ceiling to 0.85 and pushed the saturation
# point out to 80% change. Tested against the real live data during an
# active multi-station shock: Banderas alone went from ~34% to ~55% in a
# 3h window; the per-station-averaged metric across all 12 stations moved
# +0.9-1.3pp (Banderas is only 1/12 of that average) for -0.46pp on the
# full 7-day holdout.
NAIVE_BLEND_SEVERE = 0.85
NAIVE_BLEND_DRIFT_THRESHOLD = 15.0
NAIVE_BLEND_DRIFT_CAP = 80.0


def _station_naive_weights(observations: pd.DataFrame) -> dict[str, float]:
    if observations.empty:
        return {}
    cutoff = observations["observed_at"].max() - pd.Timedelta(1, unit="D")
    recent = observations[observations["observed_at"] >= cutoff]
    historical = observations[observations["observed_at"] < cutoff]
    weights: dict[str, float] = {}
    for station_id, recent_group in recent.groupby("station_id"):
        hist_mean = historical.loc[historical["station_id"] == station_id, "demand"].mean()
        if not hist_mean or pd.isna(hist_mean):
            continue
        pct_change = abs((recent_group["demand"].mean() - hist_mean) / hist_mean * 100)
        if pct_change < NAIVE_BLEND_DRIFT_THRESHOLD:
            weights[station_id] = NAIVE_BLEND_BASE
            continue
        span = NAIVE_BLEND_DRIFT_CAP - NAIVE_BLEND_DRIFT_THRESHOLD
        frac = min(1.0, (pct_change - NAIVE_BLEND_DRIFT_THRESHOLD) / span)
        weights[station_id] = NAIVE_BLEND_BASE + frac * (NAIVE_BLEND_SEVERE - NAIVE_BLEND_BASE)
    return weights


class PipelineError(RuntimeError):
    pass


class CycleClosedError(PipelineError):
    """The cycle closed between get_current_cycle() and submit() — a race,
    not a real failure: cycles here are short-lived (see closes_at) and a
    10-minute external cron can legitimately land right on that edge. Not
    worth failing the whole job over; the next run picks up whatever cycle
    is open next."""


class AlreadySubmittedError(PipelineError):
    """The API already accepted a submission for this cycle under a
    different idempotency key — found live during the stronger drift
    wave (docs/estrategia-drift.md): cycles now outlast the 10-min predict
    cadence, so a later run in the same still-open cycle recomputes
    slightly different predictions (fresh observations moved lag_1/
    naive_seasonal) and gets a 409 trying to resubmit. The API's contract
    is one accepted submission per cycle; this is that contract working
    as intended, not a bug — nothing more to do this cycle."""


def _git_commit() -> str:
    value = os.getenv("GITHUB_SHA")
    if value:
        return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _request_headers(api_key: str) -> dict[str, str]:
    headers = {"User-Agent": "pulso-transmi-pipeline/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def get_current_cycle(base_url: str, api_key: str, client: httpx.Client) -> dict[str, Any] | None:
    response = client.get(
        f"{base_url.rstrip('/')}/v1/forecast-cycles/current",
        headers=_request_headers(api_key),
    )
    if response.status_code == 404:
        return None
    if response.is_error:
        raise PipelineError(
            f"GET /v1/forecast-cycles/current failed with HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    cycle = response.json()
    if cycle.get("state") != "open":
        return None
    return cycle


def build_features(
    observations: pd.DataFrame,
    stations: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    # `context` (weather/events) has no live stream and stops at the static
    # history's end, same as plain observations_dataframe() does — but by
    # the time this runs, `observations` may extend well past that (see
    # PulsoTransmiClient.all_observations_dataframe). An inner merge would
    # silently drop every one of those newer rows for lack of a context
    # match, putting the "latest observation per station" right back to the
    # stale history_end row regardless of how fresh `observations` is. Left
    # merge + forward-fill keeps those rows, using the last known context.
    frame = observations.merge(stations, on="station_id").merge(context, on="observed_at", how="left")
    context_columns = [c for c in context.columns if c != "observed_at"]
    frame = frame.sort_values("observed_at")
    frame[context_columns] = frame[context_columns].ffill()
    frame["local_at"] = frame["observed_at"].dt.tz_convert("America/Bogota")
    frame["local_hour"] = frame["local_at"].dt.hour
    frame["local_weekday"] = frame["local_at"].dt.weekday
    frame["is_weekend"] = (frame["local_weekday"] >= 5).astype(int)
    frame = frame.sort_values(["station_id", "observed_at"]).reset_index(drop=True)
    demand_by_station = frame.groupby("station_id", sort=False)["demand"]
    for lag in (1, 4, 16, 96, 672):
        frame[f"lag_{lag}"] = demand_by_station.shift(lag)
    for window in (4, 16, 96, 672):
        frame[f"rolling_mean_{window}"] = (
            demand_by_station.shift(1)
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )
    # Must mirror src/pipeline.py::build_features (exp-20260928-hgb-poisson-004):
    # _prepare_matrix below reindexes to whatever feature_columns the active
    # model shipped with, filling anything missing with 0.0 — a model trained
    # with trend_16/trend_96 would silently get 0.0 instead of the real value
    # here if this branch didn't compute them too.
    frame["trend_16"] = frame["lag_1"] - frame["rolling_mean_16"]
    frame["trend_96"] = frame["lag_1"] - frame["rolling_mean_96"]
    # exp-20260930-seasonal-naive: must mirror src/pipeline.py — this is
    # the naive-blend reference used below in predict_targets, not a model
    # feature, but it has to be computed the same way on both sides.
    frame["rolling_mean_16_lag96"] = frame.groupby("station_id")["rolling_mean_16"].shift(96)
    frame["naive_seasonal"] = frame["lag_96"] * (
        (frame["rolling_mean_16"] + 1.0) / (frame["rolling_mean_16_lag96"] + 1.0)
    )
    return frame


def add_training_statistics(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    station_mean = source.groupby("station_id")["demand"].mean()
    station_hour_mean = source.groupby(["station_id", "local_hour"])["demand"].mean()
    station_weekday_hour_mean = source.groupby(
        ["station_id", "local_weekday", "local_hour"]
    )["demand"].mean()
    result["station_mean"] = result["station_id"].map(station_mean)
    result["station_hour_mean"] = [
        station_hour_mean.get((station, hour), np.nan)
        for station, hour in zip(result["station_id"], result["local_hour"])
    ]
    result["station_weekday_hour_mean"] = [
        station_weekday_hour_mean.get((station, weekday, hour), np.nan)
        for station, weekday, hour in zip(
            result["station_id"], result["local_weekday"], result["local_hour"]
        )
    ]
    return result


def _prepare_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    categorical = [column for column in ("station_id", "corridor") if column in frame]
    numeric = [
        column
        for column in frame.columns
        if column not in categorical
        and column not in {"observed_at", "local_at", "demand"}
    ]
    matrix = pd.get_dummies(frame[numeric + categorical], columns=categorical, dtype=float)
    return matrix.reindex(columns=feature_columns, fill_value=0.0)


def predict_targets(
    cycle: dict[str, Any],
    observations: pd.DataFrame,
    stations: pd.DataFrame,
    context: pd.DataFrame,
    package: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = int(cycle["expected_predictions"])
    targets = cycle.get("targets", [])
    if len(targets) != expected:
        raise PipelineError(f"cycle declares {expected} targets but returned {len(targets)}")
    feature_frame = build_features(observations, stations, context)
    cutoff = pd.Timestamp(cycle["data_cutoff"], tz="UTC")
    available = feature_frame[feature_frame["observed_at"] <= cutoff]
    latest = available.groupby("station_id", as_index=False).tail(1)
    latest = add_training_statistics(latest, available)
    if latest["station_id"].nunique() != len(stations):
        raise PipelineError("missing latest observations for one or more stations")

    feature_columns = package["feature_columns"]
    models = package["models"]
    naive_weights = _station_naive_weights(observations)
    predictions: list[dict[str, Any]] = []
    for target in targets:
        station_id = str(target["station_id"])
        horizon = int(target["horizon_minutes"])
        row = latest[latest["station_id"] == station_id]
        if row.empty or horizon not in HORIZONS:
            raise PipelineError(f"cannot predict station={station_id} horizon={horizon}")
        matrix = _prepare_matrix(row, feature_columns)
        model_value = float(models[horizon // 15].predict(matrix)[0])
        # naive_seasonal needs a full day of history behind this row (lag_96
        # + rolling_mean_16 from a day ago) — falls back to plain lag_1 on
        # the rare row where that's not available yet (e.g. right after a
        # station's very first observations) rather than blending in a NaN.
        naive_seasonal = row["naive_seasonal"].iloc[0]
        naive_value = float(naive_seasonal) if pd.notna(naive_seasonal) else float(row["lag_1"].iloc[0])
        weight = naive_weights.get(station_id, NAIVE_BLEND_BASE)
        value = (1 - weight) * model_value + weight * naive_value
        predictions.append(
            {
                "station_id": station_id,
                "target_at": target["target_at"],
                "value": max(0.0, value),
            }
        )
    validate_predictions(predictions, targets, expected)
    return predictions


def validate_predictions(
    predictions: list[dict[str, Any]], targets: list[dict[str, Any]], expected: int
) -> None:
    prediction_keys = [(row["station_id"], row["target_at"]) for row in predictions]
    target_keys = [(row["station_id"], row["target_at"]) for row in targets]
    values = [row["value"] for row in predictions]
    if len(predictions) != expected or len(set(prediction_keys)) != expected:
        raise PipelineError("predictions do not contain exactly one row per target")
    if set(prediction_keys) != set(target_keys):
        raise PipelineError("prediction targets do not match the current cycle")
    if not all(np.isfinite(value) and value >= 0 for value in values):
        raise PipelineError("predictions must be finite and non-negative")


def stable_idempotency_key(cycle_id: str, model_version: str, predictions: list[dict[str, Any]]) -> str:
    content = json.dumps(
        {"cycle_id": cycle_id, "model_version": model_version, "predictions": predictions},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"pulso-{hashlib.sha256(content).hexdigest()}"


def submit(
    base_url: str,
    api_key: str,
    cycle: dict[str, Any],
    model_metadata: dict[str, Any],
    predictions: list[dict[str, Any]],
    client: httpx.Client,
) -> dict[str, Any]:
    model_version = model_metadata["model_version"]
    idempotency_key = stable_idempotency_key(cycle["cycle_id"], model_version, predictions)
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle["cycle_id"],
        "client_run_id": idempotency_key,
        "data_cutoff": cycle["data_cutoff"],
        "model": {
            "version": model_version,
            "trained_at": model_metadata.get("created_at"),
            "training_data_end": model_metadata.get("training_cutoff"),
            "git_commit": _git_commit(),
        },
        "predictions": predictions,
    }
    response = client.post(
        f"{base_url.rstrip('/')}/v1/submissions",
        headers={**_request_headers(api_key), "Content-Type": "application/json", "Idempotency-Key": idempotency_key},
        json=payload,
    )
    if response.status_code == 409:
        try:
            code = response.json().get("detail", {}).get("code")
        except ValueError:
            code = None
        if code == "cycle_closed":
            raise CycleClosedError(f"cycle {cycle['cycle_id']} closed before submission: {response.text[:500]}")
        if code == "idempotency_conflict":
            raise AlreadySubmittedError(
                f"cycle {cycle['cycle_id']} already has an accepted submission: {response.text[:500]}"
            )
    if response.status_code not in (200, 201):
        raise PipelineError(f"POST /v1/submissions failed with HTTP {response.status_code}: {response.text[:500]}")
    return {"idempotency_key": idempotency_key, "response": response.json()}


def _active_model_version_id(supabase: Any) -> str | None:
    result = (
        supabase.table("model_versions")
        .select("model_version_id")
        .eq("algorithm", ALGORITHM)
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0]["model_version_id"] if rows else None


def record_predictions(
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    """Best-effort: store what we predicted so accuracy can be scored later
    once the target time has passed and the real demand is known. Never
    raises — a submission that already succeeded with the competition API
    must not fail the job over local bookkeeping."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        print("Variables Supabase no definidas, predicciones no registradas localmente.")
        return
    try:
        from supabase import create_client

        supabase = create_client(supabase_url, supabase_key)
        model_version_id = _active_model_version_id(supabase)
        if not model_version_id:
            print("No hay model_versions activo en Supabase; predicciones no registradas.")
            return
        rows = [
            {
                "model_version_id": model_version_id,
                "station_id": pred["station_id"],
                "target_at": pred["target_at"],
                "horizon_steps": int(target["horizon_minutes"]) // 15,
                "predicted_demand": pred["value"],
                "submission_status": "sent",
            }
            for target, pred in zip(targets, predictions)
        ]
        supabase.table("predictions").upsert(
            rows, on_conflict="model_version_id,station_id,target_at,horizon_steps"
        ).execute()
        print(f"{len(rows)} predicciones registradas en Supabase para evaluación de accuracy.")
    except Exception as exc:
        print(f"Advertencia: no se pudieron registrar las predicciones en Supabase: {exc}")


def main() -> None:
    api_key = os.environ.get("PULSO_API_KEY")
    if not api_key:
        raise PipelineError("PULSO_API_KEY is required")
    base_url = os.environ.get("PULSO_API_URL") or DEFAULT_BASE_URL
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise PipelineError("promoted model and metadata files are required")

    model_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    package = joblib.load(MODEL_PATH)
    with httpx.Client(timeout=60, follow_redirects=True) as http_client:
        # A single check per run: the caller (cron-job.org / workflow_dispatch)
        # is expected to trigger this pipeline every few minutes, so we don't
        # need to block the job waiting for a cycle to open.
        cycle = get_current_cycle(base_url, api_key, http_client)
        if cycle is None:
            print("no open forecast cycle")
            return
        with PulsoTransmiClient(base_url=base_url, api_key=api_key, timeout=60) as client:
            # Static history alone stops at the dataset's history_end and
            # never advances — the live stream is what actually reaches the
            # cycle's data_cutoff (see all_observations_dataframe).
            observations = client.all_observations_dataframe(page_size=5000)
            stations = client.stations()
            context = client.context_dataframe(page_size=5000)
        predictions = predict_targets(cycle, observations, stations, context, package)
        try:
            result = submit(base_url, api_key, cycle, model_metadata, predictions, http_client)
        except CycleClosedError as exc:
            print(f"ciclo cerrado antes de poder enviar, se omite este ciclo: {exc}")
            return
        except AlreadySubmittedError as exc:
            # No se reintenta con un idempotency key distinto: la API ya
            # aceptó una respuesta para este ciclo (de una corrida anterior
            # dentro del mismo ciclo, todavía abierto) y esa es la que se
            # califica — no record_predictions() aquí, porque guardaríamos
            # una predicción que nunca se envió de verdad.
            print(f"este ciclo ya tiene una submission aceptada, se omite: {exc}")
            return
    record_predictions(cycle.get("targets", []), predictions)
    receipt = {
        "cycle_id": cycle["cycle_id"],
        "model_version": model_metadata["model_version"],
        "prediction_count": len(predictions),
        "idempotency_key": result["idempotency_key"],
        "submission": result["response"],
    }
    receipt_path = Path("artifacts") / f"submission-{cycle['cycle_id']}.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(receipt, sort_keys=True)
    )


if __name__ == "__main__":
    main()
