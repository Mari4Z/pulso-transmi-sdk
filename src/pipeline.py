from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from supabase import Client, create_client

from pulso_transmi import PulsoTransmiClient
from pulso_transmi.client import DEFAULT_BASE_URL

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "pulso_hgb_poisson.joblib"
METADATA_PATH = MODEL_DIR / "pulso_hgb_poisson.json"

# Local file store, committed to git alongside models/ (see pipeline.yml's
# "Publicar modelo" step) instead of a hosted tracking server — this is a
# student project, not a team that needs concurrent write access to one
# store. Run `mlflow ui` from the repo root to browse the accumulated
# experiment history.
MLFLOW_TRACKING_DIR = Path("mlruns")
MLFLOW_EXPERIMENT_NAME = "pulso-transmi-hgb-poisson"

# Horizon steps in units of 15 minutes (1 -> 15min, 2 -> 30min, ...). The
# `models` dict in the joblib package is keyed by these step numbers, not by
# minutes: pulso_transmi/pipeline.py looks predictions up with
# `models[horizon_minutes // 15]`, so this keying must match on both sides.
HORIZONS = (1, 2, 3, 4)
DATASET_VERSION = "pulso-transmi-starter-v1"
ALGORITHM = "hgb-poisson"
FEATURE_SET_ID = "pulso-hgb-poisson-features"
FEATURE_SET_VERSION = "v1"

# exp-20260929-shock-blend / exp-20260930-adaptive-blend
# (docs/experimentos-modelos.md): must match the constants and
# _station_naive_weights() in pulso_transmi/pipeline.py — that's what
# actually gets submitted at predict time, so scoring it here with the
# same blend keeps these holdout numbers an honest preview instead of
# measuring a pure-model prediction we never actually send.
NAIVE_BLEND_BASE = 0.25
NAIVE_BLEND_SEVERE = 0.65
NAIVE_BLEND_DRIFT_THRESHOLD = 15.0
NAIVE_BLEND_DRIFT_CAP = 50.0


def _station_naive_weights(observations: pd.DataFrame) -> dict[str, float]:
    if observations.empty:
        return {}
    drift_cutoff = observations["observed_at"].max() - pd.Timedelta(1, unit="D")
    recent = observations[observations["observed_at"] >= drift_cutoff]
    historical = observations[observations["observed_at"] < drift_cutoff]
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

# exp-20260927-hgb-poisson-002 (see docs/experimentos-modelos.md): a 12-run
# random search plus a same-size follow-up around its best region, both
# scored on the real 7-day holdout, clustered tightly (83.0-83.7%) — this
# model isn't hyperparameter-starved. It replaces
# exp-20260918-hgb-poisson-001's config (max_iter=300, learning_rate=0.06,
# max_leaf_nodes=31, l2_regularization=1.0), which scored 83.14% on the same
# holdout, +0.55pp lower.
HYPERPARAMETERS = {
    "max_iter": 300,
    "learning_rate": 0.08,
    "max_leaf_nodes": 63,
    "l2_regularization": 2.0,
    "random_state": 42,
}

# rain_mm/rain_forecast/temperature_c/temperature_forecast/event_intensity
# dropped in the same search: they only ever reflect the frozen dataset
# history (`context` has no live stream, see build_features), so at predict
# time they're always the last known value forward-filled — stale signal
# that cost ~0.5pp of holdout accuracy rather than helping. Predict-time
# needs no change for this: pulso_transmi.pipeline._prepare_matrix already
# reindexes to whatever `feature_columns` the active model shipped with, so
# it drops these columns on its own once a model trained without them is
# active.
# station_mean/station_hour_mean/station_weekday_hour_mean dropped here
# (exp-20260928-hgb-poisson-003, docs/experimentos-modelos.md): they're
# plain averages over the *entire* training window, so when a station's
# demand level actually shifts (Banderas/05100 collapsed ~47% around
# 2026-09-13, isolated to that one station), they keep anchoring
# predictions to the old level for as long as the old regime's rows still
# outnumber the new ones — which, given how much history 45 days of static
# data plus a slowly growing live stream produce, is a long time. Windowing
# them to a recent slice doesn't fix it either: a short window doesn't have
# enough (station, weekday, hour) coverage, so rows whose combination falls
# outside it get an all-NaN lookup and are dropped from training entirely.
# Tested on the 2 days right after the collapse: dropping these features
# improved both the system-wide WAPE (26.4%->23.4%) and Banderas'
# specifically (accuracy 0%->23.5%) — the lag/rolling_mean features already
# carry the responsive, recent signal these were duplicating less well.
# exp-20260928-hgb-poisson-004 (docs/experimentos-modelos.md): trend_16/
# trend_96 (lag_1 minus the matching rolling_mean) tell the model how far
# the *current* reading sits from its recent baseline, instead of just
# handing it the lag and the mean separately and leaving it to infer the
# gap. Improved holdout accuracy system-wide (81.78%->82.38%) and,
# tellingly, more so at Banderas specifically (75.39%->77.31%) — the same
# station exp-003 targeted, so this compounds with dropping the slow
# station averages rather than duplicating that fix. rolling_std_*/
# hour_sin+cos were tested alongside and didn't clear the noise floor on
# their own, so they were left out.
NUMERIC_FEATURES = (
    "latitude", "longitude",
    "local_hour", "local_weekday", "is_weekend",
    "lag_1", "lag_4", "lag_16", "lag_96", "lag_672",
    "rolling_mean_4", "rolling_mean_16", "rolling_mean_96", "rolling_mean_672",
    "trend_16", "trend_96",
)
CATEGORICAL_FEATURES = ("station_id", "corridor")


def build_features(observations: pd.DataFrame, stations: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    # `context` has no live stream and stops at the static history's end;
    # `observations` now extends past that via all_observations_dataframe().
    # An inner merge would silently drop every row newer than context's max,
    # capping training at the same stale window as before. Left merge +
    # forward-fill keeps them, reusing the last known context.
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
            demand_by_station.shift(1).rolling(window).mean().reset_index(level=0, drop=True)
        )
    frame["trend_16"] = frame["lag_1"] - frame["rolling_mean_16"]
    frame["trend_96"] = frame["lag_1"] - frame["rolling_mean_96"]
    return frame


# No longer called from train_and_evaluate (see the NUMERIC_FEATURES note
# above) — kept only because src/inference.py still imports and calls it.
# Harmless either way: the columns it adds aren't in NUMERIC_FEATURES, so
# prepare_matrix() never selects them.
def add_training_statistics(train: pd.DataFrame, target: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    station_mean = train.assign(target=target).groupby("station_id")["target"].mean()
    station_hour_mean = train.assign(target=target).groupby(["station_id", "local_hour"])["target"].mean()
    station_weekday_hour_mean = train.assign(target=target).groupby(
        ["station_id", "local_weekday", "local_hour"]
    )["target"].mean()
    frame["station_mean"] = frame["station_id"].map(station_mean)
    frame["station_hour_mean"] = [
        station_hour_mean.get((station, hour), np.nan)
        for station, hour in zip(frame["station_id"], frame["local_hour"])
    ]
    frame["station_weekday_hour_mean"] = [
        station_weekday_hour_mean.get((station, weekday, hour), np.nan)
        for station, weekday, hour in zip(frame["station_id"], frame["local_weekday"], frame["local_hour"])
    ]
    return frame


def prepare_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        frame[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)],
        columns=list(CATEGORICAL_FEATURES),
        dtype=float,
    )


def _git_commit() -> str:
    value = os.getenv("GITHUB_SHA")
    if value:
        return value
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _supabase_client() -> Client | None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def train_and_evaluate(
    frame: pd.DataFrame, cutoff: pd.Timestamp, validation_end: pd.Timestamp
) -> tuple[dict, list[str], dict[str, int], dict[int, dict[str, float | int | None]]]:
    """Train each horizon's model on rows before `cutoff`, then score it on
    the held-out window [cutoff, validation_end] — same WAPE/accuracy
    definition as accuracy_monitor.py, so these numbers are directly
    comparable to what that job (and the dashboard) reports for live
    predictions, just measured immediately instead of waiting on ground
    truth to arrive."""
    models: dict[int, HistGradientBoostingRegressor] = {}
    feature_columns: list[str] | None = None
    rows_by_horizon: dict[str, int] = {}
    metrics_by_horizon: dict[int, dict[str, float | int | None]] = {}
    naive_weights = _station_naive_weights(frame[["observed_at", "station_id", "demand"]])

    for horizon in HORIZONS:
        training = frame.copy()
        training["target_at"] = training["observed_at"] + pd.Timedelta(int(15 * horizon), unit="m")
        training["target"] = training.groupby("station_id")["demand"].shift(-horizon)
        eligible = training.dropna(subset=["target"]).copy()

        train_rows = eligible[eligible["target_at"] < cutoff].copy()
        val_rows = eligible[
            (eligible["target_at"] >= cutoff) & (eligible["target_at"] <= validation_end)
        ].copy()

        train_rows = train_rows.dropna(subset=list(NUMERIC_FEATURES)).copy()
        val_rows = val_rows.dropna(subset=list(NUMERIC_FEATURES)).copy()

        y_train = train_rows.pop("target")
        matrix = prepare_matrix(train_rows)
        feature_columns = matrix.columns.tolist()
        model = HistGradientBoostingRegressor(loss="poisson", **HYPERPARAMETERS)
        model.fit(matrix, y_train)
        models[horizon] = model
        rows_by_horizon[str(horizon * 15)] = len(train_rows)

        horizon_minutes = horizon * 15
        if val_rows.empty:
            metrics_by_horizon[horizon_minutes] = {"wape": None, "accuracy": None, "n_val": 0}
            continue
        y_val = val_rows.pop("target")
        X_val = prepare_matrix(val_rows).reindex(columns=feature_columns, fill_value=0.0)
        model_preds = model.predict(X_val)
        # Blend with the naive persistence forecast (lag_1), weighted per
        # station by how severe its recent drift is — see
        # _station_naive_weights above — so this holdout score matches
        # what predict_targets() actually submits, not a pure-model number.
        row_weights = val_rows["station_id"].map(naive_weights).fillna(NAIVE_BLEND_BASE).to_numpy()
        preds = (1 - row_weights) * model_preds + row_weights * val_rows["lag_1"].to_numpy()
        # Métrica oficial: WAPE por estación, luego promediado — no agregado
        # sobre todas las estaciones (ver accuracy_monitor.py). Con demandas
        # muy distintas entre estaciones, agregar primero deja que las de
        # mayor demanda dominen el número; promediar por estación las pesa
        # por igual, igual que el leaderboard.
        errors = (y_val - preds).abs()
        station_wapes = []
        for station_id, station_actual in y_val.groupby(val_rows["station_id"]).sum().items():
            if station_actual <= 0:
                continue
            station_error = errors[val_rows["station_id"] == station_id].sum()
            station_wapes.append(float(station_error / station_actual))
        if not station_wapes:
            metrics_by_horizon[horizon_minutes] = {"wape": None, "accuracy": None, "n_val": len(val_rows)}
            continue
        wape = sum(station_wapes) / len(station_wapes)
        metrics_by_horizon[horizon_minutes] = {
            "wape": wape,
            "accuracy": max(0.0, 1 - wape),
            "n_val": len(val_rows),
        }

    assert feature_columns is not None
    return models, feature_columns, rows_by_horizon, metrics_by_horizon


def register_training_run(
    supabase: Client,
    *,
    execution_id: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    model_version: str,
    trained_at: datetime,
) -> tuple[str, str]:
    definition = {"numeric": list(NUMERIC_FEATURES), "categorical": list(CATEGORICAL_FEATURES)}
    definition_hash = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:16]
    supabase.table("feature_sets").upsert(
        {
            "feature_set_id": FEATURE_SET_ID,
            "version": FEATURE_SET_VERSION,
            "definition_hash": definition_hash,
            "definition": definition,
        },
        on_conflict="feature_set_id",
    ).execute()

    training_run_id = str(uuid.uuid4())
    supabase.table("training_runs").insert(
        {
            "training_run_id": training_run_id,
            "execution_id": execution_id,
            "dataset_version": DATASET_VERSION,
            "feature_set_id": FEATURE_SET_ID,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "validation_end": validation_end.isoformat(),
            "cutoff_at": validation_end.isoformat(),
            "code_commit": _git_commit(),
            "status": "succeeded",
        }
    ).execute()

    # Only one model_version may be active per algorithm (DB constraint):
    # retire whatever was active before registering the new one.
    supabase.table("model_versions").update({"is_active": False}).eq("algorithm", ALGORITHM).eq(
        "is_active", True
    ).execute()

    model_version_id = str(uuid.uuid4())
    supabase.table("model_versions").insert(
        {
            "model_version_id": model_version_id,
            "training_run_id": training_run_id,
            "algorithm": ALGORITHM,
            "hyperparameters": HYPERPARAMETERS,
            "artifact_uri": f"github:Mari4Z/pulso-transmi-sdk:models/pulso_hgb_poisson.joblib@{model_version}",
            "trained_at": trained_at.isoformat(),
            "is_active": True,
        }
    ).execute()
    return training_run_id, model_version_id


def record_validation_metrics(
    supabase: Client,
    *,
    model_version_id: str,
    metrics_by_horizon: dict[int, dict[str, float | int | None]],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    calculated_at: datetime,
) -> None:
    """Same metric_name convention as accuracy_monitor.py's live scoring
    (accuracy_{h}min / wape_{h}min) so these land in the same dashboard
    tiles — a holdout reading available immediately, instead of waiting on
    real predictions to resolve."""
    rows = []
    for horizon_minutes, values in metrics_by_horizon.items():
        if values["accuracy"] is None:
            continue
        for metric_name, value in (("accuracy", values["accuracy"]), ("wape", values["wape"])):
            rows.append(
                {
                    "model_version_id": model_version_id,
                    "metric_name": f"{metric_name}_{horizon_minutes}min",
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "metric_value": value,
                    "calculated_at": calculated_at.isoformat(),
                }
            )
    if rows:
        supabase.table("metrics").insert(rows).execute()


def main() -> None:
    print("Iniciando pipeline de reentrenamiento...")
    execution_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    supabase = _supabase_client()
    if supabase is None:
        print("Advertencia: variables de Supabase no definidas; no se registrará la ejecución.")
    else:
        supabase.table("pipeline_executions").insert(
            {"execution_id": execution_id, "started_at": started_at.isoformat(), "status": "running"}
        ).execute()

    try:
        # os.environ.get(...) or DEFAULT, not os.getenv's own default: the
        # Actions `vars.PULSO_API_URL` repo variable exists but is set to an
        # empty string when unconfigured, and getenv's default only kicks in
        # when the key is absent entirely.
        base_url = os.environ.get("PULSO_API_URL") or DEFAULT_BASE_URL
        with PulsoTransmiClient(base_url=base_url, timeout=60) as client:
            print("Descargando observaciones...")
            # Histórico estático + stream en vivo: sin el stream, el holdout
            # de validación (últimos 7 días de esto) queda anclado siempre
            # al mismo 2026-09-09, cada vez más lejos del momento real que
            # se está compitiendo.
            observations = client.all_observations_dataframe(page_size=5000)
            stations = client.stations()
            context = client.context_dataframe(page_size=5000)

        print("Construyendo features...")
        frame = build_features(observations, stations, context)

        # Últimos 7 días como validación (holdout temporal, sin partición aleatoria);
        # todo lo anterior entra a entrenamiento. Ver docs/experimentos-modelos.md.
        validation_end = frame["observed_at"].max()
        cutoff = validation_end - pd.Timedelta(7, unit="D")
        train_start = frame["observed_at"].min()
        train_end = cutoff - pd.Timedelta(1, unit="m")
        validation_start = cutoff

        print("Entrenando y evaluando modelos (horizontes 15/30/45/60 min)...")
        models, feature_columns, rows_by_horizon, metrics_by_horizon = train_and_evaluate(
            frame, cutoff, validation_end
        )
        for horizon_minutes, values in metrics_by_horizon.items():
            if values["accuracy"] is None:
                print(f"  {horizon_minutes}min: sin datos de validación")
            else:
                print(
                    f"  {horizon_minutes}min: accuracy={values['accuracy'] * 100:.2f}% "
                    f"wape={values['wape'] * 100:.2f}% (n={values['n_val']})"
                )

        model_version = "hgb-poisson-" + started_at.strftime("%Y%m%dT%H%M%SZ")
        MODEL_DIR.mkdir(exist_ok=True)
        package = {
            "model_version": model_version,
            "models": models,
            "feature_columns": feature_columns,
            "horizons": HORIZONS,
        }
        joblib.dump(package, MODEL_PATH, compress=3)

        triggered_by = os.environ.get("RETRAIN_TRIGGER", "manual")
        metadata = {
            "model_version": model_version,
            "model": "HistGradientBoostingRegressor",
            "loss": "poisson",
            "parameters": HYPERPARAMETERS,
            "horizons_minutes": [h * 15 for h in HORIZONS],
            "dataset": DATASET_VERSION,
            "training_cutoff": cutoff.isoformat(),
            "created_at": started_at.isoformat(),
            "feature_columns": feature_columns,
            "training_rows_by_horizon": rows_by_horizon,
            "validation_metrics": metrics_by_horizon,
            "random_state": HYPERPARAMETERS["random_state"],
            "triggered_by": triggered_by,
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Modelo {model_version} guardado en {MODEL_PATH}")

        # MLflow: registro local (mlruns/, committed a git junto al modelo —
        # ver pipeline.yml) en vez de un tracking server, para no montar
        # infraestructura aparte solo para esto. `mlflow ui` desde la raíz
        # del repo muestra el historial completo de reentrenamientos.
        # Best-effort: a stale absolute artifact_location in a committed
        # mlruns/ from a different machine (e.g. CI's /home/runner path,
        # reused when testing locally) must not block model registration —
        # same principle as the Supabase calls below.
        try:
            mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR.resolve()}")
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_name=model_version):
                mlflow.set_tags({"algorithm": ALGORITHM, "triggered_by": triggered_by, "dataset": DATASET_VERSION})
                mlflow.log_params(
                    {
                        **HYPERPARAMETERS,
                        "train_start": str(train_start),
                        "train_end": str(train_end),
                        "validation_start": str(validation_start),
                        "validation_end": str(validation_end),
                        "feature_set_id": FEATURE_SET_ID,
                    }
                )
                for horizon_minutes, values in metrics_by_horizon.items():
                    mlflow.log_metric(f"train_rows_{horizon_minutes}min", rows_by_horizon[str(horizon_minutes)])
                    if values["accuracy"] is not None:
                        mlflow.log_metric(f"accuracy_{horizon_minutes}min", values["accuracy"])
                        mlflow.log_metric(f"wape_{horizon_minutes}min", values["wape"])
                        mlflow.log_metric(f"n_val_{horizon_minutes}min", values["n_val"])
                mlflow.log_artifact(str(METADATA_PATH))
        except Exception as mlflow_exc:
            print(f"Advertencia: no se pudo registrar el experimento en MLflow: {mlflow_exc}")

        if supabase is not None:
            training_run_id, model_version_id = register_training_run(
                supabase,
                execution_id=execution_id,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                model_version=model_version,
                trained_at=started_at,
            )
            record_validation_metrics(
                supabase,
                model_version_id=model_version_id,
                metrics_by_horizon=metrics_by_horizon,
                window_start=validation_start,
                window_end=validation_end,
                calculated_at=started_at,
            )
            supabase.table("pipeline_executions").update(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "status": "succeeded",
                    "last_observed_at": validation_end.isoformat(),
                }
            ).eq("execution_id", execution_id).execute()

        print(json.dumps({"execution_id": execution_id, "model_version": model_version, "status": "succeeded"}))

    except Exception as exc:
        if supabase is not None:
            try:
                supabase.table("pipeline_executions").update(
                    {
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "status": "failed",
                        "error_message": str(exc)[:2000],
                    }
                ).eq("execution_id", execution_id).execute()
            except Exception as log_exc:  # pragma: no cover - best-effort logging
                print(f"No se pudo registrar el fallo en Supabase: {log_exc}")
        raise


if __name__ == "__main__":
    main()
