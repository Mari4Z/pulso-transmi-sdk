from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from supabase import Client, create_client

from pulso_transmi import PulsoTransmiClient
from pulso_transmi.client import DEFAULT_BASE_URL

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "pulso_hgb_poisson.joblib"
METADATA_PATH = MODEL_DIR / "pulso_hgb_poisson.json"

# Horizon steps in units of 15 minutes (1 -> 15min, 2 -> 30min, ...). The
# `models` dict in the joblib package is keyed by these step numbers, not by
# minutes: pulso_transmi/pipeline.py looks predictions up with
# `models[horizon_minutes // 15]`, so this keying must match on both sides.
HORIZONS = (1, 2, 3, 4)
DATASET_VERSION = "pulso-transmi-starter-v1"
ALGORITHM = "hgb-poisson"
FEATURE_SET_ID = "pulso-hgb-poisson-features"
FEATURE_SET_VERSION = "v1"

# Hyperparameters from the accepted experiment (docs/experimentos-modelos.md,
# exp-20260918-hgb-poisson-001) — the best of the benchmark, not a fresh guess.
HYPERPARAMETERS = {
    "max_iter": 300,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "random_state": 42,
}

NUMERIC_FEATURES = (
    "latitude", "longitude", "rain_mm", "rain_forecast",
    "temperature_c", "temperature_forecast", "event_intensity",
    "local_hour", "local_weekday", "is_weekend",
    "lag_1", "lag_4", "lag_16", "lag_96", "lag_672",
    "rolling_mean_4", "rolling_mean_16", "rolling_mean_96", "rolling_mean_672",
    "station_mean", "station_hour_mean", "station_weekday_hour_mean",
)
CATEGORICAL_FEATURES = ("station_id", "corridor")


def build_features(observations: pd.DataFrame, stations: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    frame = observations.merge(stations, on="station_id").merge(context, on="observed_at")
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
    return frame


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


def train_all_horizons(frame: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[dict, list[str], dict[str, int]]:
    models: dict[int, HistGradientBoostingRegressor] = {}
    feature_columns: list[str] | None = None
    rows_by_horizon: dict[str, int] = {}

    for horizon in HORIZONS:
        training = frame.copy()
        training["target_at"] = training["observed_at"] + pd.Timedelta(int(15 * horizon), unit="m")
        training["target"] = training.groupby("station_id")["demand"].shift(-horizon)
        eligible = training[training["target_at"] < cutoff].dropna(subset=["target"]).copy()
        eligible = add_training_statistics(eligible, eligible["target"], eligible)
        eligible = eligible.dropna(subset=list(NUMERIC_FEATURES)).copy()
        target = eligible.pop("target")
        matrix = prepare_matrix(eligible)
        feature_columns = matrix.columns.tolist()
        model = HistGradientBoostingRegressor(loss="poisson", **HYPERPARAMETERS)
        model.fit(matrix, target)
        models[horizon] = model
        rows_by_horizon[str(horizon * 15)] = len(eligible)

    assert feature_columns is not None
    return models, feature_columns, rows_by_horizon


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
) -> str:
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

    supabase.table("model_versions").insert(
        {
            "model_version_id": str(uuid.uuid4()),
            "training_run_id": training_run_id,
            "algorithm": ALGORITHM,
            "hyperparameters": HYPERPARAMETERS,
            "artifact_uri": f"github:Mari4Z/pulso-transmi-sdk:models/pulso_hgb_poisson.joblib@{model_version}",
            "trained_at": trained_at.isoformat(),
            "is_active": True,
        }
    ).execute()
    return training_run_id


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
            observations = client.observations_dataframe(page_size=5000)
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

        print("Entrenando modelos (horizontes 15/30/45/60 min)...")
        models, feature_columns, rows_by_horizon = train_all_horizons(frame, cutoff)

        model_version = "hgb-poisson-" + started_at.strftime("%Y%m%dT%H%M%SZ")
        MODEL_DIR.mkdir(exist_ok=True)
        package = {
            "model_version": model_version,
            "models": models,
            "feature_columns": feature_columns,
            "horizons": HORIZONS,
        }
        joblib.dump(package, MODEL_PATH, compress=3)

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
            "random_state": HYPERPARAMETERS["random_state"],
            "triggered_by": os.environ.get("RETRAIN_TRIGGER", "manual"),
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Modelo {model_version} guardado en {MODEL_PATH}")

        if supabase is not None:
            register_training_run(
                supabase,
                execution_id=execution_id,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                model_version=model_version,
                trained_at=started_at,
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
