from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from pulso_transmi import PulsoTransmiClient


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "pulso_hgb_poisson.joblib"
METADATA_PATH = MODEL_DIR / "pulso_hgb_poisson.json"
EXPERIMENT_ID = "exp-20260918-hgb-poisson-001"
MODEL_VERSION = "hgb-poisson-v1"
HORIZONS = (1, 2, 3, 4)
NUMERIC_FEATURES = (
    "latitude",
    "longitude",
    "rain_mm",
    "rain_forecast",
    "temperature_c",
    "temperature_forecast",
    "event_intensity",
    "local_hour",
    "local_weekday",
    "is_weekend",
    "lag_1",
    "lag_4",
    "lag_16",
    "lag_96",
    "lag_672",
    "rolling_mean_4",
    "rolling_mean_16",
    "rolling_mean_96",
    "rolling_mean_672",
    "station_mean",
    "station_hour_mean",
    "station_weekday_hour_mean",
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


def main() -> None:
    with PulsoTransmiClient(timeout=60) as client:
        metadata = client.meta()
        observations = client.observations_dataframe(page_size=5000)
        stations = client.stations()
        context = client.context_dataframe(page_size=5000)

    frame = build_features(observations, stations, context)
    cutoff = frame["observed_at"].max() - pd.Timedelta(7, unit="D")
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
        model = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(matrix, target)
        models[horizon] = model
        rows_by_horizon[str(horizon * 15)] = len(eligible)

    MODEL_DIR.mkdir(exist_ok=True)
    package = {
        "experiment_id": EXPERIMENT_ID,
        "model_version": MODEL_VERSION,
        "models": models,
        "feature_columns": feature_columns,
        "horizons": HORIZONS,
    }
    joblib.dump(
        package,
        MODEL_PATH,
        compress=3,
    )
    metadata_payload = {
        "experiment_id": EXPERIMENT_ID,
        "model_version": MODEL_VERSION,
        "model": "HistGradientBoostingRegressor",
        "loss": "poisson",
        "parameters": {
            "max_iter": 300,
            "learning_rate": 0.06,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "random_state": 42,
        },
        "horizons_minutes": [horizon * 15 for horizon in HORIZONS],
        "dataset": metadata["dataset"]["dataset"],
        "dataset_generated_at": metadata["dataset"]["generated_at"],
        "training_cutoff": cutoff.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "feature_columns": feature_columns,
        "training_rows_by_horizon": rows_by_horizon,
        "random_state": 42,
    }
    METADATA_PATH.write_text(json.dumps(metadata_payload, indent=2) + "\n", encoding="utf-8")
    print(f"modelo guardado: {MODEL_PATH}")
    print(f"metadatos guardados: {METADATA_PATH}")


if __name__ == "__main__":
    main()