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


class PipelineError(RuntimeError):
    pass


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
            demand_by_station.shift(1)
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
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
    predictions: list[dict[str, Any]] = []
    for target in targets:
        station_id = str(target["station_id"])
        horizon = int(target["horizon_minutes"])
        row = latest[latest["station_id"] == station_id]
        if row.empty or horizon not in HORIZONS:
            raise PipelineError(f"cannot predict station={station_id} horizon={horizon}")
        matrix = _prepare_matrix(row, feature_columns)
        value = float(models[horizon // 15].predict(matrix)[0])
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
    if response.status_code not in (200, 201):
        raise PipelineError(f"POST /v1/submissions failed with HTTP {response.status_code}: {response.text[:500]}")
    return {"idempotency_key": idempotency_key, "response": response.json()}


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
        cycle = get_current_cycle(base_url, api_key, http_client)
        if cycle is None:
            print("no open forecast cycle")
            return
        with PulsoTransmiClient(base_url=base_url, api_key=api_key, timeout=60) as client:
            observations = client.observations_dataframe(page_size=5000)
            stations = client.stations()
            context = client.context_dataframe(page_size=5000)
        predictions = predict_targets(cycle, observations, stations, context, package)
        result = submit(base_url, api_key, cycle, model_metadata, predictions, http_client)
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
