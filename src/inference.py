import os

import joblib
import pandas as pd
from supabase import Client, create_client

from pipeline import ALGORITHM, MODEL_PATH, add_training_statistics, build_features, prepare_matrix
from pulso_transmi import PulsoTransmiClient
from pulso_transmi.client import DEFAULT_BASE_URL


def _active_model_version_id(supabase: Client) -> str | None:
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


def main():
    print("Iniciando inferencia...")
    if not MODEL_PATH.exists():
        print(f"No se encontró el modelo en {MODEL_PATH}. Debes entrenarlo primero.")
        return

    print("Cargando modelo...")
    package = joblib.load(MODEL_PATH)
    models = package["models"]
    feature_columns = package["feature_columns"]
    horizons = package["horizons"]

    base_url = os.environ.get("PULSO_API_URL") or DEFAULT_BASE_URL
    print("Descargando datos recientes para features...")
    with PulsoTransmiClient(base_url=base_url, timeout=60) as client:
        # Estático + stream en vivo: solo el estático deja "latest_obs" pegado
        # siempre al 2026-09-09, sin importar cuándo corra este script.
        observations = client.all_observations_dataframe(page_size=5000)
        stations = client.stations()
        context = client.context_dataframe(page_size=5000)

    print("Construyendo features...")
    frame = build_features(observations, stations, context)

    # Recrear statistics como en el pipeline (necesitamos histórico para station_mean etc)
    cutoff = frame["observed_at"].max() - pd.Timedelta(7, unit="D")

    predictions_to_insert = []

    for horizon in horizons:
        # Calcular target temporal para estadísticas (como en pipeline)
        training = frame.copy()
        training["target_at"] = training["observed_at"] + pd.Timedelta(int(15 * horizon), unit="m")
        training["target"] = training.groupby("station_id")["demand"].shift(-horizon)

        # Historico para estadísticas
        train_data = training[training["target_at"] < cutoff].dropna(subset=["target"]).copy()

        # Queremos predecir SOLO la última observación de cada estación
        latest_obs = frame.sort_values("observed_at").groupby("station_id").tail(1).copy()

        # Añadir estadísticas usando el histórico
        latest_obs = add_training_statistics(train_data, train_data["target"], latest_obs)

        # Preparar matriz de inferencia
        matrix = prepare_matrix(latest_obs)

        # Rellenar columnas faltantes o quitar sobrantes (para match exacto con features_columns)
        for col in feature_columns:
            if col not in matrix.columns:
                matrix[col] = 0
        matrix = matrix[feature_columns]

        # Manejar NaNs temporalmente (imputar con 0 o media) para evitar error predict
        matrix = matrix.fillna(0)

        # Predecir
        preds = models[horizon].predict(matrix)

        # Guardar en lista — mismas columnas que predictions (ver
        # pulso_transmi.pipeline.record_predictions): horizon_steps, no
        # "horizon"; sin prediction_at, que no existe en la tabla.
        for i, (idx, row) in enumerate(latest_obs.iterrows()):
            predictions_to_insert.append(
                {
                    "station_id": row["station_id"],
                    "target_at": str(row["observed_at"] + pd.Timedelta(int(15 * horizon), unit="m")),
                    "horizon_steps": horizon,
                    "predicted_demand": max(0.0, float(preds[i])),
                }
            )

    # Guardar en Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("Variables de Supabase no definidas, predicciones no guardadas.")
        return

    supabase = create_client(supabase_url, supabase_key)
    model_version_id = _active_model_version_id(supabase)
    if not model_version_id:
        print("No hay un model_versions activo en Supabase; predicciones no guardadas.")
        return
    for row in predictions_to_insert:
        row["model_version_id"] = model_version_id

    print(f"Insertando {len(predictions_to_insert)} predicciones en Supabase...")
    try:
        supabase.table("predictions").upsert(
            predictions_to_insert, on_conflict="model_version_id,station_id,target_at,horizon_steps"
        ).execute()
        print("Predicciones insertadas con éxito.")
    except Exception as e:
        print(f"Error al insertar predicciones: {e}")


if __name__ == "__main__":
    main()
