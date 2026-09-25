import os
import joblib
import pandas as pd
from datetime import datetime, timezone
from supabase import create_client, Client
from pulso_transmi import PulsoTransmiClient
from pipeline import MODEL_PATH, build_features, add_training_statistics, prepare_matrix

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

    print("Descargando datos recientes para features...")
    with PulsoTransmiClient(timeout=60) as client:
        observations = client.observations_dataframe(page_size=5000)
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
        
        # Guardar en lista
        for i, (idx, row) in enumerate(latest_obs.iterrows()):
            predictions_to_insert.append({
                "station_id": row["station_id"],
                "prediction_at": str(datetime.now(timezone.utc)),
                "target_at": str(row["observed_at"] + pd.Timedelta(int(15 * horizon), unit="m")),
                "horizon": horizon,
                "predicted_demand": float(preds[i])
            })

    # Guardar en Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
    if supabase_url and supabase_key:
        print(f"Insertando {len(predictions_to_insert)} predicciones en Supabase...")
        try:
            supabase = create_client(supabase_url, supabase_key)
            supabase.table("predictions").insert(predictions_to_insert).execute()
            print("Predicciones insertadas con éxito.")
        except Exception as e:
            print(f"Error al insertar predicciones: {e}")
    else:
        print("Variables de Supabase no definidas, predicciones no guardadas.")

if __name__ == "__main__":
    main()
