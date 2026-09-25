import os
from supabase import create_client, Client
from pulso_transmi import PulsoTransmiClient
import pandas as pd

def main():
    print("Iniciando colector de datos...")
    
    # Extraer variables de entorno para Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: Las variables de entorno de Supabase no están definidas.")
        return

    # Inicializar cliente de Supabase
    supabase: Client = create_client(supabase_url, supabase_key)

    # Consumir API usando el SDK local
    print("Conectando a la API de Pulso TransMi...")
    with PulsoTransmiClient(timeout=60) as client:
        # Aquí puedes definir cuántos registros obtener o si filtrar por tiempo
        print("Descargando observaciones recientes...")
        observations = client.observations_dataframe(page_size=1000)
    
    if observations.empty:
        print("No hay datos para recolectar en este momento.")
        return

    # Preparar los datos para la inserción
    print(f"Obtenidas {len(observations)} observaciones. Preparando para Supabase...")
    
    # Asegurar que las fechas/timestamps se puedan serializar a JSON
    if "observed_at" in observations.columns:
        observations["observed_at"] = observations["observed_at"].dt.strftime('%Y-%m-%dT%H:%M:%S%z')

    # Reemplazar valores NaN por None para evitar errores de JSON en PostgreSQL
    observations = observations.where(pd.notnull(observations), None)
    
    # Convertir dataframe a lista de diccionarios
    records = observations.to_dict(orient="records")

    # Insertar los registros
    # NOTA: Asegúrate de que exista una tabla llamada "observations" en tu proyecto de Supabase
    # y de que los nombres de columna coincidan con las llaves del diccionario.
    print("Insertando datos en Supabase (tabla 'observations')...")
    try:
        # Se usa upsert para sobreescribir datos en caso de llaves duplicadas
        # Asegúrate de tener una Primary Key configurada en Supabase (ej. id)
        response = supabase.table("observations").upsert(records).execute()
        print("¡Inserción exitosa!")
    except Exception as e:
        print(f"Error al insertar en Supabase: {e}")

if __name__ == "__main__":
    main()

