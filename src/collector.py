import os

import pandas as pd
from supabase import Client, create_client

from pulso_transmi import PulsoTransmiClient
from pulso_transmi.client import DEFAULT_BASE_URL

# Must match the dataset_versions row the API currently serves (see
# GET /v1/meta -> dataset.dataset). observations.dataset_version is a
# required FK to that table.
DATASET_VERSION = "pulso-transmi-starter-v1"


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


def main():
    print("Iniciando colector de datos...")

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: Las variables de entorno de Supabase no están definidas.")
        return

    supabase: Client = create_client(supabase_url, supabase_key)

    # Solo traer lo nuevo desde la última observación ya guardada, en vez de
    # volver a bajar el histórico completo (45 días) en cada corrida.
    since = _latest_observed_at(supabase)
    if since:
        print(f"Última observación guardada: {since}. Descargando solo datos posteriores...")
    else:
        print("Sin observaciones previas; descargando histórico completo.")

    base_url = os.environ.get("PULSO_API_URL") or DEFAULT_BASE_URL
    with PulsoTransmiClient(base_url=base_url, timeout=60) as client:
        print("Conectando a la API de Pulso TransMi...")
        observations = client.observations_dataframe(start=since, page_size=5000)

    if observations.empty:
        print("No hay datos nuevos para recolectar en este momento.")
        return

    print(f"Obtenidas {len(observations)} observaciones. Preparando para Supabase...")

    if "observed_at" in observations.columns:
        observations["observed_at"] = observations["observed_at"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    observations["dataset_version"] = DATASET_VERSION
    observations = observations.where(pd.notnull(observations), None)
    records = observations.to_dict(orient="records")

    print(f"Insertando {len(records)} filas en Supabase (tabla 'observations')...")
    try:
        supabase.table("observations").upsert(records, on_conflict="station_id,observed_at").execute()
        print("¡Inserción exitosa!")
    except Exception as e:
        print(f"Error al insertar en Supabase: {e}")


if __name__ == "__main__":
    main()
