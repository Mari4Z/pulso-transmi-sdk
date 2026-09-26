import os

import pandas as pd
from supabase import Client, create_client

from pulso_transmi import PulsoTransmiClient
from pulso_transmi.client import DEFAULT_BASE_URL

# Must match the dataset_versions row the API currently serves (see
# GET /v1/meta -> dataset.dataset). observations.dataset_version is a
# required FK to that table.
DATASET_VERSION = "pulso-transmi-starter-v1"


def _has_any_observations(supabase: Client) -> bool:
    result = supabase.table("observations").select("station_id").limit(1).execute()
    return bool(result.data)


def _upsert(supabase: Client, observations: pd.DataFrame) -> int:
    if observations.empty:
        return 0
    frame = observations.copy()
    frame["observed_at"] = frame["observed_at"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    frame["dataset_version"] = DATASET_VERSION
    frame = frame.where(pd.notnull(frame), None)
    records = frame.to_dict(orient="records")
    supabase.table("observations").upsert(records, on_conflict="station_id,observed_at").execute()
    return len(records)


def main():
    print("Iniciando colector de datos...")

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("Error: Las variables de entorno de Supabase no están definidas.")
        return

    supabase: Client = create_client(supabase_url, supabase_key)
    base_url = os.environ.get("PULSO_API_URL") or DEFAULT_BASE_URL

    with PulsoTransmiClient(base_url=base_url, timeout=60) as client:
        print("Conectando a la API de Pulso TransMi...")

        # /v1/observations only ever serves the fixed 45-day static window —
        # it never advances. Only /v1/stream/observations grows over time
        # (it's what actually reaches each forecast cycle's data_cutoff), so
        # that's what needs collecting on every run. The static window only
        # needs loading once, the first time this ever runs.
        if not _has_any_observations(supabase):
            print("Tabla vacía: cargando el histórico estático (45 días) una vez...")
            history = client.observations_dataframe(page_size=5000)
            n = _upsert(supabase, history)
            print(f"{n} filas históricas insertadas.")

        print("Descargando datos nuevos del stream en vivo...")
        stream = client.stream_observations_dataframe(page_size=5000)

    if stream.empty:
        print("No hay datos nuevos en el stream en este momento.")
        return

    n = _upsert(supabase, stream)
    print(f"{n} filas del stream insertadas/actualizadas en Supabase.")


if __name__ == "__main__":
    main()
