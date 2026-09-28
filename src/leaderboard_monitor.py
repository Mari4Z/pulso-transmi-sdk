"""Fetches our own row from the official leaderboard (both windows the API
exposes) and logs it to Supabase, so the dashboard can show "posición en el
leaderboard" (docs/student-project.md) without ever putting PULSO_API_KEY in
the browser — the anon key that page uses can only read this table (see
leaderboard_log RLS policy), never the real leaderboard endpoint.
"""
import os

import httpx
from supabase import Client, create_client

from pulso_transmi.client import DEFAULT_BASE_URL

# Matches the account name on the competition leaderboard.
DISPLAY_NAME = "María José Cortés Romero"
WINDOWS = ("cumulative", "rolling_24h")


def _supabase_client() -> Client | None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def fetch_leaderboard(client: httpx.Client, base_url: str, api_key: str, window_label: str) -> dict:
    response = client.get(
        f"{base_url.rstrip('/')}/v1/leaderboard",
        params={"window": window_label},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    print("Iniciando monitor de leaderboard...")
    api_key = os.environ.get("PULSO_API_KEY")
    if not api_key:
        print("PULSO_API_KEY no definido; no se puede consultar el leaderboard.")
        return
    base_url = os.environ.get("PULSO_API_URL") or DEFAULT_BASE_URL

    supabase = _supabase_client()
    rows = []
    with httpx.Client(timeout=30) as client:
        for window_label in WINDOWS:
            try:
                payload = fetch_leaderboard(client, base_url, api_key, window_label)
            except httpx.HTTPStatusError as exc:
                print(f"  {window_label}: error HTTP {exc.response.status_code}, se omite")
                continue
            entries = payload.get("data", [])
            mine = next((r for r in entries if r.get("display_name") == DISPLAY_NAME), None)
            if mine is None:
                print(f"  {window_label}: no se encontró '{DISPLAY_NAME}' en el leaderboard")
                continue
            print(
                f"  {window_label}: accuracy={mine['accuracy']:.2f}% rank={mine['rank']} "
                f"coverage={mine.get('coverage')}"
            )
            rows.append(
                {
                    "window_label": window_label,
                    "accuracy": mine["accuracy"],
                    "raw_wape": mine.get("raw_wape"),
                    "coverage": mine.get("coverage"),
                    "rank": mine["rank"],
                    "participant_count": payload.get("count"),
                    "resolved_cycles": payload.get("resolved_cycles"),
                }
            )

    if not rows:
        print("Nada que guardar.")
        return
    if supabase is None:
        print("Variables de Supabase no definidas; leaderboard no guardado.")
        return
    supabase.table("leaderboard_log").insert(rows).execute()
    print(f"{len(rows)} filas guardadas en Supabase (tabla 'leaderboard_log').")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Advertencia: el monitor de leaderboard falló. Error: {e}")
