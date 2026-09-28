-- Tracks our own row on the official leaderboard over time (both windows
-- the API exposes), so the dashboard's "posición en el leaderboard" bullet
-- doesn't need the private PULSO_API_KEY in the browser — a workflow with
-- that key writes here with service_role, and the dashboard just reads it
-- with anon, same pattern as model_drift_log.
create table if not exists public.leaderboard_log (
    id uuid primary key default gen_random_uuid(),
    measured_at timestamptz not null default now(),
    window_label text not null check (window_label in ('cumulative', 'rolling_24h')),
    accuracy double precision not null check (accuracy >= 0),
    raw_wape double precision,
    coverage double precision check (coverage between 0 and 1),
    rank integer check (rank > 0),
    participant_count integer check (participant_count > 0),
    resolved_cycles integer,
    created_at timestamptz not null default now()
);

create index if not exists leaderboard_log_measured_at_idx
    on public.leaderboard_log (window_label, measured_at desc);

alter table public.leaderboard_log enable row level security;

create policy "anon read leaderboard_log" on public.leaderboard_log
    for select to anon using (true);

-- "Última ejecución del pipeline" (student-project.md) needs to read
-- pipeline_executions; RLS was enabled on it with no policy in the earlier
-- migration (nothing needed it from the browser yet).
create policy "anon read pipeline_executions" on public.pipeline_executions
    for select to anon using (true);
