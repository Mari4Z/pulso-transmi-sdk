-- RLS was off on every table: with no policies, Postgres lets any role read
-- and write through PostgREST once a valid API key is presented — including
-- the anon key, which is the only key safe to ship in a public dashboard.
-- Enabling RLS makes every table default-deny; the SELECT policies below are
-- the only access anon gets, and only on what a public dashboard needs.
-- Service-role writes (GitHub Actions) are unaffected: that role bypasses
-- RLS entirely, by Postgres design, regardless of policies.

alter table public.dataset_versions enable row level security;
alter table public.stations enable row level security;
alter table public.context enable row level security;
alter table public.observations enable row level security;
alter table public.feature_sets enable row level security;
alter table public.pipeline_executions enable row level security;
alter table public.training_runs enable row level security;
alter table public.feature_vectors enable row level security;
alter table public.model_versions enable row level security;
alter table public.predictions enable row level security;
alter table public.metrics enable row level security;
alter table public.drift_signals enable row level security;
alter table public.model_drift_log enable row level security;

-- Read-only, for the accuracy/drift dashboard.
create policy "anon read stations" on public.stations
    for select to anon using (true);

create policy "anon read model_versions" on public.model_versions
    for select to anon using (true);

create policy "anon read predictions" on public.predictions
    for select to anon using (true);

create policy "anon read metrics" on public.metrics
    for select to anon using (true);

create policy "anon read model_drift_log" on public.model_drift_log
    for select to anon using (true);

-- Everything else (raw observations/context, training internals,
-- execution/pipeline bookkeeping) has RLS on and no policy: anon gets
-- nothing, service_role is untouched.
