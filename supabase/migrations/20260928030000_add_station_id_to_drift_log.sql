-- Per-station drift: the aggregate check in model_drift_log pools all 12
-- stations together, so a real, isolated shift at just one station (e.g.
-- Banderas/05100's ~47% demand collapse) gets diluted into a system-wide
-- reading of a few percent and never crosses the severe threshold. NULL
-- station_id keeps meaning "aggregate, all stations" (existing rows and
-- the system-wide check are unaffected); a populated station_id is one
-- station's own reading.
alter table public.model_drift_log
    add column if not exists station_id text references public.stations(station_id);

create index if not exists model_drift_log_station_idx
    on public.model_drift_log (station_id, measured_at desc);

-- Already RLS-enabled with an unconditional "anon read model_drift_log"
-- SELECT policy (20260925220000_enable_rls.sql) — a new column on an
-- already-permitted table needs no new policy.
