-- Standalone drift log: intentionally not tied to the training_runs /
-- pipeline_executions FK chain (nothing currently populates that chain),
-- so the hourly drift monitor can write here directly.
create table if not exists public.model_drift_log (
    id uuid primary key default gen_random_uuid(),
    measured_at timestamptz not null default now(),
    feature text not null,
    ks_stat double precision not null check (ks_stat >= 0),
    p_value double precision not null check (p_value between 0 and 1),
    drift_detected boolean not null,
    historical_mean double precision,
    recent_mean double precision,
    mean_change_pct double precision,
    historical_std double precision,
    recent_std double precision,
    std_change_pct double precision,
    recent_samples integer not null check (recent_samples > 0),
    historical_samples integer not null check (historical_samples > 0),
    created_at timestamptz not null default now()
);

create index if not exists model_drift_log_measured_at_idx
    on public.model_drift_log (measured_at desc);

create index if not exists model_drift_log_feature_idx
    on public.model_drift_log (feature, measured_at desc);
