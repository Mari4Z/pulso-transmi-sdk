create extension if not exists pgcrypto;

create table if not exists public.dataset_versions (
    dataset_version text primary key,
    dataset_name text not null,
    api_version text,
    source_generated_at date,
    history_start timestamptz,
    history_end timestamptz,
    frequency_minutes smallint not null default 15 check (frequency_minutes > 0),
    station_count integer check (station_count >= 0),
    observation_rows bigint check (observation_rows >= 0),
    context_rows bigint check (context_rows >= 0),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create table if not exists public.stations (
    station_id text primary key,
    station_name text not null,
    corridor text not null,
    latitude double precision not null check (latitude between -90 and 90),
    longitude double precision not null check (longitude between -180 and 180),
    created_at timestamptz not null default now()
);

create table if not exists public.context (
    observed_at timestamptz primary key,
    rain_mm double precision not null check (rain_mm >= 0),
    rain_forecast double precision not null check (rain_forecast >= 0),
    temperature_c double precision not null,
    temperature_forecast double precision not null,
    event_intensity double precision not null check (event_intensity >= 0),
    dataset_version text not null references public.dataset_versions(dataset_version),
    created_at timestamptz not null default now()
);

create table if not exists public.observations (
    station_id text not null references public.stations(station_id),
    observed_at timestamptz not null,
    demand integer not null check (demand >= 0),
    dataset_version text not null references public.dataset_versions(dataset_version),
    created_at timestamptz not null default now(),
    primary key (station_id, observed_at)
);

create table if not exists public.feature_sets (
    feature_set_id text primary key,
    version text not null,
    definition_hash text not null,
    definition jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (feature_set_id, version)
);

create table if not exists public.pipeline_executions (
    execution_id uuid primary key default gen_random_uuid(),
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    last_observed_at timestamptz,
    cursor text,
    status text not null default 'running' check (status in ('running', 'succeeded', 'failed', 'skipped')),
    error_message text,
    check (finished_at is null or finished_at >= started_at)
);

create table if not exists public.training_runs (
    training_run_id uuid primary key default gen_random_uuid(),
    execution_id uuid references public.pipeline_executions(execution_id),
    dataset_version text not null references public.dataset_versions(dataset_version),
    feature_set_id text not null references public.feature_sets(feature_set_id),
    train_start timestamptz not null,
    train_end timestamptz not null,
    validation_start timestamptz not null,
    validation_end timestamptz not null,
    cutoff_at timestamptz not null,
    code_commit text not null,
    status text not null default 'running' check (status in ('running', 'succeeded', 'failed')),
    created_at timestamptz not null default now(),
    check (train_start <= train_end),
    check (train_end < validation_start),
    check (validation_start <= validation_end),
    check (cutoff_at >= validation_end)
);

create table if not exists public.feature_vectors (
    feature_set_id text not null references public.feature_sets(feature_set_id),
    station_id text not null references public.stations(station_id),
    observed_at timestamptz not null,
    training_run_id uuid references public.training_runs(training_run_id),
    lag_1 double precision,
    lag_4 double precision,
    lag_96 double precision,
    rolling_mean_96 double precision,
    local_hour smallint check (local_hour between 0 and 23),
    local_weekday smallint check (local_weekday between 0 and 6),
    extra_features jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    primary key (feature_set_id, station_id, observed_at)
);

create table if not exists public.model_versions (
    model_version_id uuid primary key default gen_random_uuid(),
    training_run_id uuid not null references public.training_runs(training_run_id),
    algorithm text not null,
    hyperparameters jsonb not null default '{}'::jsonb,
    artifact_uri text not null,
    trained_at timestamptz not null default now(),
    is_active boolean not null default false,
    created_at timestamptz not null default now()
);

create unique index if not exists one_active_model_per_algorithm
    on public.model_versions (algorithm)
    where is_active;

create table if not exists public.predictions (
    prediction_id uuid primary key default gen_random_uuid(),
    model_version_id uuid not null references public.model_versions(model_version_id),
    execution_id uuid references public.pipeline_executions(execution_id),
    station_id text not null references public.stations(station_id),
    target_at timestamptz not null,
    created_at timestamptz not null default now(),
    horizon_steps smallint not null check (horizon_steps > 0),
    predicted_demand double precision not null check (predicted_demand >= 0),
    actual_demand integer check (actual_demand >= 0),
    absolute_error double precision generated always as (
        case when actual_demand is null then null else abs(actual_demand - predicted_demand) end
    ) stored,
    submission_status text not null default 'pending' check (submission_status in ('pending', 'sent', 'accepted', 'rejected', 'failed')),
    unique (model_version_id, station_id, target_at, horizon_steps)
);

create table if not exists public.metrics (
    metric_id uuid primary key default gen_random_uuid(),
    training_run_id uuid references public.training_runs(training_run_id),
    model_version_id uuid references public.model_versions(model_version_id),
    station_id text references public.stations(station_id),
    metric_name text not null,
    window_start timestamptz not null,
    window_end timestamptz not null,
    metric_value double precision not null,
    calculated_at timestamptz not null default now(),
    check (window_start <= window_end),
    check (training_run_id is not null or model_version_id is not null)
);

create table if not exists public.drift_signals (
    drift_signal_id uuid primary key default gen_random_uuid(),
    execution_id uuid not null references public.pipeline_executions(execution_id),
    feature_name text not null,
    drift_type text not null check (drift_type in ('data', 'concept', 'target')), 
    window_start timestamptz not null,
    window_end timestamptz not null,
    score double precision not null,
    threshold double precision not null,
    triggered_retraining boolean not null default false,
    created_at timestamptz not null default now(),
    check (window_start <= window_end)
);

create index if not exists observations_observed_at_idx on public.observations (observed_at);
create index if not exists observations_station_time_idx on public.observations (station_id, observed_at);
create index if not exists context_observed_at_idx on public.context (observed_at);
create index if not exists feature_vectors_station_time_idx on public.feature_vectors (station_id, observed_at);
create index if not exists predictions_target_at_idx on public.predictions (target_at);
create index if not exists predictions_execution_idx on public.predictions (execution_id);
create index if not exists metrics_window_idx on public.metrics (window_start, window_end);
create index if not exists drift_signals_execution_idx on public.drift_signals (execution_id);

create or replace view public.prediction_evaluation as
select
    p.prediction_id,
    p.model_version_id,
    p.station_id,
    p.target_at,
    p.horizon_steps,
    p.predicted_demand,
    p.actual_demand,
    p.absolute_error,
    case
        when p.actual_demand is null then null
        when p.actual_demand = 0 then null
        else p.absolute_error / p.actual_demand
    end as absolute_percentage_error
from public.predictions p;