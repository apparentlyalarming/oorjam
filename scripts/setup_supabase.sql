-- AI Energy-Waste Auditor · Supabase / Postgres schema
-- Run once in the Supabase SQL editor (or via psql) before enabling the
-- Supabase storage backend in the auditor.

create table if not exists public.telemetry (
    id           bigserial primary key,
    building_id  text not null,
    zone_id      text not null,
    ts           timestamptz not null,
    payload      jsonb not null,
    marker       text not null default 'green',
    score        double precision not null default 0
);
create index if not exists idx_telemetry_zone_ts
    on public.telemetry (zone_id, ts desc);

create table if not exists public.anomalies (
    id                 bigserial primary key,
    zone_id            text not null,
    start_ts           timestamptz not null,
    end_ts             timestamptz,
    diagnosis          text not null,
    severity           text not null,
    marker             text not null default 'green',
    energy_wasted_kwh  double precision not null default 0,
    financial_waste_usd double precision not null default 0,
    iforest_score      double precision not null default 0,
    details            jsonb not null default '{}'::jsonb
);
create index if not exists idx_anomalies_end on public.anomalies (end_ts desc);

-- PostgREST needs select/insert grants for the API role.
grant select, insert on public.telemetry to anon, authenticated;
grant select, insert on public.anomalies to anon, authenticated;
grant usage, select on sequence public.telemetry_id_seq to anon, authenticated;
grant usage, select on sequence public.anomalies_id_seq to anon, authenticated;