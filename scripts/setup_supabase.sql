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

-- Manager topology and operating configuration.
create table if not exists public.buildings (
    building_id text primary key,
    label text not null
);
create table if not exists public.floors (
    building_id text not null references public.buildings(building_id) on delete cascade,
    floor_id text not null,
    label text not null,
    primary key (building_id, floor_id)
);
create table if not exists public.zone_configs (
    zone_id text primary key,
    building_id text not null,
    floor_id text not null,
    area_m2 double precision not null,
    ceiling_height_m double precision not null,
    volume_m3 double precision not null,
    config jsonb not null,
    foreign key (building_id, floor_id) references public.floors(building_id, floor_id)
);
alter table public.zone_configs add column if not exists area_m2 double precision not null default 100;
alter table public.zone_configs add column if not exists ceiling_height_m double precision not null default 3;
alter table public.zone_configs add column if not exists volume_m3 double precision not null default 300;
create table if not exists public.auditor_config (
    config_key text primary key,
    config_value jsonb not null
);

-- PostgREST needs select/insert grants for the API role.
grant select, insert on public.telemetry to anon, authenticated;
grant select, insert on public.anomalies to anon, authenticated;
grant select, insert, update on public.buildings, public.floors, public.zone_configs, public.auditor_config to anon, authenticated;
grant usage, select on sequence public.telemetry_id_seq to anon, authenticated;
grant usage, select on sequence public.anomalies_id_seq to anon, authenticated;
