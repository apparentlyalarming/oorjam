# AI Energy-Waste Auditor for Commercial Buildings

A complete vertical slice: a synthetic telemetry generator feeds a sliding-window
feature pipeline, an Isolation-Forest detector plus an engineering ruleset tag
and monetises energy waste in near real time, and a FastAPI + WebSocket backend
drives a live dashboard and an injection simulator.  Persistence is optional and
pluggable: PostgREST (Supabase), raw Postgres, or in-memory.

```
TelemetryGenerator ─▶ FeatureEngine ─▶ IsolationForest ─▶ DiagnosticClassifier
       (3 zones)         (20 min window,        (score)       (4 root-cause rules)
                          10 s fast sub-window)
                              │
                              ▼
                    AnalyticsEngine (kWh + $, ROI-ranked measures)
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                  ▼
       RestAPI + WS     dashboard.html      simulator.html
   (REST + broadcast)   (Chart.js KPIs)   (inject faults)
```

## Feature set

- **Synthetic telemetry**: 3 zones (`floor_1_west`, `floor_2_east`,
  `floor_3_central`) emitting CO2 / humidity / indoor+outdoor temp / HVAC /
  lighting / plug-load at 1 Hz, following diurnal + weekend occupancy rhythms,
  gradual occupancy movement, volume-scaled CO2 mixing, and HVAC compressor
  duty cycles. The simulator clock and individual readings are controllable.
- **Topology and manager settings**: create buildings, floors, and zones from
  the dashboard; configure area, ceiling height, zone type, operating hours,
  and utility rates. Volume is derived from area × ceiling height and affects
  simulated CO2 behavior. Configuration uses Postgres/Supabase or memory
  storage when no database is configured.
- **Detection (Module 3)**: Isolation Forest scores a 20-minute sliding window
  (mean ‖ std feature vector); four deterministic rules resolve the root cause
  on a 10-second *fast* sub-window so a fault is flagged within seconds, not
  minutes.
- **Diagnoses** (dashboard shows red/orange/yellow markers):
  `DEMAND_VENTILATION_OVERCOOLING`, `UNOCCUPIED_LIGHTING_WASTE`,
  `HUMIDITY_ENVELOPE_LEAK`, `OFF_HOURS_BASELINE_DRIFT`.
- **Monetisation (Module 4)**: each closed anomaly accrues `kWh → $` at a
  configurable utility rate; the report ranks retrofit measures by payback then
  ROI and closes the loop from diagnosis to recommendation.
- **Injection simulator**: flip faults on/off per zone via the web UI or the
  bidirectional WebSocket control channel, adjust occupancy and sensor values,
  and watch the pipeline react. Unoccupied-lighting waste sends a simulated
  command to turn the zone lighting off on the next sample; both UIs log it.
- **Custom training**: upload historical CSV telemetry from the dashboard to
  retrain and hot-reload the Isolation Forest and expected-load models.
- **Persistence**: `TelemetryStore` abstraction; comes with PostgREST
  (Supabase), asyncpg Postgres, and a bounded in-memory store.

## Project layout

```
app/
  api/            FastAPI routers + WebSocket endpoints
  telemetry/      synthetic telemetry generator + fault injection controller
  features/       sliding-window + fast-sub-window feature engineering
  ml/             isolation forest detector, baseline regressors, rule engine, training
  analytics/      report engine (kWh/$/ROI) + recommendation catalogue
  storage/        TelemetryStore backends (Supabase PostgREST / Postgres / memory)
  web/static/     dashboard.html + simulator.html
  runtime.py      orchestrator wiring the whole pipeline
  main.py         uvicorn entry point (``uvicorn app.main:app``)
scripts/          train_model.py, run_server.py, setup_supabase.sql
tests/            pytest suite (unit + runtime integration)
models/           trained artifacts (detector.joblib, baseline_models.joblib)
```

## Quick start

Python 3.12 is recommended (3.10+ works). No system packages are required;
everything installs into a project-local virtualenv.

```bash
cd energy-auditor

# 1. Create the virtualenv and install runtime + test dependencies.
python3 -m venv --without-pip .venv        # only needed if ensurepip is missing
curl -sS https://bootstrap.pypa.io/get-pip.py | ./.venv/bin/python
./.venv/bin/pip install -r requirements.txt

# 2. Train the detection + baseline models (~10-20 s).
./.venv/bin/python scripts/train_model.py --days 7 --resolution 10

# 3. Run the server.
./.venv/bin/python scripts/run_server.py        # http://127.0.0.1:8000

# 4. Open the UI.
#    Dashboard:   http://127.0.0.1:8000/
#    Simulator:   http://127.0.0.1:8000/simulator
```

Tests:

```bash
./.venv/bin/pytest -q
```

## Persistence (optional)

By default the runtime keeps history in memory — fine for demos. Two persistent
options exist; both fall back to memory automatically if unavailable.

### Supabase (PostgREST)

Point the app at an existing Supabase project with the parent `.env`:

```
url=https://<project-ref>.supabase.co
secretkey=sb_secret_...
```

Run the bootstrap DDL once in the Supabase SQL editor (creates
`public.telemetry` / `public.anomalies` and PostgREST grants):

```sql
-- contents of scripts/setup_supabase.sql
```

`config.resolve_persistence()` maps the plain `url`/`secretkey` names onto the
`ENERGY_AUDITOR_SUPABASE_URL` / `ENERGY_AUDITOR_SUPABASE_KEY` settings at import
time, so no further configuration is needed.

### Postgres (asyncpg)

Set `ENERGY_AUDITOR_DATABASE_URL=postgresql://user:pass@host:5432/db`. The
store creates the schema itself on connect.

## Injection demo flow

1. Open **Dashboard** and note the three zones cycling between green markers.
2. Open **Simulator** in a second tab, pick a zone, hit an inject button.
3. Dashboard vertices turn red/yellow within a few seconds; the anomaly feed
   accumulates kWh and dollars; the report gains a ranked recommendation.
4. `Reset` (or `Reset all`) clears the zone's windows immediately and logs the
   accrued waste as a closed anomaly record.
5. Download the report via **Download report (md)** on the dashboard.

## Configuration

All settings live in `app/config.py` and can be overridden with
`ENERGY_AUDITOR_*` env vars or a `.env` file. Useful knobs:

| Variable | Default | Meaning |
|---|---|---|
| `ENERGY_AUDITOR_EMIT_INTERVAL_SECONDS` | `1.0` | sensor tick rate |
| `ENERGY_AUDITOR_WINDOW_MINUTES` | `20` | sliding analysis window |
| `ENERGY_AUDITOR_WINDOW_MIN_SAMPLES` | `15` | samples before ML inference |
| `ENERGY_AUDITOR_FAST_WINDOW_SECONDS` | `10.0` | rules' reactive sub-window |
| `ENERGY_AUDITOR_UTILITY_RATE_USD_PER_KWH` | `0.15` | blended utility rate |
| `ENERGY_AUDITOR_DATABASE_URL` | — | asyncpg connection string |
| `url` + `secretkey` (`.env`) | — | Supabase project credentials |

## API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | zones, detector + storage status |
| GET | `/api/zones`, `/api/status`, `/api/injection` | live zone summaries |
| GET/POST/PATCH | `/api/topology`, `/api/topology/zones` | browse and configure topology |
| PUT | `/api/config/utility-rate` | set building utility rate |
| POST | `/api/v1/train` | upload CSV and retrain local models |
| GET | `/api/telemetry/current` | latest scored point per zone |
| GET | `/api/anomalies` | closed anomaly records |
| GET | `/api/analytics/current` | JSON audit report |
| GET | `/api/report/markdown` | downloadable Markdown brief |
| GET | `/api/storage` | store kind + row counts |
| POST | `/api/inject` | inject / clear / reset a fault |
| WS | `/ws/telemetry?zone=&watch_all=true` | selected-zone history + all-zone alerts |
| WS | `/ws/simulator` | control channel + pipeline log |

For Supabase, run or rerun `scripts/setup_supabase.sql` to create or upgrade
the telemetry, anomaly, building, floor, zone configuration, and app
configuration tables before starting the app.
