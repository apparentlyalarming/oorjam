"""AI Energy-Waste Auditor for Commercial Buildings.

Package layout
--------------
app.config        : environment-driven application settings
app.schemas       : pydantic contracts for telemetry / injection / reports
app.zones         : static catalog of simulated building zones
app.telemetry     : Module 1 - IoT simulator & manual anomaly injection engine
app.features      : Module 2 - sliding-window feature engineering pipeline
app.ml            : Module 3 - Isolation Forest detector + diagnostic ruleset
app.analytics     : Module 4 - energy waste + ROI report engine
app.storage       : persistence layer (Postgres / Supabase / in-memory)
app.api           : FastAPI REST + WebSocket interfaces
app.runtime       : end-to-end orchestrator (generator -> pipeline -> UI)
"""

__version__ = "1.0.0"