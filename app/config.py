"""Application configuration.

All settings are loaded from environment variables (prefix ``ENERGY_AUDITOR_``)
and from an optional ``.env`` file. The Supabase fields mirror the variables
already present in the parent ``.env`` (``url`` and ``secretkey``); they are
re-read here so the auditor can persist to a hosted PostgREST backend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PARENT_DIR = _PROJECT_ROOT.parent


class Settings(BaseSettings):
    """Typed, env-driven configuration for the auditor."""

    model_config = SettingsConfigDict(
        env_prefix="ENERGY_AUDITOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # ``model_dir`` intentionally starts with ``model_``; the generated
        # defaults would otherwise collide with pydantic's protected namespace.
        protected_namespaces=(),
    )

    # ---- Hosting / web -----------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000

    # ---- Persistence -------------------------------------------------------
    # Postgres / Supabase connection details. If none are resolvable the app
    # falls back to a process-local in-memory store so the demo always runs.
    database_url: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    max_dashboard_history: int = 6000   # ring-buffer size for live WS replay
    max_anomaly_history: int = 2000     # ring-buffer size for closed anomalies

    # ---- Telemetry engine --------------------------------------------------
    building_id: str = "bldg_01"
    emit_interval_seconds: float = 1.0  # tick rate of the synthetic sensors

    # ---- Feature engineering (Module 2) ------------------------------------
    window_minutes: int = 20     # sliding analysis window duration
    window_min_samples: int = 15  # min samples before ML inference is allowed
    fast_window_seconds: float = 10.0  # short retroactive window for RULES

    # ---- ML tuning (Module 3) ----------------------------------------------
    contamination: float = 0.05          # expected anomaly share in training set
    random_state: int = 42
    isolation_forest_n_estimators: int = 200

    # ---- Model artifacts ---------------------------------------------------
    model_dir: Path = Path("models")

    # ---- Analytics (Module 4) ----------------------------------------------
    utility_rate_usd_per_kwh: float = 0.15  # blended commercial utility rate
    max_persist_points: int = 200_000       # in-memory retention cap

    # ---- Web UI ------------------------------------------------------------
    dashboard_history_points: int = 420  # samples replayed to a new dashboard ws

    @property
    def window_seconds(self) -> int:
        """Duration of the sliding feature window, in seconds."""
        return self.window_minutes * 60

    @property
    def window_points(self) -> int:
        """# of 1-second samples that fill the sliding window."""
        return int(self.window_seconds // self.emit_interval_seconds)

    @property
    def fast_window_points(self) -> int:
        """# of samples that fill the fast rule-evaluation window."""
        return max(2, int(self.fast_window_seconds // self.emit_interval_seconds))

    def resolve_persistence(self) -> "Settings":
        """Fill Supabase fields from the plain ``.env`` names if unset.

        The parent ``.env`` ships with ``url`` / ``secretkey`` keys; the
        settings class itself only knows the ``ENERGY_AUDITOR_`` prefix, so we
        map those plain names here when the prefixed ones are absent.
        """
        if not self.supabase_url:
            self.supabase_url = os.getenv("url") or os.getenv("SUPABASE_URL")
        if not self.supabase_key:
            self.supabase_key = os.getenv("secretkey") or os.getenv(
                "SUPABASE_SERVICE_ROLE_KEY"
            )
        return self


_env_file_candidates = [
    _PARENT_DIR / ".env",
    _PROJECT_ROOT / ".env",
    Path(".env"),
]
# Load the flat ``.env`` keys (``url`` / ``secretkey``) into the process
# environment so ``resolve_persistence`` can map them onto the prefixed fields.
for _p in _env_file_candidates:
    if _p.exists():
        load_dotenv(_p, override=False)
settings = Settings(
    _env_file=[p for p in _env_file_candidates if p.exists()],
).resolve_persistence()