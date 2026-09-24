"""Shared fixtures: hermetic settings (no network in unit tests)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings() -> Settings:
    """Settings with persistence forced off (memory store), never resolves the
    parent .env so tests stay offline and deterministic."""
    return Settings(
        _env_file=[],  # ignore any local .env
        database_url=None,
        supabase_url=None,
        supabase_key=None,
        emit_interval_seconds=0.25,
        window_min_samples=15,
        window_minutes=20,
        fast_window_seconds=10.0,
        model_dir=PROJECT_ROOT / "models",
    )


@pytest.fixture
def zone() -> "ZoneDef":
    from app.zones import ZONE_BY_ID

    return ZONE_BY_ID["floor_2_east"]