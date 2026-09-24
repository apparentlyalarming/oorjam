from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from app.api.app import _parse_training_csv, create_app
from app.config import Settings


def test_csv_parser_reads_core_metrics():
    csv_text = (
        "timestamp,zone_id,occupancy_count,co2_ppm,humidity_percent,"
        "temp_indoor_c,temp_outdoor_c,hvac_kw,lighting_kw,plug_load_kw\n"
        "2026-09-25T10:00:00Z,floor_1_west,12,620,52,22,29,19,4,6\n"
    )
    points = _parse_training_csv(csv_text.encode())
    assert len(points) == 1
    assert points[0].telemetry.occupancy_count == 12
    assert points[0].telemetry.co2_ppm == 620


def test_csv_parser_rejects_missing_metrics():
    with pytest.raises(ValueError, match="missing CSV columns"):
        _parse_training_csv(b"timestamp,zone_id\n2026-09-25T10:00:00Z,floor_1_west\n")


def test_manager_and_retraining_routes_are_registered():
    @asynccontextmanager
    async def lifespan(_app):
        yield

    app = create_app(Settings(_env_file=[]), lifespan)
    paths = {route.path for route in app.routes}
    assert "/api/topology" in paths
    assert "/api/topology/zones" in paths
    assert "/api/v1/train" in paths
