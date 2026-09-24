"""Feature-engine unit tests: fast sub-window responds before the full window."""

from __future__ import annotations

from app.features.engineering import FeatureEngine
from app.schemas import Telemetry, TelemetryPoint


def _point(zone_id: str, occupancy: int, hvac: float, lighting: float, plug: float) -> TelemetryPoint:
    return TelemetryPoint(
        timestamp="2026-09-24T12:00:00Z",
        building_id="bldg_01",
        zone_id=zone_id,
        telemetry=Telemetry(
            occupancy_count=occupancy,
            co2_ppm=900.0,
            humidity_percent=55.0,
            temp_indoor_c=22.0,
            temp_outdoor_c=30.0,
            hvac_kw=hvac,
            lighting_kw=lighting,
            plug_load_kw=plug,
        ),
    )


def test_fast_window_precedes_full_window_mean():
    engine = FeatureEngine(window_points=100, min_samples=2, fast_window_points=10)
    # 50 normal samples fill more than the fast window...
    for _ in range(50):
        engine.update(_point("z", 50, 30.0, 5.0, 6.0))
    # ...then a 5x plug spike bursts in.
    for _ in range(4):
        fv = engine.update(_point("z", 50, 30.0, 5.0, 35.0))
    assert fv is not None
    # Full-window mean is barely moved after 4 spikes...
    assert fv.plug_mean < 12.0
    # ...but the fast window (last 10 samples: 6 normal + 4 spikes) is pulled
    # up well above the full-window average.
    assert fv.plug_fast > 15.0


def test_fast_window_tracks_last_k_samples(settings):
    engine = FeatureEngine(
        settings.window_points, min_samples=5, fast_window_points=4
    )
    values = [3.0, 3.0, 3.0, 10.0, 10.0]
    fv = None
    for v in values:
        fv = engine.update(_point("z", 1, 30.0, 5.0, v))
    # Last 4 of [3,3,3,10,10] => (3+3+10+10)/4 = 6.5
    assert fv is not None and fv.plug_fast == 6.5


def test_reset_drops_buffered_history():
    engine = FeatureEngine(window_points=20, min_samples=2, fast_window_points=4)
    for _ in range(20):
        engine.update(_point("z", 50, 30.0, 5.0, 6.0))
    engine.reset()
    assert engine.update(_point("z", 50, 30.0, 5.0, 6.0)) is None  # below min_samples