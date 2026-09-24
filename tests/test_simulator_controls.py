from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.ml.rules import BaselinePredictor
from app.runtime import AuditorRuntime
from app.schemas import AnomalyKind, Verdict
from app.config import Settings
from app.telemetry.generator import InjectionController, TelemetryGenerator
from app.zones import ZONE_BY_ID, ZoneDef


async def _sink(*args, **kwargs):
    return None


def _generator(controller: InjectionController, seed: int = 17) -> TelemetryGenerator:
    return TelemetryGenerator(
        "bldg_01", ZONE_BY_ID["floor_2_east"], controller, _sink, 1.0, seed=seed
    )


def test_occupancy_override_changes_sensor_but_not_generated_power():
    baseline = _generator(InjectionController()).generate_point(
        datetime(2026, 9, 25, 14, tzinfo=timezone.utc)
    )
    controller = InjectionController()
    controller.set_occupancy("floor_2_east", 0)
    overridden = _generator(controller).generate_point(
        datetime(2026, 9, 25, 14, tzinfo=timezone.utc)
    )

    assert overridden.telemetry.occupancy_count == 0
    assert overridden.telemetry.co2_ppm < 500
    assert overridden.telemetry.hvac_kw == baseline.telemetry.hvac_kw
    assert overridden.telemetry.lighting_kw == baseline.telemetry.lighting_kw
    assert overridden.telemetry.plug_load_kw == baseline.telemetry.plug_load_kw


def test_sensor_override_and_actuator_apply_on_generated_frame():
    controller = InjectionController()
    controller.set_sensor_override("floor_2_east", "co2_ppm", 420.0)
    controller.set_sensor_override("floor_2_east", "humidity_percent", 78.0)
    controller.set_actuation("floor_2_east", "lighting_kw", 0.0)
    point = _generator(controller).generate_point(
        datetime(2026, 9, 25, 14, tzinfo=timezone.utc)
    )

    assert point.telemetry.co2_ppm == 420.0
    assert point.telemetry.humidity_percent == 78.0
    assert point.telemetry.lighting_kw == 0.0


def test_simulated_time_and_schedule_support_overnight_hours():
    controller = InjectionController()
    controller.set_time_of_day(23 * 60 + 30)
    point = _generator(controller).generate_point(
        datetime(2026, 9, 25, 14, tzinfo=timezone.utc)
    )
    zone = ZoneDef(
        zone_id="overnight", occupancy_max=10, lighting_peak_kw=2,
        plug_peak_kw=2, hvac_base_kw=2,
        operating_start="22:00", operating_end="06:00",
    )

    assert point.simulated_hour_fraction == 23.5
    assert zone.is_operating(23.5)
    assert zone.is_operating(4.0)
    assert not zone.is_operating(12.0)


def test_unoccupied_lighting_verdict_sends_actuator_command(settings):
    runtime = AuditorRuntime(settings)
    runtime.predictor = BaselinePredictor({})
    zone = ZONE_BY_ID["floor_1_west"]
    queue = runtime.subscribe()
    generator = TelemetryGenerator(
        "bldg_01", zone, runtime.controller, _sink, 1.0, seed=17
    )
    point = generator.generate_point(
        datetime(2026, 9, 25, 2, tzinfo=timezone.utc)
    )
    features = SimpleNamespace(
        timestamp=point.timestamp, hour_fraction=2.0, occupancy_mean=0.0,
        temp_outdoor_mean=20.0, temp_indoor_mean=22.0,
    )
    verdict = Verdict(
        zone_id=zone.zone_id, timestamp=point.timestamp, iforest_score=-0.2,
        is_anomaly=True, diagnosis="UNOCCUPIED_LIGHTING_WASTE", severity="high",
        marker="red",
    )

    runtime.add_verdict(zone, features, verdict, point)
    next_point = generator.generate_point(
        datetime(2026, 9, 25, 2, 0, 1, tzinfo=timezone.utc)
    )

    assert next_point.telemetry.lighting_kw == 0.0
    event = queue.get_nowait()
    assert event["type"] == "actuation"
    assert event["actuation"]["target"] == "lighting"
