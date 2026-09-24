"""Runtime integration tests: full pipeline (without a server) detects faults
and turns the waste into closed, monetised anomaly records."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.runtime import AuditorRuntime
from app.schemas import AnomalyKind, SimulatorCommand
from app.telemetry.generator import TelemetryGenerator

FAULT_EXPECTED = {
    AnomalyKind.HVAC_OVERVENTILATION: "DEMAND_VENTILATION_OVERCOOLING",
    AnomalyKind.THERMAL_ENVELOPE_LEAK: "HUMIDITY_ENVELOPE_LEAK",
    AnomalyKind.UNOCCUPIED_LIGHTING: "UNOCCUPIED_LIGHTING_WASTE",
    AnomalyKind.EQUIPMENT_DRIFT: "OFF_HOURS_BASELINE_DRIFT",
}


async def _pump_fault(rt, zone, kind, samples: int = 40) -> list[str]:
    rt.controller.reset()
    rt.engines[zone.zone_id].reset()
    rt.controller.inject(zone.zone_id, kind)
    gen = TelemetryGenerator(
        "bldg_01", zone, rt.controller, sink=asyncio.sleep, interval=0.25, seed=3
    )
    base = datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc)
    diags: list[str] = []
    for i in range(samples):
        hf = 14.0 + i * 0.25 / 3600.0
        point = gen._normal_sample(base, hf)
        gen._apply_faults(point)
        point.injected_state = gen.controller.state_string(zone.zone_id)
        await rt._on_sample(zone, point)
        verdict = rt.current.get(zone.zone_id)
        if verdict is not None and verdict.diagnosis:
            diags.append(verdict.diagnosis)
    await asyncio.sleep(0)  # let fire-and-forget store tasks settle
    return diags


@pytest.mark.parametrize("kind,expected", list(FAULT_EXPECTED.items()))
async def test_every_fault_reaches_its_root_cause(settings, zone, kind, expected):
    rt = await AuditorRuntime.create(settings)
    try:
        diags = await _pump_fault(rt, zone, kind)
        assert expected in diags, f"{kind}: got {diags[-5:]}"
        # The generic flag may open the window; the specific tag must win.
        last = [d for d in diags if d == expected]
        assert last
    finally:
        await rt.shutdown()


async def test_reset_closes_anomaly_record_and_clears_verdict(settings, zone):
    rt = await AuditorRuntime.create(settings)
    try:
        await _pump_fault(rt, zone, AnomalyKind.UNOCCUPIED_LIGHTING, samples=60)
        assert rt.current[zone.zone_id].is_anomaly
        rt.apply_command(
            SimulatorCommand(action="reset", zone=zone.zone_id, anomaly=None)
        )
        assert not rt.current[zone.zone_id].is_anomaly
        assert rt.current[zone.zone_id].marker == "green"
        assert any(a.zone_id == zone.zone_id for a in rt.anomalies)
        closed = [a for a in rt.anomalies if a.zone_id == zone.zone_id]
        assert all(a.energy_wasted_kwh >= 0 for a in closed)
        assert all(a.sample_count >= 1 for a in closed)
    finally:
        await rt.shutdown()