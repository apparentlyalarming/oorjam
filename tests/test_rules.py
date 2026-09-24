"""Diagnostic-classifier unit tests: the four rule scenarios → root-cause tags."""

from __future__ import annotations

import numpy as np

from app.ml.rules import DiagnosticClassifier
from app.features.engineering import WindowFeatures
from app.zones import ZONE_BY_ID


def _fv(zone, **overrides) -> WindowFeatures:
    base = dict(
        timestamp="2026-09-24T12:00:00Z",
        zone_id=zone.zone_id,
        X=np.zeros(10),
        feature_names=[],
        occupancy_mean=80.0,
        co2_mean=900.0,
        humidity_mean=55.0,
        temp_indoor_mean=22.0,
        temp_outdoor_mean=30.0,
        hvac_mean=30.0,
        lighting_mean=5.0,
        plug_mean=6.0,
        hour_fraction=14.0,
        occupancy_fast=80.0,
        co2_fast=900.0,
        humidity_fast=55.0,
        temp_indoor_fast=22.0,
        temp_outdoor_fast=30.0,
        hvac_fast=30.0,
        lighting_fast=5.0,
        plug_fast=6.0,
    )
    base.update(overrides)
    return WindowFeatures(**base)


def _classify(zone, fv, expected=None):
    clf = DiagnosticClassifier(ZONE_BY_ID)
    exp = expected or {"hvac": 30.0, "lighting": 5.0, "plug": 6.0}
    return clf.classify(zone, fv, exp, -0.35)


def test_humidity_envelope_leak(zone):
    v = _classify(zone, _fv(zone, humidity_fast=88.0, hvac_fast=46.0), {"hvac": 30.0})
    assert v.is_anomaly and v.diagnosis == "HUMIDITY_ENVELOPE_LEAK"
    assert v.marker == "red"


def test_unoccupied_lighting_waste(zone):
    v = _classify(
        zone,
        _fv(zone, occupancy_fast=0.0, lighting_fast=18.0, lighting_mean=18.0),
        {"lighting": 5.0},
    )
    assert v.diagnosis == "UNOCCUPIED_LIGHTING_WASTE"


def test_demand_ventilation_overcooling(zone):
    v = _classify(
        zone,
        _fv(zone, hvac_fast=52.0, co2_fast=410.0),
        {"hvac": 30.0},
    )
    assert v.diagnosis == "DEMAND_VENTILATION_OVERCOOLING"


def test_baseline_drift(zone):
    v = _classify(
        zone,
        _fv(zone, plug_fast=9.0, hour_fraction=2.0),   # 1.5x the expected 6.0 kW baseline off-hours
        {"hvac": 30.0, "lighting": 5.0, "plug": 6.0},
    )
    assert v.diagnosis == "OFF_HOURS_BASELINE_DRIFT"
    assert v.marker == "yellow"


def test_generic_fallback_when_no_rule_fires(zone):
    v = _classify(zone, _fv(zone), {"hvac": 30.0, "lighting": 5.0, "plug": 6.0})
    assert not v.is_anomaly or v.diagnosis == "GENERIC_ENERGY_ANOMALY"


def test_mechanical_spike_does_not_mislabel_as_drift(zone):
    v = _classify(
        zone,
        _fv(zone, plug_fast=9.0, hvac_fast=46.0),  # drift signal but hvac surging
        {"hvac": 30.0, "plug": 6.0},
    )
    assert v.diagnosis != "OFF_HOURS_BASELINE_DRIFT"
