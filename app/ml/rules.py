"""Diagnostic Classifier Ruleset (Module 3 inference stage).

When the Isolation Forest (or a strong physical signature) flags a window, the
ruleset maps high-dimensional signals onto a root-cause tag.  Each rule encodes
the *engineering* reason a pattern is wasteful, using the window aggregates the
feature pipeline already computed:

* UNOCCUPIED_LIGHTING_WASTE        : empty zone lit at near-nameplate level.
* DEMAND_VENTILATION_OVERCOOLING   : HVAC ran hard while CO2 sat at outdoor
                                     baseline (air exchanged for nobody).
* HUMIDITY_ENVELOPE_LEAK           : humidity spike + HVAC spike exposes latent
                                     envelope infiltration / condensation duty.
* OFF_HOURS_BASELINE_DRIFT         : plug load persistently above the baseline
                                     model while mechanical duty stays steady
                                     (parasitic / zombie loads).

Rules are evaluated on the FAST sub-window aggregates (see
``WindowFeatures.*_fast``) so an injected fault flips the marker almost
immediately; the Isolation Forest still scores the full sliding window.

``PRIMARY_METRIC`` ties each tag to the consumption meter that carries its waste
so the analytics engine can quantify kWh and dollars cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from ..features.engineering import WindowFeatures
from ..schemas import Verdict
from ..zones import ZoneDef

#: Marker colour per severity, mapping directly onto the dashboard legend.
MARKER_COLORS: Dict[str, str] = {
    "normal": "green",
    "high": "red",
    "moderate": "orange",
    "subtle": "yellow",
}
#: Hex colours used by the Chart.js rendering layer.
MARKER_HEX: Dict[str, str] = {
    "green": "#22c55e",
    "red": "#ef4444",
    "orange": "#f97316",
    "yellow": "#eab308",
}

#: The consumption meter that monetises each fault class.
PRIMARY_METRIC: Dict[str, str] = {
    "UNOCCUPIED_LIGHTING_WASTE": "lighting_kw",
    "DEMAND_VENTILATION_OVERCOOLING": "hvac_kw",
    "HUMIDITY_ENVELOPE_LEAK": "hvac_kw",
    "OFF_HOURS_BASELINE_DRIFT": "plug_load_kw",
}

#: Rule thresholds (tuned while validating against the simulated fault sets).
HVAC_SPIKE_RATIO = 1.45          # hvac vs baseline-model expectation
LEAK_HUMIDITY_PCT = 72.0         # zone-average relative humidity
LEAK_HVAC_RATIO = 1.30           # hvac vs baseline model for leak diagnosis
CO2_STALE_PPM = 460.0            # CO2 near outdoor baseline
OCCUPIED_PERSON = 3.0            # below this the zone counts as empty
LIGHTING_EMPTY_FRACTION = 0.55   # share of nameplate lighting treated as waste
DRIFT_PLUG_RATIO = 1.15          # plug vs baseline-model expectation
MECHANICAL_STEADY_RATIO = 1.25   # hvac below this => load is not mechanical


class BaselinePredictor:
    """Expected-consumption oracle backed by the fitted baseline regressors."""

    def __init__(self, models: Dict[str, object]) -> None:
        self.models = models  # zone_id -> {metric: regressor}

    def predict(self, zone_id: str, metric: str, fv: WindowFeatures) -> float:
        """Predict expected kW for ``metric`` from the window aggregates.

        Uses the exact design matrix the baseline models were trained on:
        cyclic hour, mean occupancy and (for HVAC) the mean temperature
        gradient - so train time and inference live in the same feature space.
        """
        model = self.models.get(zone_id, {}).get(metric)
        if model is None:
            return 0.0  # pragma: no cover - defensive
        sin_h = math.sin(2.0 * math.pi * fv.hour_fraction / 24.0)
        cos_h = math.cos(2.0 * math.pi * fv.hour_fraction / 24.0)
        occ = fv.occupancy_mean
        if metric == "hvac":
            dT = max(0.0, fv.temp_outdoor_mean - fv.temp_indoor_mean)
            x = [[sin_h, cos_h, occ, dT]]
        else:
            x = [[sin_h, cos_h, occ]]
        return float(model.predict(x)[0])

    def predicted_kws(self, zone_id: str, fv: WindowFeatures) -> Dict[str, float]:
        """Expected kW for hvac / lighting / plug for one window."""
        return {
            metric: self.predict(zone_id, metric, fv)
            for metric in ("hvac", "lighting", "plug")
        }


class DiagnosticClassifier:
    """Assigns a root-cause tag + severity + marker to each analysed window."""

    def __init__(self, zones_by_id: Dict[str, ZoneDef]) -> None:
        self.zones = zones_by_id

    def classify(
        self,
        zone: ZoneDef,
        fv: WindowFeatures,
        expected: Dict[str, float],
        iforest_score: float,
    ) -> Verdict:
        """Run the ordered ruleset.  Returns a verdict for the current window.

        ``expected`` holds the baseline-model kW for hvac / lighting / plug.
        """
        # --- signal-to-baseline ratios using the FAST sub-window, so a fault
        #     injected seconds ago is judged against the expected baseline the
        #     moment it arrives (the iForest sees the full 15-30 min window).
        hvac_ratio = fv.hvac_fast / max(1e-3, expected.get("hvac", 0.0))
        plug_ratio = fv.plug_fast / max(1e-3, expected.get("plug", 0.0))

        # Strong physical signatures make the ML flag redundant; the ruleset
        # carries the final tagging decision.
        fired: Optional[tuple[str, str, str, float, str]] = None

        # 1) Humidity envelope leak first - it is the most energy-expensive.
        if fv.humidity_fast >= LEAK_HUMIDITY_PCT and hvac_ratio >= LEAK_HVAC_RATIO:
            fired = (
                "HUMIDITY_ENVELOPE_LEAK",
                "high",
                MARKER_COLORS["high"],
                min(1.0, hvac_ratio / 2.0),
                f"zone humidity {fv.humidity_fast:.1f}% with HVAC {hvac_ratio:.2f}x expected",
            )
        # 2) Unoccupied lighting - empty space at near-nameplate light load.
        elif fv.occupancy_fast < OCCUPIED_PERSON and fv.lighting_fast > LIGHTING_EMPTY_FRACTION * zone.lighting_peak_kw:
            fired = (
                "UNOCCUPIED_LIGHTING_WASTE",
                "high",
                MARKER_COLORS["high"],
                min(1.0, fv.lighting_fast / zone.lighting_peak_kw),
                f"zone empty but lit at {fv.lighting_fast:.1f} kW ({zone.lighting_peak_kw} kW peak)",
            )
        # 3) Demand-ventilation overcooling - high HVAC, stale-low CO2.
        elif hvac_ratio >= HVAC_SPIKE_RATIO and fv.co2_fast < CO2_STALE_PPM:
            fired = (
                "DEMAND_VENTILATION_OVERCOOLING",
                "moderate",
                MARKER_COLORS["moderate"],
                min(1.0, hvac_ratio / 2.0),
                f"HVAC at {hvac_ratio:.2f}x expected while CO2 only {fv.co2_fast:.0f} ppm",
            )
        # 4) Baseline drift - persistent parasitic plug load while mechanical
        #    duty stays steady (a stubborn elevated baseline, day or night).
        elif plug_ratio >= DRIFT_PLUG_RATIO and hvac_ratio < MECHANICAL_STEADY_RATIO:
            fired = (
                "OFF_HOURS_BASELINE_DRIFT",
                "subtle",
                MARKER_COLORS["subtle"],
                0.6,
                f"baseline plug load {plug_ratio:.2f}x model with steady HVAC ({hvac_ratio:.2f}x)",
            )

        if fired is not None:
            tag, severity, marker, confidence, why = fired
            return Verdict(
                zone_id=zone.zone_id,
                timestamp=fv.timestamp,
                iforest_score=round(iforest_score, 6),
                is_anomaly=True,
                diagnosis=tag,
                severity=severity,
                marker=marker,
                confidence=round(confidence, 4),
                explanation=why,
            )

        # No rule fired: defer to the ML model alone.
        if iforest_score < 0.0:  # below zero -> structurally unusual
            return Verdict(
                zone_id=zone.zone_id,
                timestamp=fv.timestamp,
                iforest_score=round(iforest_score, 6),
                is_anomaly=True,
                diagnosis="GENERIC_ENERGY_ANOMALY",
                severity="moderate",
                marker=MARKER_COLORS["moderate"],
                confidence=0.5,
                explanation=f"isolation-forest score {iforest_score:.3f} outside baseline envelope",
            )

        return Verdict(
            zone_id=zone.zone_id,
            timestamp=fv.timestamp,
            iforest_score=round(iforest_score, 6),
            is_anomaly=False,
            diagnosis=None,
            severity="normal",
            marker=MARKER_COLORS["normal"],
            confidence=0.0,
            explanation="within normal operating envelope",
        )