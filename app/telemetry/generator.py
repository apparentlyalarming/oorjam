"""Module 1 backend: synthetic telemetry engine + manual fault injection.

Design notes
------------
* Each zone runs its own ``TelemetryGenerator`` ``asyncio`` task.
* ``diurnal_factor()`` models the building's daily occupancy rhythm as a cosine
  curve that is 0.0 at 03:00 (night-time baseline) and 1.0 at 15:00 (peak). The
  cosine form keeps derivatives continuous, so telemetry ramps smoothly.
* Gaussian noise is added to every sensor to make the stream realistic; the ML
  detector must therefore learn to separate *signal* from *noise*.
* Manual faults are applied AFTER the normal sample is synthesised, mutating
  only the fields that describe that fault signature (see each transform).
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, Set

from ..schemas import AnomalyKind, Telemetry, TelemetryPoint
from ..zones import ZoneDef

#: agnostic type alias for the per-sample sink the caller attaches.
SampleCallback = Callable[[ZoneDef, TelemetryPoint], Awaitable[None]]


class InjectionController:
    """Holds the set of active manual faults, keyed by zone.

    ``equipment_drift`` is meant to persist as a subtle step change, while the
    other faults are transient, so the panel can independently toggle any fault
    on a chosen zone; ``reset`` clears everything.
    """

    def __init__(self) -> None:
        self._active: Dict[str, Set[str]] = defaultdict(set)
        self._occupancy_override: Dict[str, int] = {}
        self._sensor_overrides: Dict[str, Dict[str, float]] = defaultdict(dict)

    def set_occupancy(self, zone: str, occupancy: int) -> None:
        self._occupancy_override[zone] = max(0, occupancy)

    def occupancy_override(self, zone: str) -> int | None:
        return self._occupancy_override.get(zone)

    def clear_occupancy(self, zone: str) -> None:
        self._occupancy_override.pop(zone, None)

    def set_sensor_override(self, zone: str, metric: str, value: float) -> None:
        self._sensor_overrides[zone][metric] = value

    def clear_sensor_override(self, zone: str, metric: str) -> None:
        self._sensor_overrides.get(zone, {}).pop(metric, None)

    def sensor_overrides(self, zone: str) -> Dict[str, float]:
        return dict(self._sensor_overrides.get(zone, {}))

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        return {
            zone: {
                "active": sorted(active),
                "injected_state": self.state_string(zone),
            }
            for zone, active in sorted(self._active.items())
        }

    def inject(self, zone: str, kind: AnomalyKind) -> None:
        self._active[zone].add(kind.value)

    def clear(self, zone: str, kind: AnomalyKind) -> None:
        self._active[zone].discard(kind.value)

    def clear_all(self, zone: str) -> None:
        self._active.pop(zone, None)
        self._occupancy_override.pop(zone, None)
        self._sensor_overrides.pop(zone, None)

    def reset(self) -> None:
        self._active.clear()
        self._occupancy_override.clear()
        self._sensor_overrides.clear()

    def is_active(self, zone: str, kind: AnomalyKind) -> bool:
        return kind.value in self._active.get(zone, ())

    def active_set(self, zone: str) -> Set[str]:
        return set(self._active.get(zone, ()))

    def state_string(self, zone: str) -> str:
        """Wire format of ``injected_state``: ``"normal"`` or comma-separated tags."""
        tags = sorted(self._active.get(zone, ()))
        return "normal" if not tags else ",".join(tags)


def diurnal_factor(hour_fraction: float) -> float:
    """Normalised occupancy rhythm in [0, 1].

    ``1 - cos(2*pi*(h - 3)/24)`` gives 0.0 at 03:00 and 1.0 at 15:00.  Adding
    1 and halving maps the cosine range [-1, 1] onto [0, 1].
    """
    return (1.0 - math.cos(2.0 * math.pi * (hour_fraction - 3.0) / 24.0)) / 2.0


def _now_iso() -> str:
    """UTC ISO-8601 timestamp with ``Z`` suffix, per the payload schema."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TelemetryGenerator:
    """Synthesises one normal telemetry sample per tick, then applies faults."""

    def __init__(
        self,
        building_id: str,
        zone: ZoneDef,
        controller: InjectionController,
        sink: SampleCallback,
        interval: float,
        seed: int | None = None,
    ) -> None:
        self.building_id = building_id
        self.zone = zone
        self.controller = controller
        self.sink = sink
        self.interval = interval
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ synth
    def _normal_sample(self, now: datetime, hour_fraction: float) -> TelemetryPoint:
        """Build a physically-plausible *normal* sample for the zone.

        Occupancy follows the diurnal rhythm (weekend-shaded by 25%), and each
        dependent sensor is a deterministic function of occupancy / thermal
        forcing plus Gaussian noise:

        * CO2   = outdoor baseline + occupancy contribution
        * HVAC  = circulation base + cooling load (outdoor-indoor) + occupancy
        * lights = off-hours standby + per-occupant demand
        * plug   = always-on baseline + per-occupant equipment
        """
        z = self.zone
        rng = self._rng

        weekend_scale = 0.75 if now.weekday() >= 5 else 1.0
        # Diurnal rhythm multiplied by small shot noise keeps peaks variable.
        factor = max(0.0, diurnal_factor(hour_fraction) * weekend_scale) * rng.uniform(0.95, 1.05)

        occupancy = max(0, min(z.occupancy_max, int(z.occupancy_max * factor + rng.gauss(0.0, z.occupancy_max * 0.04))))
        load_occupancy = occupancy
        occupancy_override = self.controller.occupancy_override(z.zone_id)
        if occupancy_override is not None:
            # Override only the occupancy sensor; load stays generated from
            # the normal daily pattern so mismatch scenarios can be simulated.
            occupancy = min(z.occupancy_max, occupancy_override)

        co2_ppm = max(350.0, 400.0 + occupancy * z.co2_per_occupant_ppm + rng.gauss(0.0, 45.0))
        # Normal relative humidity stays in the ~45-65% band even at peak
        # occupancy, so a genuine envelope leak (> 72%) is unambiguous.
        humidity_percent = min(92.0, max(25.0, 46.0 + 12.0 * factor + 0.03 * occupancy + rng.gauss(0.0, 2.5)))

        # Outdoor ambient swings sinusoidally around 23 C, peaking mid-afternoon.
        temp_outdoor_c = 23.0 + 7.0 * math.sin(2.0 * math.pi * (hour_fraction - 10.0) / 24.0) + rng.gauss(0.0, 1.5)
        temp_indoor_c = 22.0 + 0.04 * (temp_outdoor_c - 23.0) + rng.gauss(0.0, 0.35)

        delta_t = max(0.0, temp_outdoor_c - temp_indoor_c)  # positive -> cooling duty
        hvac_kw = z.hvac_base_kw + 1.7 * delta_t + 0.22 * load_occupancy + 0.5 * z.hvac_base_kw * factor + rng.gauss(0.0, 0.8)

        lighting_kw = max(0.4, 1.6 + 0.135 * load_occupancy + rng.gauss(0.0, 0.35))
        plug_load_kw = max(0.8, 5.5 + 0.10 * load_occupancy + 0.5 * (0.02 * z.occupancy_max) * factor + rng.gauss(0.0, 0.5))

        return TelemetryPoint(
            timestamp=_now_iso(),
            building_id=self.building_id,
            zone_id=z.zone_id,
            telemetry=Telemetry(
                occupancy_count=occupancy,
                co2_ppm=round(co2_ppm, 2),
                humidity_percent=round(humidity_percent, 2),
                temp_indoor_c=round(temp_indoor_c, 2),
                temp_outdoor_c=round(temp_outdoor_c, 2),
                hvac_kw=round(hvac_kw, 3),
                lighting_kw=round(lighting_kw, 3),
                plug_load_kw=round(plug_load_kw, 3),
            ),
        )

    # ---------------------------------------------------------------- faults
    def _apply_faults(self, point: TelemetryPoint) -> None:
        """Mutate the sample in place according to every active manual fault.

        Faults are intentionally applied on top of an otherwise-normal sample so
        their signatures stay unambiguous and reproducible.
        """
        t = point.telemetry
        z = self.zone

        if self.controller.is_active(point.zone_id, AnomalyKind.HVAC_OVERVENTILATION):
            # Demand-ventilation fault: mechanical ventilation drives HVAC load
            # up while CO2 is clamped back to outdoor baseline (~400 ppm).
            t.hvac_kw *= 1.75
            t.co2_ppm = round(400.0 + self._rng.gauss(0.0, 8.0), 2)

        if self.controller.is_active(point.zone_id, AnomalyKind.THERMAL_ENVELOPE_LEAK):
            # Envelope leak: humid outdoor air infiltrates, forcing latent
            # (dehumidification) duty - humidity climbs and HVAC load spikes.
            t.humidity_percent = min(94.0, t.humidity_percent * 1.55 + 14.0)
            t.hvac_kw = t.hvac_kw * 1.5 + 4.0

        if self.controller.is_active(point.zone_id, AnomalyKind.UNOCCUPIED_LIGHTING):
            # Lighting left on while the space is empty: occupancy forced to 0,
            # lighting driven to nameplate peak, CO2 decays toward baseline.
            t.occupancy_count = 0
            t.lighting_kw = z.lighting_peak_kw + self._rng.gauss(0.0, 0.2)
            t.co2_ppm = min(t.co2_ppm, 430.0)

        if self.controller.is_active(point.zone_id, AnomalyKind.EQUIPMENT_DRIFT):
            # Subtle permanent step: parasitic equipment load rises ~18% and
            # stays elevated until the panel explicitly clears/resets it.
            t.plug_load_kw += 3.4

    # ------------------------------------------------------------------ loop
    async def run(self) -> None:
        """Infinite generation loop - one sample per ``interval`` seconds."""
        while True:
            now = datetime.now(timezone.utc)
            hour_fraction = now.hour + now.minute / 60.0 + now.second / 3600.0
            point = self._normal_sample(now, hour_fraction)
            self._apply_faults(point)
            for metric, value in self.controller.sensor_overrides(point.zone_id).items():
                setattr(point.telemetry, metric, int(value) if metric == "occupancy_count" else value)
            point.injected_state = self.controller.state_string(point.zone_id)
            # Round rounded metrics again so results are stable for storage.
            point.telemetry.hvac_kw = round(point.telemetry.hvac_kw, 3)
            point.telemetry.lighting_kw = round(point.telemetry.lighting_kw, 3)
            point.telemetry.plug_load_kw = round(point.telemetry.plug_load_kw, 3)
            await self.sink(self.zone, point)
            await asyncio.sleep(self.interval)
