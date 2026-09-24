"""Static catalogue of simulated building zones.

Each zone carries its physical sizing parameters. These feed (a) the telemetry
synthesis (scaling of diurnal curves) and (b) the diagnostic ruleset (peak load
references used to recognise waste signatures such as unoccupied lighting).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneDef:
    """Physical signature of one simulated zone."""

    zone_id: str
    occupancy_max: int          # design occupant count at peak
    lighting_peak_kw: float     # nameplate lighting load when fully lit
    plug_peak_kw: float         # nameplate plug-load capacity
    hvac_base_kw: float         # HVAC no-load / circulation baseline
    co2_per_occupant_ppm: float = 2.6  # CO2 contribution per occupant


ZONE_CATALOG: tuple[ZoneDef, ...] = (
    ZoneDef(zone_id="floor_1_west", occupancy_max=90, lighting_peak_kw=15.0, plug_peak_kw=15.0, hvac_base_kw=14.0),
    ZoneDef(zone_id="floor_2_east", occupancy_max=120, lighting_peak_kw=18.5, plug_peak_kw=18.0, hvac_base_kw=15.0),
    ZoneDef(zone_id="floor_3_central", occupancy_max=70, lighting_peak_kw=12.0, plug_peak_kw=11.0, hvac_base_kw=12.0),
)

ZONE_BY_ID = {zone.zone_id: zone for zone in ZONE_CATALOG}

def get_zone(zone_id: str) -> ZoneDef:
    """Resolve a zone id to its definition, raising a helpful KeyError."""
    try:
        return ZONE_BY_ID[zone_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown zone {zone_id!r}; known zones: {list(ZONE_BY_ID)}") from exc