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
    building_id: str = "bldg_01"
    floor_id: str = "floor_1"
    area_m2: float = 100.0
    ceiling_height_m: float = 3.0
    zone_type: str = "Office"
    operating_start: str = "08:00"
    operating_end: str = "18:00"
    utility_rate_usd_per_kwh: float | None = None

    @property
    def volume_m3(self) -> float:
        """Room volume derived from floor area and ceiling height."""
        return self.area_m2 * self.ceiling_height_m

    def is_operating(self, hour_fraction: float) -> bool:
        """Return whether local clock time falls inside this zone's schedule."""
        start_h, start_m = (int(part) for part in self.operating_start.split(":"))
        end_h, end_m = (int(part) for part in self.operating_end.split(":"))
        start = start_h + start_m / 60.0
        end = end_h + end_m / 60.0
        if start == end:
            return True  # equal endpoints represent a continuously operating zone
        if start <= end:
            return start <= hour_fraction < end
        return hour_fraction >= start or hour_fraction < end


ZONE_CATALOG: list[ZoneDef] = [
    ZoneDef(zone_id="floor_1_west", floor_id="floor_1", occupancy_max=90, lighting_peak_kw=15.0, plug_peak_kw=15.0, hvac_base_kw=14.0, area_m2=420.0),
    ZoneDef(zone_id="floor_2_east", floor_id="floor_2", occupancy_max=120, lighting_peak_kw=18.5, plug_peak_kw=18.0, hvac_base_kw=15.0, area_m2=560.0),
    ZoneDef(zone_id="floor_3_central", floor_id="floor_3", occupancy_max=70, lighting_peak_kw=12.0, plug_peak_kw=11.0, hvac_base_kw=12.0, area_m2=330.0),
]

ZONE_BY_ID = {zone.zone_id: zone for zone in ZONE_CATALOG}


def register_zone(zone: ZoneDef) -> None:
    """Add or replace one zone in the in-process registry."""
    if zone.zone_id not in ZONE_BY_ID:
        ZONE_CATALOG.append(zone)
    else:
        ZONE_CATALOG[:] = [zone if z.zone_id == zone.zone_id else z for z in ZONE_CATALOG]
    ZONE_BY_ID[zone.zone_id] = zone

def get_zone(zone_id: str) -> ZoneDef:
    """Resolve a zone id to its definition, raising a helpful KeyError."""
    try:
        return ZONE_BY_ID[zone_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown zone {zone_id!r}; known zones: {list(ZONE_BY_ID)}") from exc
