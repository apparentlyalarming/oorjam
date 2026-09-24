"""Pydantic contracts shared across the simulator, detectors, APIs and store.

``TelemetryPoint`` mirrors the specification JSON payload exactly; ``marker`` /
``score`` are optional enrichment fields attached after ML inference so a
single record can round-trip through storage and the dashboard websocket.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnomalyKind(str, Enum):
    """Manual fault classes the simulator control panel can inject."""

    HVAC_OVERVENTILATION = "hvac_overventilation"
    THERMAL_ENVELOPE_LEAK = "thermal_envelope_leak"
    UNOCCUPIED_LIGHTING = "unoccupied_lighting"
    EQUIPMENT_DRIFT = "equipment_drift"


class Telemetry(BaseModel):
    """Raw per-sensor reading set for one zone."""

    occupancy_count: int
    co2_ppm: float
    humidity_percent: float
    temp_indoor_c: float
    temp_outdoor_c: float
    hvac_kw: float
    lighting_kw: float
    plug_load_kw: float


class TelemetryPoint(BaseModel):
    """One telemetry sample, as defined by the ingestion schema.

    ``marker`` and ``score`` default to green/0.0 so the payload is valid
    *before* ML scoring; the runtime overwrites them with the verdict.
    """

    timestamp: str
    building_id: str
    zone_id: str
    telemetry: Telemetry
    injected_state: str = "normal"
    marker: str = "green"
    score: float = 0.0
    diagnosis: Optional[str] = None
    simulated_hour_fraction: Optional[float] = None


class SimulatorCommand(BaseModel):
    """Client -> server message from the injection control panel."""

    action: str = Field(description="inject | clear | reset | set_sensor | set_time_of_day | ...")
    zone: str = "floor_2_east"
    anomaly: Optional[AnomalyKind] = None
    occupancy: Optional[int] = Field(default=None, ge=0)
    metric: Optional[str] = None
    value: Optional[float] = None
    time_minutes: Optional[int] = Field(default=None, ge=0, le=1439)


class ZoneCreate(BaseModel):
    building_id: str = Field(min_length=1, max_length=80)
    floor_id: str = Field(min_length=1, max_length=80)
    zone_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    area_m2: float = Field(gt=0, le=100000)
    ceiling_height_m: float = Field(gt=0, le=100)
    occupancy_max: int = Field(gt=0, le=100000)
    zone_type: str = Field(default="Office", min_length=1, max_length=80)
    operating_start: str = Field(default="08:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    operating_end: str = Field(default="18:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    utility_rate_usd_per_kwh: Optional[float] = Field(default=None, ge=0, le=100)


class ZoneUpdate(BaseModel):
    building_id: Optional[str] = Field(default=None, min_length=1, max_length=80)
    floor_id: Optional[str] = Field(default=None, min_length=1, max_length=80)
    area_m2: Optional[float] = Field(default=None, gt=0, le=100000)
    ceiling_height_m: Optional[float] = Field(default=None, gt=0, le=100)
    occupancy_max: Optional[int] = Field(default=None, gt=0, le=100000)
    zone_type: Optional[str] = Field(default=None, min_length=1, max_length=80)
    operating_start: Optional[str] = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    operating_end: Optional[str] = Field(default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    utility_rate_usd_per_kwh: Optional[float] = Field(default=None, ge=0, le=100)


class UtilityRateUpdate(BaseModel):
    utility_rate_usd_per_kwh: float = Field(ge=0, le=100)


class InjectionState(BaseModel):
    """Server -> client acknowledgement of the current controller state."""

    zone: str
    active: List[str] = Field(default_factory=list)
    injected_state: str = "normal"
    occupancy_override: Optional[int] = None
    sensor_overrides: Dict[str, float] = Field(default_factory=dict)
    time_of_day_override_minutes: Optional[int] = None
    actuations: Dict[str, float] = Field(default_factory=dict)


class Verdict(BaseModel):
    """ML + ruleset output attached to every processed sample."""

    zone_id: str
    timestamp: str
    iforest_score: float
    is_anomaly: bool
    diagnosis: Optional[str] = None
    severity: str = "normal"  # normal | high | moderate | subtle
    marker: str = "green"      # green | red | orange | yellow
    confidence: float = 0.0
    explanation: str = ""


class AnomalyRecord(BaseModel):
    """A completed (closed) waste event with monetised impact."""

    start_timestamp: str
    end_timestamp: str
    zone_id: str
    diagnosis: str
    severity: str
    marker: str = "green"
    iforest_score: float
    energy_wasted_kwh: float
    financial_waste_usd: float
    peak_kw: float
    sample_count: int
    duration_hours: float
    details: Dict[str, Any] = Field(default_factory=dict)


class Recommendation(BaseModel):
    """One retrofit / behavioural recommendation with ROI maths."""

    id: str
    title: str
    category: str  # Retrofit | Behavioural
    applies_to: List[str] = Field(default_factory=list)
    capital_cost_usd: float
    maintenance_usd_per_year: float
    annual_energy_savings_kwh: float
    lifetime_years: int
    rate_usd_per_kwh: float = 0.15
    payback_months: float = 0.0
    annual_roi_pct: float = 0.0
    annual_net_savings_usd: float = 0.0


class DiagnosisBucket(BaseModel):
    """Aggregated waste for one root-cause tag."""

    count: int
    kwh: float
    usd: float
    severity: str = "normal"
    marker: str = "green"
    zones: List[str] = Field(default_factory=list)


class AnalyticsReport(BaseModel):
    """Structured JSON summary emitted by the analytics engine."""

    generated_at: str
    period_seconds: float
    uptime_seconds: float
    total_energy_wasted_kwh: float
    total_financial_waste_usd: float
    total_anomalies: int
    open_anomalies: int
    current_markers: Dict[str, str] = Field(default_factory=dict)
    by_diagnosis: Dict[str, DiagnosisBucket] = Field(default_factory=dict)
    anomalies: List[AnomalyRecord] = Field(default_factory=list)
    top_recommendations: List[Recommendation] = Field(default_factory=list)


class ZoneStatus(BaseModel):
    """Live per-zone dashboard header state."""

    zone_id: str
    marker: str = "green"
    severity: str = "normal"
    diagnosis: Optional[str] = None
    iforest_score: float = 0.0
    injected_state: str = "normal"
    exposure_usd: float = 0.0
    exposure_kwh: float = 0.0


class HealthResponse(BaseModel):
    status: str
    detector_loaded: bool
    baseline_model_zones: List[str] = Field(default_factory=list)
    storage_backend: str
    zones: List[str] = Field(default_factory=list)
    injection: List[str] = Field(default_factory=list)
    uptime_seconds: float = 0.0
    samples_processed: int = 0
