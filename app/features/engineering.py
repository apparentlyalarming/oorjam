"""Module 2: sliding-window feature engineering.

The pipeline keeps an in-memory window of raw samples (15-30 min, default 20)
and derives the five engineering features required by the specification:

1. ``ventilation_efficiency_ratio`` = hvac_kw / co2_ppm
       Energy spent per unit of ventilation effectiveness.  A high value with a
       LOW CO2 reading means the AHU is over-ventilating an unoccupied or
       lightly loaded space (air is being conditioned for nobody).

2. ``lighting_per_occupant`` = lighting_kw / (occupancy_count + 1)
       Lighting intensity per person.  The ``+1`` regularises the ratio so an
       empty zone does not divide by zero (avoids infinite feature values that
       would poison the Isolation Forest scaling).

3. ``thermal_stress_ratio`` = hvac_kw * |temp_outdoor_c - temp_indoor_c|
       Total HVAC power multiplied by the envelope temperature gradient - a
       proxy for how hard the system works against the building shell.

4. ``latent_load_index`` = humidity_percent * hvac_kw
       Latent (moisture) burden: high humidity combined with high HVAC power
       typically indicates envelope leaks / condensation problems.

5. ``power_delta_dt`` = hvac_kw_t - hvac_kw_{t-1}
       First derivative of HVAC power.  Detects step changes (equipment drift,
       fault injection) that static levels cannot.

The detector consumes **window aggregates** of these five features - the mean
and standard deviation over the last N minutes (10-dimensional vector).  Using
aggregates rather than a single instant gives the model temporal context and
makes scores robust to sensor noise; means also make the feature vector
independent of the sampling resolution, so training data and live inference
live in the same statistical space.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence

import numpy as np

from ..schemas import TelemetryPoint

BASE_FEATURES: tuple[str, ...] = (
    "ventilation_efficiency_ratio",
    "lighting_per_occupant",
    "thermal_stress_ratio",
    "latent_load_index",
    "power_delta_dt",
)


@dataclass(frozen=True)
class WindowFeatures:
    """Feature vector plus the raw window aggregates used by diagnostic rules.

    ``X`` holds the 10-dim model input (mean || std over the full sliding
    window, fed to the Isolation Forest).  The ``*_fast`` fields are the same
    sensor means computed over a short *retroactive* sub-window so the
    real-time diagnostic rules react to a freshly injected fault immediately
    instead of waiting for the whole 15-30 min window to fill.
    """

    timestamp: str
    zone_id: str
    X: np.ndarray                     # 10-dim model input (mean || std)
    feature_names: List[str]
    occupancy_mean: float
    co2_mean: float
    humidity_mean: float
    temp_indoor_mean: float
    temp_outdoor_mean: float
    hvac_mean: float
    lighting_mean: float
    plug_mean: float
    hour_fraction: float
    # Fast (short-window) aggregations for the diagnostic ruleset.
    occupancy_fast: float
    co2_fast: float
    humidity_fast: float
    temp_indoor_fast: float
    temp_outdoor_fast: float
    hvac_fast: float
    lighting_fast: float
    plug_fast: float


def _per_sample_features(
    hvac: np.ndarray,
    co2: np.ndarray,
    lighting: np.ndarray,
    occupancy: np.ndarray,
    temp_indoor: np.ndarray,
    temp_outdoor: np.ndarray,
    humidity: np.ndarray,
    delta_t: np.ndarray,
) -> np.ndarray:
    """Vectorised evaluation of the five spec features over a raw window.

    Returns an ``(n, 5)`` array.  Division guards use ``np.maximum`` with a
    small epsilon only to satisfy numerical safety - occupancy is guarded in the
    formula itself by the ``+ 1`` regularisation required by the spec.
    """
    ventilation_efficiency_ratio = hvac / np.maximum(co2, 1.0)          # kW / ppm
    lighting_per_occupant = lighting / (occupancy + 1.0)                # kW / person
    thermal_stress_ratio = hvac * np.abs(temp_outdoor - temp_indoor)    # kW * deltaC
    latent_load_index = humidity * hvac                                 # % * kW
    power_delta_dt = delta_t                                            # kW step
    return np.column_stack(
        [
            ventilation_efficiency_ratio,
            lighting_per_occupant,
            thermal_stress_ratio,
            latent_load_index,
            power_delta_dt,
        ]
    )


class FeatureEngine:
    """In-memory sliding-window feature extractor (one instance per zone)."""

    def __init__(
        self, window_points: int, min_samples: int, fast_window_points: int = 30
    ) -> None:
        self.window_points = max(2, window_points)
        self.min_samples = max(2, min_samples)
        self.fast_window_points = max(2, fast_window_points)
        self._buffer: Deque[TelemetryPoint] = deque(maxlen=self.window_points)
        self._prev_hvac_kw: Optional[float] = None
        self._prev_ts: Optional[float] = None
        # Aggregate feature names are fed to the Isolation Forest as-is.
        self.feature_names: List[str] = [f"{name}__mean" for name in BASE_FEATURES] + [
            f"{name}__std" for name in BASE_FEATURES
        ]

    def reset(self) -> None:
        self._buffer.clear()
        self._prev_hvac_kw = None
        self._prev_ts = None

    def seed(self, points: List[TelemetryPoint]) -> None:
        """Bulk-load baseline history so the sliding window is full immediately.

        Bulk-fill avoids the O(n^2) cost of calling :meth:`update` once per
        historical sample (each call re-aggregates the whole buffer).
        """
        if not points:
            return
        self._buffer.extend(points)  # deque(maxlen=...) evicts oldest as needed
        last = points[-1]
        self._prev_hvac_kw = last.telemetry.hvac_kw
        self._prev_ts = self._parse_ts(last.timestamp)

    def _parse_ts(self, timestamp: str) -> float:
        # Timestamps are ISO-8601; slicing to 19 chars keeps second precision
        # and avoids the dependency on datetime parsing in the hot loop.
        from datetime import datetime, timezone

        try:
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
        except ValueError:  # pragma: no cover - defensive
            return datetime.now(timezone.utc).timestamp()

    def update(self, point: TelemetryPoint) -> Optional[WindowFeatures]:
        """Append ``point`` and return the aggregated feature vector.

        Returns ``None`` until at least ``min_samples`` points have accumulated
        (the model must not score an under-filled window).
        """
        self._buffer.append(point)
        if len(self._buffer) < self.min_samples:
            self._prev_hvac_kw = point.telemetry.hvac_kw
            self._prev_ts = self._parse_ts(point.timestamp)
            return None

        window = list(self._buffer)
        n = len(window)

        hvac = np.empty(n)
        co2 = np.empty(n)
        lighting = np.empty(n)
        occupancy = np.empty(n)
        temp_in = np.empty(n)
        temp_out = np.empty(n)
        humidity = np.empty(n)
        plug = np.empty(n)
        for i, p in enumerate(window):
            t = p.telemetry
            hvac[i] = t.hvac_kw
            co2[i] = t.co2_ppm
            lighting[i] = t.lighting_kw
            occupancy[i] = t.occupancy_count
            temp_in[i] = t.temp_indoor_c
            temp_out[i] = t.temp_outdoor_c
            humidity[i] = t.humidity_percent
            plug[i] = t.plug_load_kw

        # power_delta_dt: first-order difference of HVAC power.  Using np.diff
        # with a zero-padded head keeps array length stable.
        delta = np.diff(hvac, prepend=hvac[0])

        feats = _per_sample_features(hvac, co2, lighting, occupancy, temp_in, temp_out, humidity, delta)

        # Window aggregates: mean and std over the full sliding window.
        means = feats.mean(axis=0)
        stds = feats.std(axis=0)
        X = np.concatenate([means, stds]).astype(np.float64)

        # Maintain running previous-point state (kept for completeness of the
        # spec's t-1 definition; np.diff already covers the window).
        self._prev_hvac_kw = hvac[-1]
        self._prev_ts = self._parse_ts(point.timestamp)

        last = window[-1]
        hour = float(last.timestamp[11:13] or 0) + float(last.timestamp[14:16] or 0) / 60.0

        # Fast sub-window (the most recent `fast_window_points` samples) for
        # real-time diagnostic rules - means of the last k samples.
        k = min(self.fast_window_points, n)

        return WindowFeatures(
            timestamp=last.timestamp,
            zone_id=last.zone_id,
            X=X,
            feature_names=self.feature_names,
            occupancy_mean=float(occupancy.mean()),
            co2_mean=float(co2.mean()),
            humidity_mean=float(humidity.mean()),
            temp_indoor_mean=float(temp_in.mean()),
            temp_outdoor_mean=float(temp_out.mean()),
            hvac_mean=float(hvac.mean()),
            lighting_mean=float(lighting.mean()),
            plug_mean=float(plug.mean()),
            hour_fraction=hour,
            occupancy_fast=float(occupancy[-k:].mean()),
            co2_fast=float(co2[-k:].mean()),
            humidity_fast=float(humidity[-k:].mean()),
            temp_indoor_fast=float(temp_in[-k:].mean()),
            temp_outdoor_fast=float(temp_out[-k:].mean()),
            hvac_fast=float(hvac[-k:].mean()),
            lighting_fast=float(lighting[-k:].mean()),
            plug_fast=float(plug[-k:].mean()),
        )

    def warm(self) -> bool:
        """True when the window has enough samples for inference."""
        return len(self._buffer) >= self.min_samples