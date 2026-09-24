"""Model training: baseline synthesis + Isolation Forest + baseline regressors.

The pipeline is fully re-runnable via ``python scripts/train_model.py``.  It:

1. Synthesises *baseline-only* telemetry (no injected faults) for every zone at
   a configurable resolution over ``days`` of simulated time.
2. Feeds the raw stream through the same :class:`FeatureEngine` used at runtime,
   so the trained Isolation Forest sees the *identical* 10-dimensional feature
   space as live inference.
3. Fits an Isolation Forest (primary unsupervised model) plus a ``StandardScaler``
   and derives the anomaly-score decision threshold from the training scores.
4. Fits simple per-zone *baseline consumption models* (random forests) that
   predict expected HVAC / lighting / plug kW from calendar + occupancy + load
   inputs.  These predict the **Baseline Model kW** term used by the analytics
   engine to monetise waste (actual - expected).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ..config import Settings, settings as default_settings
from ..features.engineering import FeatureEngine
from ..schemas import Telemetry, TelemetryPoint
from ..telemetry.generator import TelemetryGenerator, InjectionController, _now_iso
from ..zones import ZoneDef, ZONE_CATALOG

ARTIFACT_FILENAME = "detector.joblib"
BASELINE_FILENAME = "baseline_models.joblib"


def _baseline_inputs(
    points: Sequence[TelemetryPoint],
) -> Dict[str, np.ndarray]:
    """Design matrix shared by the three baseline regression models.

    Calendar time is encoded cyclically (sin/cos of hour) so that 23:00 and
    01:00 are near neighbours; occupancy carries the human-driven load and the
    thermal gradient carries the shell load.
    """
    n = len(points)
    hour = np.empty(n)
    occ = np.empty(n)
    delta_t = np.empty(n)
    for i, p in enumerate(points):
        hh = float(p.timestamp[11:13] or 0)
        mm = float(p.timestamp[14:16] or 0)
        hour[i] = hh + mm / 60.0
        occ[i] = p.telemetry.occupancy_count
        delta_t[i] = max(0.0, p.telemetry.temp_outdoor_c - p.telemetry.temp_indoor_c)

    sin_h = np.sin(2.0 * math.pi * hour / 24.0)
    cos_h = np.cos(2.0 * math.pi * hour / 24.0)
    # hvac and plug/lighting get slightly different feature sets.
    return {
        "hvac": np.column_stack([sin_h, cos_h, occ, delta_t]),
        "lighting": np.column_stack([sin_h, cos_h, occ]),
        "plug": np.column_stack([sin_h, cos_h, occ]),
    }


def _targets(points: Sequence[TelemetryPoint]) -> Dict[str, np.ndarray]:
    return {
        "hvac": np.array([p.telemetry.hvac_kw for p in points]),
        "lighting": np.array([p.telemetry.lighting_kw for p in points]),
        "plug": np.array([p.telemetry.plug_load_kw for p in points]),
    }


def _fit_baseline_models(
    zone: ZoneDef, points: List[TelemetryPoint], seed: int
) -> Dict[str, RandomForestRegressor]:
    """Fit expected-consumption regressors (Baseline Model kW source)."""
    X = _baseline_inputs(points)
    y = _targets(points)
    plants: Dict[str, RandomForestRegressor] = {}
    for metric, xmat in X.items():
        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(xmat, y[metric])
        plants[metric] = model
    return plants


def _zone_baseline_series(
    zone: ZoneDef, days: int, resolution_sec: float, seed: int
) -> List[TelemetryPoint]:
    """Generate baseline telemetry by reusing the live generator machinery.

    The generator's own diurnal + noise synthesis is exactly what the auditor
    will see at runtime, guaranteeing train/serve consistency.
    """
    controller = InjectionController()  # no faults -> pure baseline
    points: List[TelemetryPoint] = []

    async def _collect(_z: ZoneDef, point: TelemetryPoint) -> None:
        points.append(point)

    generator = TelemetryGenerator(
        building_id="bldg_train",
        zone=zone,
        controller=controller,
        sink=_collect,
        interval=1.0,
        seed=seed,
    )

    # Run the same rhythm generator but fast-forwarded: drive clock manually by
    # faking timestamps so `days` of data is produced instantly rather than in
    # real time.  A tiny inner runner advances a synthetic clock.
    from datetime import datetime, timedelta, timezone

    steps = int(days * 24 * 3600 / resolution_sec)
    base_ts = datetime(2026, 1, 6, 0, 0, tzinfo=timezone.utc)  # a Tuesday
    rng = np.random.default_rng(seed)

    for i in range(steps):
        clock = base_ts + timedelta(seconds=i * resolution_sec)
        hour_fraction = clock.hour + clock.minute / 60.0 + clock.second / 3600.0
        point = generator._normal_sample(clock, hour_fraction)
        point.timestamp = clock.isoformat().replace("+00:00", "Z")
        points.append(point)
    return points


def _extract_runtime_features(points: List[TelemetryPoint], window_points: int, min_samples: int):
    """Replay baseline points through FeatureEngine -> (X, feature_names)."""
    engine = FeatureEngine(window_points=window_points, min_samples=min_samples)
    vectors: List[np.ndarray] = []
    for point in points:
        fv = engine.update(point)
        if fv is not None:
            vectors.append(fv.X)
    X = np.vstack(vectors) if vectors else np.zeros((0, len(engine.feature_names)))
    return X, engine.feature_names


def train_artifacts(
    settings: Settings = default_settings,
    days: int = 7,
    resolution_sec: float = 10.0,
) -> None:
    """Fit and persist all model artifacts into ``settings.model_dir``."""
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    window_points = max(2, int(settings.window_seconds / resolution_sec))
    feature_names: Optional[List[str]] = None
    all_X: List[np.ndarray] = []
    baseline_models: Dict[str, Dict[str, RandomForestRegressor]] = {}

    for idx, zone in enumerate(ZONE_CATALOG):
        print(f"[train] synthesising {days}d baseline for {zone.zone_id} @ {resolution_sec}s ...")
        points = _zone_baseline_series(zone, days, resolution_sec, seed=settings.random_state + idx)
        X, feature_names = _extract_runtime_features(points, window_points, settings.window_min_samples)
        all_X.append(X)
        baseline_models[zone.zone_id] = _fit_baseline_models(zone, points, settings.random_state + idx)
        print(f"[train]     {len(points)} samples -> {len(X)} feature vectors")

    X_full = np.vstack(all_X)
    scaler = StandardScaler().fit(X_full)
    X_scaled = scaler.transform(X_full)

    forest = IsolationForest(
        n_estimators=settings.isolation_forest_n_estimators,
        contamination=settings.contamination,
        random_state=settings.random_state,
        n_jobs=-1,
    )
    forest.fit(X_scaled)
    train_scores = forest.decision_function(X_scaled)
    # Threshold = the score at the contamination quantile of the training
    # distribution, so ~`contamination` of baseline data sits below it.
    threshold = float(np.quantile(train_scores, settings.contamination))

    detector_payload = {
        "forest": forest,
        "scaler": scaler,
        "threshold": threshold,
        "feature_names": feature_names,
        "trained_at": _now_iso(),
        "n_samples": int(len(X_full)),
        "contamination": settings.contamination,
    }
    joblib.dump(detector_payload, settings.model_dir / ARTIFACT_FILENAME)
    joblib.dump(
        {"zones": baseline_models, "rate": settings.utility_rate_usd_per_kwh},
        settings.model_dir / BASELINE_FILENAME,
    )
    print(f"[train] artifacts written to {settings.model_dir}")


def ensure_artifacts(settings: Settings = default_settings) -> None:
    """Train on the spot if the artifacts are missing (used at app startup)."""
    if not (settings.model_dir / ARTIFACT_FILENAME).exists() or not (
        settings.model_dir / BASELINE_FILENAME
    ).exists():
        print("[startup] model artifacts missing - training baseline models ...")
        train_artifacts(settings, days=5, resolution_sec=10.0)