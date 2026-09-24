"""Isolation Forest inference pipeline.

The primary model is an unsupervised Isolation Forest; anomaly scores are the
``decision_function`` output (lower = more anomalous, roughly symmetric around
0).  A score below the contamination-derived threshold marks the window as
anomalous and hands off to the :class:`DiagnosticClassifier` ruleset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np

from ..config import Settings, settings as default_settings
from .training import ARTIFACT_FILENAME, ensure_artifacts


class Detector:
    """Thin wrapper over the persisted IsolationForest + StandardScaler."""

    def __init__(
        self,
        forest,
        scaler,
        threshold: float,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        self.forest = forest
        self.scaler = scaler
        self.threshold = threshold
        self.feature_names = feature_names or []

    def score(self, X: np.ndarray) -> float:
        """Standardised Isolation Forest decision score for a single vector.

        Returns a lower-is-more-anomalous float in roughly [-0.5, 0.5].
        """
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float64).reshape(1, -1))
        return float(self.forest.decision_function(X_scaled)[0])

    def is_anomalous(self, X: np.ndarray) -> bool:
        return self.score(X) < self.threshold

    def as_payload(self) -> dict:
        return {
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "forest_class": type(self.forest).__name__,
            "n_estimators": getattr(self.forest, "n_estimators", None),
        }


def load_detector(
    model_dir: Path,
    settings: Settings = default_settings,
) -> Optional[Detector]:
    """Load persisted detector artifacts, training them if necessary."""
    ensure_artifacts(settings)
    path = model_dir / ARTIFACT_FILENAME
    if not path.exists():
        return None  # pragma: no cover - defensive
    payload: Dict = joblib.load(path)
    return Detector(
        forest=payload["forest"],
        scaler=payload["scaler"],
        threshold=payload["threshold"],
        feature_names=payload.get("feature_names"),
    )