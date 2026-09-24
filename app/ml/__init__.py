"""ML subpackage: unsupervised anomaly detection and diagnostic ruleset."""

from .detector import Detector, load_detector
from .rules import DiagnosticClassifier, MARKER_COLORS, PRIMARY_METRIC
from .training import ensure_artifacts, train_artifacts

__all__ = [
    "Detector",
    "load_detector",
    "DiagnosticClassifier",
    "MARKER_COLORS",
    "PRIMARY_METRIC",
    "ensure_artifacts",
    "train_artifacts",
]