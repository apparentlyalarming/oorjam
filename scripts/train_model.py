"""CLI: train baseline Isolation Forest + consumption regressors.

Run from the repository root::

    ./.venv/bin/python scripts/train_model.py --days 7 --resolution 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.ml.training import train_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train energy-auditor models")
    parser.add_argument("--days", type=int, default=7, help="synthetic baseline days")
    parser.add_argument(
        "--resolution", type=float, default=10.0, help="sample spacing in seconds"
    )
    args = parser.parse_args()

    train_artifacts(settings, days=args.days, resolution_sec=args.resolution)


if __name__ == "__main__":
    main()