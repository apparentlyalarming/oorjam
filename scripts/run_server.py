"""CLI: run the FastAPI app with the auditor runtime.

Convenience wrapper around uvicorn (equivalent to
``uvicorn app.main:app --host 0.0.0.0 --port 8000``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402

from app.config import settings  # noqa: E402


def main() -> None:
    print(f"Open in your browser: http://127.0.0.1:{settings.port}/")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()