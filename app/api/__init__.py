"""FastAPI interface layer (REST + WebSocket)."""

from .app import create_app

__all__ = ["create_app"]