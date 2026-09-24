"""Process entrypoint: ``uvicorn app.main:app``.

Owns the runtime lifecycle: builds the pipeline before serving requests and
cancels the generator tasks cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.app import create_app
from .config import settings
from .runtime import AuditorRuntime


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = await AuditorRuntime.create(settings)
    task = asyncio.create_task(runtime.run())
    app.state.runtime = runtime
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            pass
        close = getattr(runtime.store, "close", None)
        if close is not None:  # memory store has no resource pool
            await close()


app = create_app(settings, lifespan)