"""FastAPI application factory: REST + WebSocket interfaces to the runtime.

The app object is created by :func:`create_app` and given a lifespan that owns
the :class:`app.runtime.AuditorRuntime`; all endpoints read live state from
``request.app.state.runtime``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings
from ..runtime import AuditorRuntime
from ..schemas import (
    AnomalyKind,
    AnalyticsReport,
    AnomalyRecord,
    HealthResponse,
    InjectionState,
    SimulatorCommand,
    ZoneStatus,
)
from ..zones import ZONE_CATALOG


def _static_dir() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent / "web" / "static")


def create_app(settings: Settings, lifespan) -> FastAPI:
    app = FastAPI(title="AI Energy-Waste Auditor", version="1.0.0", lifespan=lifespan)

    # ------------------------------------------------------------- static UI
    app.mount("/static", StaticFiles(directory=_static_dir()), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(f"{_static_dir()}/dashboard.html")

    @app.get("/simulator", include_in_schema=False)
    async def simulator_page() -> FileResponse:
        return FileResponse(f"{_static_dir()}/simulator.html")

    # ---------------------------------------------------------------- REST
    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        rt: AuditorRuntime = request.app.state.runtime
        return HealthResponse(
            status="ok",
            detector_loaded=rt.detector is not None,
            baseline_model_zones=rt.baseline_model_zones(),
            storage_backend=rt.store.kind if rt.store else "n/a",
            zones=[z.zone_id for z in ZONE_CATALOG],
            injection=[k.value for k in AnomalyKind],
            uptime_seconds=round(rt.uptime_seconds(), 1),
            samples_processed=rt.samples_processed,
        )

    @app.get("/api/zones")
    async def zones() -> List[Dict[str, Any]]:
        return [
            {
                "zone_id": z.zone_id,
                "occupancy_max": z.occupancy_max,
                "lighting_peak_kw": z.lighting_peak_kw,
                "plug_peak_kw": z.plug_peak_kw,
                "hvac_base_kw": z.hvac_base_kw,
            }
            for z in ZONE_CATALOG
        ]

    @app.get("/api/status", response_model=List[ZoneStatus])
    async def status(request: Request) -> List[ZoneStatus]:
        rt: AuditorRuntime = request.app.state.runtime
        return [_build_zone_status(rt, z.zone_id) for z in ZONE_CATALOG]

    @app.get("/api/telemetry/current")
    async def telemetry_current(request: Request) -> Dict[str, Any]:
        rt: AuditorRuntime = request.app.state.runtime
        out: Dict[str, Any] = {}
        if rt.store is not None:
            for z in ZONE_CATALOG:
                recent = await rt.store.recent_telemetry(z.zone_id, limit=1)
                if recent:
                    out[z.zone_id] = recent[-1].model_dump()
        return out

    @app.get("/api/anomalies", response_model=List[AnomalyRecord])
    async def anomalies(
        request: Request, limit: int = Query(50, ge=1, le=500)
    ) -> List[AnomalyRecord]:
        rt: AuditorRuntime = request.app.state.runtime
        return list(reversed(rt.anomalies[-limit:]))

    @app.get("/api/analytics/current", response_model=AnalyticsReport)
    async def analytics_current(request: Request) -> AnalyticsReport:
        rt: AuditorRuntime = request.app.state.runtime
        return rt.current_report()

    @app.get("/api/report/markdown")
    async def report_markdown(request: Request) -> PlainTextResponse:
        rt: AuditorRuntime = request.app.state.runtime
        markdown = rt.analytics.as_markdown(rt.current_report())
        return PlainTextResponse(markdown, media_type="text/markdown")

    @app.get("/api/storage")
    async def storage_snapshot(request: Request) -> Dict[str, Any]:
        rt: AuditorRuntime = request.app.state.runtime
        if rt.store is None:
            return {"kind": "n/a", "ready": False}
        return await rt.store.snapshot()

    @app.get("/api/injection")
    async def injection_snapshot(request: Request) -> Dict[str, Any]:
        rt: AuditorRuntime = request.app.state.runtime
        return {"zones": rt.controller_snapshot(), "kinds": [k.value for k in AnomalyKind]}

    @app.post("/api/inject", response_model=InjectionState)
    async def inject(cmd: SimulatorCommand, request: Request) -> InjectionState:
        rt: AuditorRuntime = request.app.state.runtime
        return rt.apply_command(cmd)

    # ---------------------------------------------------------------- WebSockets
    @app.websocket("/ws/telemetry")
    async def ws_telemetry(
        websocket: WebSocket,
        zone: Optional[str] = Query(None),
        watch_all: bool = Query(False),
    ) -> None:
        """Live stream: replay history, then push points/verdicts/anomalies.

        ``zone`` filters point/verdict frames to one zone unless ``watch_all``
        is enabled; dashboards can chart one zone and alert on others.
        """
        rt: AuditorRuntime = websocket.app.state.runtime
        await websocket.accept()

        await websocket.send_json(
            {
                "type": "hello",
                "zones": [z.zone_id for z in ZONE_CATALOG],
                "interval_seconds": rt.settings.emit_interval_seconds,
            }
        )

        # Replay persisted history for the requested zone.
        if rt.store is not None:
            points = await rt.store.recent_telemetry(zone, limit=rt.settings.dashboard_history_points)
            await websocket.send_json(
                {"type": "history", "zone": zone, "points": [p.model_dump() for p in points]}
            )

        queue = rt.subscribe()
        try:
            while True:
                message = await queue.get()
                mtype = message.get("type")
                if mtype in ("point", "verdict") and zone is not None and not watch_all:
                    subject = (message.get("point") or message.get("verdict") or {}).get("zone_id")
                    if subject != zone:
                        continue
                await websocket.send_json(message)
        except WebSocketDisconnect:
            pass
        finally:
            rt.unsubscribe(queue)

    @app.websocket("/ws/simulator")
    async def ws_simulator(websocket: WebSocket) -> None:
        """Injection control channel: commands in, state + live events out."""
        rt: AuditorRuntime = websocket.app.state.runtime
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "ready",
                "zones": rt.controller_snapshot(),
                "kinds": [k.value for k in AnomalyKind],
                "log": rt.log.snapshot(),
            }
        )

        queue = rt.subscribe()
        forwarder: Optional[Any] = None
        try:
            import asyncio

            async def _forward() -> None:
                while True:
                    message = await queue.get()
                    if message.get("type") in ("injection", "anomaly", "verdict", "log"):
                        await websocket.send_json(message)

            forwarder = asyncio.create_task(_forward())
            while True:
                raw = await websocket.receive_text()
                try:
                    cmd = SimulatorCommand.model_validate_json(raw)
                except Exception:
                    await websocket.send_json(
                        {"type": "error", "detail": "invalid simulator command"}
                    )
                    continue
                state = rt.apply_command(cmd)
                await websocket.send_json({"type": "state", "state": state.model_dump()})
        except WebSocketDisconnect:
            pass
        finally:
            rt.unsubscribe(queue)
            if forwarder is not None:
                forwarder.cancel()

    return app


def _build_zone_status(rt: AuditorRuntime, zone_id: str) -> ZoneStatus:
    """Live header state for one zone."""
    verdict = rt.current.get(zone_id)
    zone_anoms = [a for a in rt.anomalies if a.zone_id == zone_id]
    return ZoneStatus(
        zone_id=zone_id,
        marker=verdict.marker if verdict else "green",
        severity=verdict.severity if verdict else "normal",
        diagnosis=verdict.diagnosis if verdict else None,
        iforest_score=verdict.iforest_score if verdict else 0.0,
        injected_state=rt.controller.state_string(zone_id),
        exposure_usd=round(sum(a.financial_waste_usd for a in zone_anoms), 2),
        exposure_kwh=round(sum(a.energy_wasted_kwh for a in zone_anoms), 3),
    )
