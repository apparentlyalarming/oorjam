"""FastAPI application factory: REST + WebSocket interfaces to the runtime.

The app object is created by :func:`create_app` and given a lifespan that owns
the :class:`app.runtime.AuditorRuntime`; all endpoints read live state from
``request.app.state.runtime``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
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
    ZoneCreate,
    ZoneUpdate,
    UtilityRateUpdate,
    ZoneStatus,
)
from ..zones import ZoneDef
from ..ml.training import train_from_points
from ..schemas import Telemetry, TelemetryPoint
import csv
import io
import math
from datetime import datetime
from dataclasses import asdict, replace
from starlette.concurrency import run_in_threadpool


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
            zones=list(rt.zones),
            injection=[k.value for k in AnomalyKind],
            uptime_seconds=round(rt.uptime_seconds(), 1),
            samples_processed=rt.samples_processed,
        )

    @app.get("/api/zones")
    async def zones(request: Request) -> List[Dict[str, Any]]:
        return [
            {
                "zone_id": z.zone_id,
                "occupancy_max": z.occupancy_max,
                "lighting_peak_kw": z.lighting_peak_kw,
                "plug_peak_kw": z.plug_peak_kw,
                "hvac_base_kw": z.hvac_base_kw,
            }
            for z in request.app.state.runtime.zones.values()
        ]

    @app.get("/api/topology")
    async def topology(request: Request) -> Dict[str, Any]:
        rt: AuditorRuntime = request.app.state.runtime
        buildings: Dict[str, Dict[str, Any]] = {}
        for zone in rt.zones.values():
            building = buildings.setdefault(zone.building_id, {"building_id": zone.building_id, "floors": {}})
            floor = building["floors"].setdefault(zone.floor_id, {"floor_id": zone.floor_id, "zones": []})
            floor["zones"].append(_zone_payload(zone))
        return {"buildings": list(buildings.values()), "utility_rate_usd_per_kwh": rt.settings.utility_rate_usd_per_kwh}

    @app.post("/api/topology/zones")
    async def create_zone(body: ZoneCreate, request: Request) -> Dict[str, Any]:
        rt: AuditorRuntime = request.app.state.runtime
        if body.zone_id in rt.zones:
            raise HTTPException(status_code=409, detail="zone_id already exists")
        zone = ZoneDef(
            zone_id=body.zone_id, building_id=body.building_id, floor_id=body.floor_id,
            area_m2=body.area_m2, ceiling_height_m=body.ceiling_height_m,
            occupancy_max=body.occupancy_max, zone_type=body.zone_type,
            operating_start=body.operating_start, operating_end=body.operating_end,
            utility_rate_usd_per_kwh=body.utility_rate_usd_per_kwh,
            lighting_peak_kw=round(body.area_m2 * 0.035, 2),
            plug_peak_kw=round(body.area_m2 * 0.04, 2),
            hvac_base_kw=round(body.area_m2 * 0.03, 2),
        )
        await rt.register_zone(zone)
        if rt.store is not None:
            await rt.store.save_zone_config(asdict(zone))
        return _zone_payload(zone)

    @app.patch("/api/topology/zones/{zone_id}")
    async def update_zone(zone_id: str, body: ZoneUpdate, request: Request) -> Dict[str, Any]:
        rt: AuditorRuntime = request.app.state.runtime
        zone = rt.zones.get(zone_id)
        if zone is None:
            raise HTTPException(status_code=404, detail="zone not found")
        changes = body.model_dump(exclude_unset=True)
        changes = {key: value for key, value in changes.items() if value is not None or key == "utility_rate_usd_per_kwh"}
        if changes.get("area_m2") is not None:
            changes["lighting_peak_kw"] = round(changes["area_m2"] * 0.035, 2)
            changes["plug_peak_kw"] = round(changes["area_m2"] * 0.04, 2)
            changes["hvac_base_kw"] = round(changes["area_m2"] * 0.03, 2)
        updated = replace(zone, **changes)
        await rt.register_zone(updated, replace=True)
        if rt.store is not None:
            await rt.store.save_zone_config(asdict(updated))
        return _zone_payload(updated)

    @app.put("/api/config/utility-rate")
    async def update_utility_rate(body: UtilityRateUpdate, request: Request) -> Dict[str, float]:
        rt: AuditorRuntime = request.app.state.runtime
        rt.settings.utility_rate_usd_per_kwh = body.utility_rate_usd_per_kwh
        rt.analytics.rate = body.utility_rate_usd_per_kwh
        rt.analytics.zone_rates = {
            zone.zone_id: (
                zone.utility_rate_usd_per_kwh
                if zone.utility_rate_usd_per_kwh is not None
                else body.utility_rate_usd_per_kwh
            )
            for zone in rt.zones.values()
        }
        if rt.store is not None:
            await rt.store.save_app_config("utility_rate_usd_per_kwh", body.utility_rate_usd_per_kwh)
        return {"utility_rate_usd_per_kwh": rt.settings.utility_rate_usd_per_kwh}

    @app.post("/api/v1/train")
    async def train_model(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="upload a CSV file")
        content = await file.read(25 * 1024 * 1024 + 1)
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="CSV must be 25 MB or smaller")
        try:
            points = _parse_training_csv(content)
            counts = await run_in_threadpool(train_from_points, points, request.app.state.runtime.settings)
        except (ValueError, UnicodeDecodeError, csv.Error) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        rt: AuditorRuntime = request.app.state.runtime
        rt.reload_models()
        return {"status": "trained", "rows_by_zone": counts, "total_rows": len(points)}

    @app.get("/api/status", response_model=List[ZoneStatus])
    async def status(request: Request) -> List[ZoneStatus]:
        rt: AuditorRuntime = request.app.state.runtime
        return [_build_zone_status(rt, z.zone_id) for z in rt.zones.values()]

    @app.get("/api/telemetry/current")
    async def telemetry_current(request: Request) -> Dict[str, Any]:
        rt: AuditorRuntime = request.app.state.runtime
        out: Dict[str, Any] = {}
        if rt.store is not None:
            for z in rt.zones.values():
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
                "zones": list(rt.zones),
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
                    if message.get("type") in ("point", "injection", "anomaly", "verdict", "log", "actuation"):
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


def _zone_payload(zone: ZoneDef) -> Dict[str, Any]:
    return {
        "zone_id": zone.zone_id,
        "building_id": zone.building_id,
        "floor_id": zone.floor_id,
        "area_m2": zone.area_m2,
        "ceiling_height_m": zone.ceiling_height_m,
        "volume_m3": round(zone.volume_m3, 2),
        "occupancy_max": zone.occupancy_max,
        "zone_type": zone.zone_type,
        "operating_start": zone.operating_start,
        "operating_end": zone.operating_end,
        "utility_rate_usd_per_kwh": zone.utility_rate_usd_per_kwh,
        "lighting_peak_kw": zone.lighting_peak_kw,
        "plug_peak_kw": zone.plug_peak_kw,
        "hvac_base_kw": zone.hvac_base_kw,
    }


def _parse_training_csv(content: bytes) -> List[TelemetryPoint]:
    """Parse a strict UTF-8 CSV with one timestamped row per zone sample."""
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    required = {
        "timestamp", "zone_id", "occupancy_count", "co2_ppm",
        "humidity_percent", "temp_indoor_c", "temp_outdoor_c",
        "hvac_kw", "lighting_kw", "plug_load_kw",
    }
    headers = set(reader.fieldnames or [])
    missing = required - headers
    if missing:
        raise ValueError("missing CSV columns: " + ", ".join(sorted(missing)))
    points: List[TelemetryPoint] = []
    for line, row in enumerate(reader, start=2):
        try:
            timestamp = (row.get("timestamp") or "").strip()
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            values = {key: float(row[key]) for key in required - {"timestamp", "zone_id"}}
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError("non-finite numeric value")
            if values["occupancy_count"] < 0 or not values["occupancy_count"].is_integer():
                raise ValueError("occupancy_count must be a non-negative integer")
            telemetry = Telemetry(
                **{**values, "occupancy_count": int(values["occupancy_count"])}
            )
            points.append(TelemetryPoint(
                timestamp=timestamp,
                building_id=row.get("building_id") or "csv_upload",
                zone_id=(row.get("zone_id") or "").strip(),
                telemetry=telemetry,
            ))
        except Exception as exc:
            raise ValueError(f"invalid CSV row {line}: {exc}") from exc
    if not points:
        raise ValueError("CSV contains no data rows")
    return points
