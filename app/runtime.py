"""End-to-end runtime orchestrator.

Wires the modules together into one always-on audit pipeline, one loop per
zone::

    TelemetryGenerator -> FeatureEngine -> Detector(IsolationForest)
                         -> DiagnosticClassifier -> Analytics accournal
                         -> Storage  -> WebSocket broadcast (dashboard/simulator)

The runtime owns all mutable live state (current verdicts, open anomaly
windows, closed anomaly records, injection controller) so the FastAPI layer
stays thin and stateless.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import joblib

from .analytics.reporting import AnalyticsEngine
from .config import Settings, settings as default_settings
from .features.engineering import FeatureEngine, WindowFeatures
from .ml.detector import Detector, load_detector
from .ml.rules import (
    PRIMARY_METRIC,
    BaselinePredictor,
    DiagnosticClassifier,
)
from .ml.training import BASELINE_FILENAME
from .schemas import (
    AnalyticsReport,
    AnomalyRecord,
    InjectionState,
    SimulatorCommand,
    Verdict,
)
from .storage.store import TelemetryStore, create_store
from .telemetry.generator import InjectionController, TelemetryGenerator
from .zones import ZONE_BY_ID, ZONE_CATALOG, ZoneDef


@dataclass
class Accrual:
    """Accumulating state for one open (currently-flagged) anomaly window."""

    zone_id: str
    start_timestamp: str
    diagnosis: str
    severity: str
    marker: str
    min_score: float
    peak_kw: float = 0.0
    energy_wasted_kwh: float = 0.0
    sample_count: int = 0


class Logger:
    """Tiny bounded transcript of pipeline events (for the simulator readout)."""

    def __init__(self, maxlen: int = 40) -> None:
        self._items: List[str] = []
        self._maxlen = maxlen

    def add(self, message: str) -> None:
        self._items.append(message)
        del self._items[:-self._maxlen]

    def snapshot(self) -> List[str]:
        return list(self._items)


class AuditorRuntime:
    """Create once per process; run the zone generators as background tasks."""

    def __init__(self, settings: Settings = default_settings) -> None:
        self.settings = settings
        self.controller = InjectionController()
        self.analytics = AnalyticsEngine(settings, rate=settings.utility_rate_usd_per_kwh)
        self.log = Logger()

        # Wiring filled in by ``create``.
        self.store: Optional[TelemetryStore] = None
        self.detector: Optional[Detector] = None
        self.engines: Dict[str, FeatureEngine] = {}
        self.predictor: Optional[BaselinePredictor] = None
        self.classifier: Optional[DiagnosticClassifier] = None

        # Live state.
        self.current: Dict[str, Verdict] = {}
        self.accruals: Dict[str, Accrual] = {}
        self.anomalies: List[AnomalyRecord] = []
        self.samples_processed: int = 0
        self._started_at: float = time.monotonic()

        # WebSocket fan-out.
        self._queues: List[asyncio.Queue] = []

    # ------------------------------------------------------------- lifecycle
    @classmethod
    async def create(cls, settings: Settings = default_settings) -> "AuditorRuntime":
        """Build the runtime: load cached models, create the store, wire up."""
        rt = cls(settings)

        rt.detector = load_detector(settings.model_dir, settings)
        rt.predictor = BaselinePredictor(_load_baseline_models(settings))
        rt.classifier = DiagnosticClassifier(ZONE_BY_ID)
        rt.engines = {
            zone.zone_id: FeatureEngine(
                window_points=settings.window_points,
                min_samples=settings.window_min_samples,
                fast_window_points=settings.fast_window_points,
            )
            for zone in ZONE_CATALOG
        }
        rt._prewarm_engines(settings)
        rt.store = await create_store(settings)
        rt.log.add(
            f"runtime ready · detector={'loaded' if rt.detector else 'missing'} · "
            f"storage={rt.store.kind if rt.store else 'n/a'}"
        )
        return rt

    def baseline_model_zones(self) -> List[str]:
        return list(self.engines)

    def _prewarm_engines(self, settings: Settings) -> None:
        """Fill each zone's sliding window with synthetic baseline history.

        The Isolation Forest is trained on full 20-minute windows.  Without
        this, a freshly booted runtime scores *short* partial windows as
        anomalous until 20 minutes of live data accumulates (a flood of
        ``GENERIC_ENERGY_ANOMALY`` verdicts).  Generating the history offline
        at startup makes the very first live verdict statistically sound and
        gives the dashboard a meaningful 20-minute context immediately.
        """
        base = datetime.now(timezone.utc)
        interval_h = settings.emit_interval_seconds / 3600.0
        start_hf = (base.hour + base.minute / 60.0 + base.second / 3600.0)
        for zone in ZONE_CATALOG:
            engine = self.engines[zone.zone_id]
            gen = TelemetryGenerator(
                settings.building_id,
                zone,
                self.controller,
                sink=self._noop_sink,
                interval=settings.emit_interval_seconds,
                seed=1337,
            )
            history = [
                gen._normal_sample(base, (start_hf + i * interval_h) % 24.0)
                for i in range(settings.window_points)
            ]
            engine.seed(history)

    @staticmethod
    async def _noop_sink(*args, **kwargs) -> None:
        return None

    def uptime_seconds(self) -> float:
        return time.monotonic() - self._started_at

    async def shutdown(self) -> None:
        """Release any backing-store resources (connection pools / HTTP clients)."""
        if self.store is not None:
            await self.store.close()

    # ------------------------------------------------------------- websockets
    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._queues.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        try:
            self._queues.remove(queue)
        except ValueError:  # pragma: no cover - defensive
            pass

    def _broadcast(self, message: dict) -> None:
        for queue in self._queues:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:  # slow client - drop the frame
                pass

    # ---------------------------------------------------------------- control
    def apply_command(self, cmd: SimulatorCommand) -> InjectionState:
        """Apply an injection-panel command and return the affected zone state."""
        kind = cmd.anomaly
        if cmd.action == "inject" and kind is not None:
            self.controller.inject(cmd.zone, kind)
            self.log.add(f"inject {kind.value} -> {cmd.zone}")
        elif cmd.action == "clear" and kind is not None:
            self.controller.clear(cmd.zone, kind)
            self.log.add(f"clear {kind.value} -> {cmd.zone}")
        elif cmd.action == "reset":
            self.controller.clear_all(cmd.zone)
            self.log.add(f"reset {cmd.zone}")
            self.flush_zone(cmd.zone)
        elif cmd.action == "reset_all":
            self.controller.reset()
            self.log.add("reset all zones")
            for zone_id in self.engines:
                self.flush_zone(zone_id)

        state = self.injection_state(cmd.zone)
        self._broadcast({"type": "injection", "state": state.model_dump()})
        self._broadcast({"type": "log", "lines": self.log.snapshot()})
        return state

    def flush_zone(self, zone_id: str) -> None:
        """Drop the zone's feature window + finalise any open accrual.

        Lets a manual reset give instantaneous demo feedback instead of waiting
        up to the full 15-30 min sliding window to age out.  The waste already
        accrued is closed as an anomaly record.
        """
        engine = self.engines.get(zone_id)
        if engine is not None:
            engine.reset()
        accrual = self.accruals.pop(zone_id, None)
        if accrual is not None:
            self._finalize_accrual(
                zone_id,
                accrual,
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )
        verdict = self.current.get(zone_id)
        if verdict is not None:
            self.current[zone_id] = verdict.model_copy(
                update={
                    "is_anomaly": False,
                    "marker": "green",
                    "severity": "normal",
                    "diagnosis": None,
                    "confidence": 0.0,
                    "explanation": "zone reset by simulator",
                }
            )

    def injection_state(self, zone: str) -> InjectionState:
        active = self.controller.active_set(zone)
        return InjectionState(
            zone=zone,
            active=sorted(active),
            injected_state=self.controller.state_string(zone),
        )

    def controller_snapshot(self) -> Dict[str, dict]:
        return {
            zone_id: self.injection_state(zone_id).model_dump()
            for zone_id in (z.zone_id for z in ZONE_CATALOG)
        }

    # ------------------------------------------------------------------ sink
    async def _on_sample(self, zone: ZoneDef, point) -> None:
        """Per-sample pipeline stage (the generator sink)."""
        self.samples_processed += 1
        engine = self.engines[zone.zone_id]
        fv = engine.update(point)

        scored = point
        if fv is not None and self.detector is not None and self.classifier is not None:
            score = self.detector.score(fv.X)
            verdict = self.classifier.classify(
                zone, fv, self.predictor.predicted_kws(zone.zone_id, fv), score
            )
            scored = point.model_copy(
                update={
                    "marker": verdict.marker,
                    "score": verdict.iforest_score,
                    "diagnosis": verdict.diagnosis,
                }
            )
            self.add_verdict(zone, fv, verdict, point)
            self._broadcast({"type": "verdict", "verdict": verdict.model_dump()})

        # Persist the scored sample (fire-and-forget so the HTTP store backends
        # never stall the synchronous 1 Hz generation loop).
        if self.store is not None:
            asyncio.create_task(self.store.insert_telemetry(scored))

        self._broadcast({"type": "point", "point": scored.model_dump()})

    # ---------------------------------------------------- anomaly accrual
    def add_verdict(
        self, zone: ZoneDef, fv: WindowFeatures, verdict: Verdict, point
    ) -> None:
        """Track the latest verdict and accrue/close anomaly windows."""
        self.current[zone.zone_id] = verdict

        if not verdict.is_anomaly:
            self._close_accrual(zone, fv)
            return

        accrual = self.accruals.get(zone.zone_id)
        metric = PRIMARY_METRIC.get(verdict.diagnosis, "hvac_kw")
        actual_kw = float(getattr(point.telemetry, metric))
        expected_kw = self.predictor.predict(zone.zone_id, metric, fv)
        interval_h = self.settings.emit_interval_seconds / 3600.0

        if accrual is None:
            self.accruals[zone.zone_id] = Accrual(
                zone_id=zone.zone_id,
                start_timestamp=verdict.timestamp,
                diagnosis=verdict.diagnosis or "GENERIC_ENERGY_ANOMALY",
                severity=verdict.severity,
                marker=verdict.marker,
                min_score=verdict.iforest_score,
                peak_kw=actual_kw,
                energy_wasted_kwh=max(0.0, actual_kw - expected_kw) * interval_h,
                sample_count=1,
            )
            self.log.add(
                f"{zone.zone_id}: anomaly opened -> {verdict.diagnosis or 'generic'}"
            )
            return

        accrual.peak_kw = max(accrual.peak_kw, actual_kw)
        accrual.min_score = min(accrual.min_score, verdict.iforest_score)
        accrual.energy_wasted_kwh += max(0.0, actual_kw - expected_kw) * interval_h
        accrual.sample_count += 1

        # Upgrade the accrual's tag if the ruleset has now resolved a more
        # specific root cause than the generic isolation-forest flag that
        # opened the window.
        if verdict.diagnosis and verdict.diagnosis != "GENERIC_ENERGY_ANOMALY":
            accrual.diagnosis = verdict.diagnosis
            accrual.severity = verdict.severity
            accrual.marker = verdict.marker

    def _close_accrual(self, zone: ZoneDef, fv: WindowFeatures) -> None:
        accrual = self.accruals.pop(zone.zone_id, None)
        if accrual is None:
            return
        self._finalize_accrual(zone.zone_id, accrual, fv.timestamp)

    def _finalize_accrual(
        self, zone_id: str, accrual: Accrual, end_timestamp: str
    ) -> Optional[AnomalyRecord]:
        """Persist a closed anomaly window as a monetised record."""
        if accrual.energy_wasted_kwh <= 0.0:
            return None
        duration_h = max(0.0, _seconds_between(accrual.start_timestamp, end_timestamp)) / 3600.0
        record = AnomalyRecord(
            start_timestamp=accrual.start_timestamp,
            end_timestamp=end_timestamp,
            zone_id=accrual.zone_id,
            diagnosis=accrual.diagnosis,
            severity=accrual.severity,
            marker=accrual.marker,
            iforest_score=accrual.min_score,
            energy_wasted_kwh=round(accrual.energy_wasted_kwh, 4),
            financial_waste_usd=round(
                accrual.energy_wasted_kwh * self.settings.utility_rate_usd_per_kwh, 4
            ),
            peak_kw=round(accrual.peak_kw, 4),
            sample_count=accrual.sample_count,
            duration_hours=round(duration_h, 4),
            details={
                "interval_seconds": self.settings.emit_interval_seconds,
                "injected_state": self.controller.state_string(zone_id),
            },
        )
        self.anomalies.append(record)
        self._broadcast({"type": "anomaly", "anomaly": record.model_dump()})
        self.log.add(
            f"{record.zone_id}: closed {record.diagnosis} · "
            f"{record.energy_wasted_kwh:.2f} kWh · ${record.financial_waste_usd:.2f}"
        )
        if self.store is not None:
            asyncio.create_task(self.store.insert_anomaly(record))
        return record

    # -------------------------------------------------------------- reporting
    def current_report(self) -> AnalyticsReport:
        markers = {z: v.marker for z, v in self.current.items()}
        return self.analytics.build_report(
            self.anomalies, markers, self.uptime_seconds()
        )

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        """Start one generation task per zone and run until cancelled."""
        tasks = [
            asyncio.create_task(
                TelemetryGenerator(
                    building_id=self.settings.building_id,
                    zone=zone,
                    controller=self.controller,
                    sink=self._on_sample,
                    interval=self.settings.emit_interval_seconds,
                    seed=self.settings.random_state + idx,
                ).run()
            )
            for idx, zone in enumerate(ZONE_CATALOG)
        ]
        await asyncio.gather(*tasks)


def _load_baseline_models(settings: Settings) -> Dict[str, object]:
    """Load the fitted per-zone baseline consumption regressors."""
    path = settings.model_dir / BASELINE_FILENAME
    if not path.exists():
        return {}
    payload = joblib.load(path)
    return payload.get("zones", {})


def _seconds_between(start_iso: str, end_iso: str) -> float:
    """Wall-clock seconds between two ISO-8601 timestamps."""

    def _ts(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    return (_ts(end_iso) - _ts(start_iso)).total_seconds()