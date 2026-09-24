"""Storage layer (Module 5 persistence) - telemetry + anomaly persistence.

Three interchangeable backends implement the same async surface:

* :class:`MemoryStore`       - bounded ring buffers; always the safe fallback.
* :class:`PostgresStore`     - full SQL persistence via ``asyncpg``.
* :class:`SupabaseHttpStore` - PostgREST REST bridge (Supabase project URL +
                               service-role / secret key), no DB driver needed.

``create_store`` picks the backend at startup: explicit ``DATABASE_URL`` wins,
then a Supabase ``url`` + ``secretkey`` pair, then memory.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from ..config import Settings
from ..schemas import AnomalyRecord, TelemetryPoint


def _parse_ts(timestamp: str) -> datetime:
    """ISO-8601 string -> naive/aware UTC datetime for database columns."""
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


class TelemetryStore:
    """Base interface for persisting telemetry and anomaly records."""

    kind: str = "base"

    async def is_ready(self) -> bool:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError

    async def insert_telemetry(self, record: TelemetryPoint) -> None:
        raise NotImplementedError

    async def insert_anomaly(self, record: AnomalyRecord) -> None:
        raise NotImplementedError

    async def recent_telemetry(
        self, zone_id: Optional[str] = None, limit: int = 500
    ) -> List[TelemetryPoint]:
        raise NotImplementedError

    async def recent_anomalies(self, limit: int = 200) -> List[AnomalyRecord]:
        raise NotImplementedError

    async def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "ready": await self.is_ready(),
            "telemetry_rows": None,
            "anomaly_rows": None,
        }


class MemoryStore(TelemetryStore):
    """Bounded in-memory ring buffers for live-demo history."""

    kind = "memory"

    def __init__(self, settings: Settings) -> None:
        self._telemetry: Deque[TelemetryPoint] = deque(
            maxlen=settings.max_dashboard_history
        )
        self._anomalies: Deque[AnomalyRecord] = deque(
            maxlen=settings.max_anomaly_history
        )
        self._ready = True

    async def is_ready(self) -> bool:
        return self._ready

    async def close(self) -> None:
        """No pooled resources to release - present for interface parity."""

    async def insert_telemetry(self, record: TelemetryPoint) -> None:
        self._telemetry.append(record)

    async def insert_anomaly(self, record: AnomalyRecord) -> None:
        self._anomalies.append(record)

    async def recent_telemetry(
        self, zone_id: Optional[str] = None, limit: int = 500
    ) -> List[TelemetryPoint]:
        if zone_id:
            return [t for t in self._telemetry if t.zone_id == zone_id][-limit:]
        return list(self._telemetry)[-limit:]

    async def recent_anomalies(self, limit: int = 200) -> List[AnomalyRecord]:
        return list(self._anomalies)[-limit:]

    async def snapshot(self) -> Dict[str, Any]:
        base = await super().snapshot()
        base["telemetry_rows"] = len(self._telemetry)
        base["anomaly_rows"] = len(self._anomalies)
        return base


class PostgresStore(TelemetryStore):
    """Asyncpg-backed SQL persistence (Supabase/libsql/RDS/local Postgres)."""

    kind = "postgres"

    def __init__(self, settings: Settings, database_url: str) -> None:
        self.dsn = database_url
        self.settings = settings
        self._pool: Optional[Any] = None
        self._ready = False

    async def connect(self) -> None:
        """Open the pool and create the two tables if they do not exist."""
        import asyncpg

        self._pool = await asyncpg.create_pool(
            self.dsn, min_size=1, max_size=5, command_timeout=10
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry (
                    id          BIGSERIAL PRIMARY KEY,
                    building_id TEXT NOT NULL,
                    zone_id     TEXT NOT NULL,
                    ts          TIMESTAMPTZ NOT NULL,
                    payload     JSONB NOT NULL,
                    marker      TEXT NOT NULL DEFAULT 'green',
                    score       DOUBLE PRECISION NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_telemetry_zone_ts
                    ON telemetry (zone_id, ts DESC);
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS anomalies (
                    id                 BIGSERIAL PRIMARY KEY,
                    zone_id            TEXT NOT NULL,
                    start_ts           TIMESTAMPTZ NOT NULL,
                    end_ts             TIMESTAMPTZ,
                    diagnosis          TEXT NOT NULL,
                    severity           TEXT NOT NULL,
                    marker             TEXT NOT NULL DEFAULT 'green',
                    energy_wasted_kwh  DOUBLE PRECISION NOT NULL DEFAULT 0,
                    financial_waste_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
                    iforest_score      DOUBLE PRECISION NOT NULL DEFAULT 0,
                    details            JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS idx_anomalies_end ON anomalies (end_ts DESC);
                """
            )
        self._ready = True

    async def is_ready(self) -> bool:
        return self._ready

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        self._ready = False

    async def insert_telemetry(self, record: TelemetryPoint) -> None:
        if not self._ready:
            return
        payload = record.model_dump(exclude={"timestamp", "building_id", "zone_id"})
        await self._pool.execute(
            "INSERT INTO telemetry (building_id, zone_id, ts, payload, marker, score) "
            "VALUES ($1, $2, $3, $4::jsonb, $5, $6)",
            record.building_id,
            record.zone_id,
            _parse_ts(record.timestamp),
            json.dumps(payload),
            record.marker,
            record.score,
        )

    async def insert_anomaly(self, record: AnomalyRecord) -> None:
        if not self._ready:
            return
        await self._pool.execute(
            "INSERT INTO anomalies "
            "(zone_id, start_ts, end_ts, diagnosis, severity, marker, "
            " energy_wasted_kwh, financial_waste_usd, iforest_score, details) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb)",
            record.zone_id,
            _parse_ts(record.start_timestamp),
            _parse_ts(record.end_timestamp),
            record.diagnosis,
            record.severity,
            record.marker,
            record.energy_wasted_kwh,
            record.financial_waste_usd,
            record.iforest_score,
            json.dumps(record.model_dump(exclude={"zone_id", "start_timestamp", "end_timestamp"})),
        )

    async def recent_telemetry(
        self, zone_id: Optional[str] = None, limit: int = 500
    ) -> List[TelemetryPoint]:
        if not self._ready or self._pool is None:
            return []
        if zone_id:
            rows = await self._pool.fetch(
                "SELECT * FROM telemetry WHERE zone_id=$1 ORDER BY ts DESC LIMIT $2",
                zone_id,
                limit,
            )
        else:
            rows = await self._pool.fetch(
                "SELECT * FROM telemetry ORDER BY ts DESC LIMIT $2", limit
            )
        points: List[TelemetryPoint] = []
        for r in reversed(rows):
            payload = dict(r["payload"])
            points.append(
                TelemetryPoint.model_validate(
                    {
                        "timestamp": r["ts"].isoformat(),
                        "building_id": r["building_id"],
                        "zone_id": r["zone_id"],
                        "marker": r["marker"],
                        "score": r["score"],
                        **payload,
                    }
                )
            )
        return points

    async def recent_anomalies(self, limit: int = 200) -> List[AnomalyRecord]:
        if not self._ready or self._pool is None:
            return []
        rows = await self._pool.fetch(
            "SELECT * FROM anomalies ORDER BY end_ts DESC LIMIT $1", limit
        )
        out: List[AnomalyRecord] = []
        for r in rows:
            details = dict(r["details"])
            out.append(
                AnomalyRecord.model_validate(
                    {
                        "zone_id": r["zone_id"],
                        "start_timestamp": r["start_ts"].isoformat(),
                        "end_timestamp": r["end_ts"].isoformat(),
                        "diagnosis": r["diagnosis"],
                        "severity": r["severity"],
                        "marker": r["marker"],
                        "energy_wasted_kwh": r["energy_wasted_kwh"],
                        "financial_waste_usd": r["financial_waste_usd"],
                        "iforest_score": r["iforest_score"],
                        **details,
                    }
                )
            )
        return out

    async def snapshot(self) -> Dict[str, Any]:
        base = await super().snapshot()
        base["ready"] = self._ready
        if self._ready and self._pool is not None:
            base["telemetry_rows"] = await self._pool.fetchval("SELECT COUNT(*) FROM telemetry")
            base["anomaly_rows"] = await self._pool.fetchval("SELECT COUNT(*) FROM anomalies")
        return base


class SupabaseHttpStore(TelemetryStore):
    """PostgREST-backed persistence for a hosted Supabase project.

    Uses the project URL + service-role/secret key pair with async ``httpx``.
    Tables ``telemetry`` and ``anomalies`` are NOT created by PostgREST (REST
    cannot run DDL); run ``scripts/setup_supabase.sql`` in the SQL editor once.
    """

    kind = "supabase"

    def __init__(self, settings: Settings, base_url: str, apikey: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.apikey = apikey
        self.settings = settings
        self._ready = False
        self._client: Optional[Any] = None

    def _headers(self) -> Dict[str, str]:
        return {
            "apikey": self.apikey,
            "Authorization": f"Bearer {self.apikey}",
            "Content-Type": "application/json",
        }

    async def connect(self) -> None:
        """Check the tables exist with a cheap paginated read."""
        import httpx

        self._client = httpx.AsyncClient(timeout=10.0)
        resp = await self._client.get(
            f"{self.base_url}/rest/v1/telemetry",
            params={"select": "id", "limit": "1"},
            headers=self._headers(),
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Supabase PostgREST returned {resp.status_code}: {resp.text[:200]}"
            )
        self._ready = True

    async def is_ready(self) -> bool:
        return self._ready

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._ready = False

    async def insert_telemetry(self, record: TelemetryPoint) -> None:
        if not self._ready or self._client is None:
            return
        row = {
            "building_id": record.building_id,
            "zone_id": record.zone_id,
            "ts": _parse_ts(record.timestamp).isoformat(),
            "payload": record.model_dump(
                exclude={"timestamp", "building_id", "zone_id"}
            ),
            "marker": record.marker,
            "score": record.score,
        }
        await self._client.post(
            f"{self.base_url}/rest/v1/telemetry",
            json=row,
            headers=self._headers(),
        )

    async def insert_anomaly(self, record: AnomalyRecord) -> None:
        if not self._ready or self._client is None:
            return
        details = record.model_dump(exclude={"zone_id", "start_timestamp", "end_timestamp"})
        row = {
            "zone_id": record.zone_id,
            "start_ts": _parse_ts(record.start_timestamp).isoformat(),
            "end_ts": _parse_ts(record.end_timestamp).isoformat(),
            "diagnosis": record.diagnosis,
            "severity": record.severity,
            "marker": record.marker,
            "energy_wasted_kwh": record.energy_wasted_kwh,
            "financial_waste_usd": record.financial_waste_usd,
            "iforest_score": record.iforest_score,
            "details": details,
        }
        await self._client.post(
            f"{self.base_url}/rest/v1/anomalies",
            json=row,
            headers=self._headers(),
        )

    async def recent_telemetry(
        self, zone_id: Optional[str] = None, limit: int = 500
    ) -> List[TelemetryPoint]:
        if not self._ready or self._client is None:
            return []
        params: Dict[str, str] = {"select": "*", "order": "ts.desc", "limit": str(limit)}
        if zone_id:
            params["zone_id"] = f"eq.{zone_id}"
        resp = await self._client.get(
            f"{self.base_url}/rest/v1/telemetry",
            params=params,
            headers=self._headers(),
        )
        if resp.status_code >= 400:
            return []
        rows = resp.json()
        points: List[TelemetryPoint] = []
        for r in reversed(rows):
            payload = dict(r.get("payload") or {})
            points.append(
                TelemetryPoint.model_validate(
                    {
                        "timestamp": r["ts"],
                        "building_id": r["building_id"],
                        "zone_id": r["zone_id"],
                        "marker": r.get("marker", "green"),
                        "score": r.get("score", 0.0),
                        **payload,
                    }
                )
            )
        return points

    async def recent_anomalies(self, limit: int = 200) -> List[AnomalyRecord]:
        if not self._ready or self._client is None:
            return []
        params = {"select": "*", "order": "end_ts.desc", "limit": str(limit)}
        resp = await self._client.get(
            f"{self.base_url}/rest/v1/anomalies",
            params=params,
            headers=self._headers(),
        )
        if resp.status_code >= 400:
            return []
        rows = resp.json()
        out: List[AnomalyRecord] = []
        for r in reversed(rows):
            details = dict(r.get("details") or {})
            out.append(
                AnomalyRecord.model_validate(
                    {
                        "zone_id": r["zone_id"],
                        "start_timestamp": r["start_ts"],
                        "end_timestamp": r["end_ts"],
                        "diagnosis": r["diagnosis"],
                        "severity": r["severity"],
                        "marker": r.get("marker", "green"),
                        "energy_wasted_kwh": r.get("energy_wasted_kwh", 0.0),
                        "financial_waste_usd": r.get("financial_waste_usd", 0.0),
                        "iforest_score": r.get("iforest_score", 0.0),
                        **details,
                    }
                )
            )
        return out

    async def snapshot(self) -> Dict[str, Any]:
        base = await super().snapshot()
        base["ready"] = self._ready
        if self._ready and self._client is not None:
            for table, key in (("telemetry", "telemetry_rows"), ("anomalies", "anomaly_rows")):
                resp = await self._client.get(
                    f"{self.base_url}/rest/v1/{table}",
                    params={"select": "id", "limit": "1", "order": "id.desc"},
                    headers={**self._headers(), "Range": "0-0", "Prefer": "count=exact"},
                )
                if resp.status_code < 400:
                    crange = resp.headers.get("content-range", "")
                    base[key] = int(crange.split("/")[-1]) if "/" in crange else None
        return base


async def create_store(settings: Settings) -> TelemetryStore:
    """Build the appropriate store for the configured environment.

    Precedence:
      1. An explicit ``database_url`` -> PostgresStore (asyncpg).
      2. A Supabase ``url`` + ``secretkey`` pair -> SupabaseHttpStore (PostgREST).
      3. In-memory store (always a safe fallback for demos and CI).
    """
    if settings.database_url:
        store = PostgresStore(settings, settings.database_url)
        try:
            await store.connect()
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[storage] Postgres unavailable ({exc}); falling back to memory")
            await store.close()
            return MemoryStore(settings)
        return store

    if settings.supabase_url and settings.supabase_key:
        store = SupabaseHttpStore(settings, settings.supabase_url, settings.supabase_key)
        try:
            await store.connect()
            print(f"[storage] Supabase PostgREST ready at {settings.supabase_url}")
            return store
        except Exception as exc:
            print(f"[storage] Supabase REST unavailable ({exc}); falling back to memory")
            await store.close()
            return MemoryStore(settings)

    return MemoryStore(settings)