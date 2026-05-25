from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from vecl.provenance.events import EventType, ProvenanceEvent


class _AppendValidationLedger(Protocol):
    def get(self, event_id: str) -> ProvenanceEvent | None: ...

    def find_by_type(self, event_type: EventType) -> list[ProvenanceEvent]: ...


def _validate_append(ledger: _AppendValidationLedger, event: ProvenanceEvent) -> None:
    if ledger.get(event.event_id) is not None:
        raise ValueError(f"duplicate event_id: {event.event_id}")
    if event.event_type == EventType.SPARSE_UPDATE_APPLIED:
        prepared_id = event.payload.get("learning_event_id") or event.payload.get("event_id")
        if prepared_id and ledger.get(str(prepared_id)) is None:
            raise ValueError("SparseUpdateApplied requires a prepared learning event")
        if not prepared_id and not ledger.find_by_type(EventType.LEARNING_EVENT_PREPARED):
            raise ValueError("SparseUpdateApplied requires LearningEventPrepared")
    if event.event_type == EventType.LEARNING_EVENT_COMMITTED:
        prepared_id = event.payload.get("learning_event_id") or event.payload.get("event_id")
        if prepared_id and ledger.get(str(prepared_id)) is None:
            raise ValueError("LearningEventCommitted requires a prepared learning event")
    if event.event_type == EventType.ROLLBACK_PERFORMED:
        if not (
            event.payload.get("source_id")
            or event.payload.get("target_event_id")
            or event.payload.get("checkpoint_id")
        ):
            raise ValueError("RollbackPerformed must point to a source, event, or checkpoint")


@dataclass
class ProvenanceLedger:
    _events: list[ProvenanceEvent] = field(default_factory=list)
    _by_id: dict[str, ProvenanceEvent] = field(default_factory=dict)
    strict_tenant_queries: bool = False

    def append(self, event: ProvenanceEvent) -> ProvenanceEvent:
        _validate_append(self, event)
        previous = self._events[-1].chain_hash if self._events else "GENESIS"
        chained = event.with_chain_hash(previous)
        self._events.append(chained)
        self._by_id[chained.event_id] = chained
        return chained

    def events(self) -> tuple[ProvenanceEvent, ...]:
        return tuple(self._events)

    def get(self, event_id: str) -> ProvenanceEvent | None:
        return self._by_id.get(event_id)

    def require(self, event_id: str) -> ProvenanceEvent:
        event = self.get(event_id)
        if event is None:
            raise KeyError(event_id)
        return event

    def query_by_tenant(self, tenant_id: str | None = None) -> list[ProvenanceEvent]:
        if self.strict_tenant_queries and not tenant_id:
            raise PermissionError("tenant_id is required for tenant-scoped ledger query")
        if tenant_id is None:
            return list(self._events)
        return [event for event in self._events if event.tenant_id == tenant_id]

    def query_by_source_hash(self, source_hash: str) -> list[ProvenanceEvent]:
        return [
            event
            for event in self._events
            if event.payload.get("source_hash") == source_hash
            or event.payload.get("source_set_hash") == source_hash
            or source_hash in event.payload.get("source_hashes", [])
        ]

    def find_by_type(self, event_type: EventType) -> list[ProvenanceEvent]:
        return [event for event in self._events if event.event_type == event_type]

    def export_jsonl(self, events: Iterable[ProvenanceEvent] | None = None) -> str:
        selected = list(events) if events is not None else self._events
        return "\n".join(json.dumps(event.to_dict(), sort_keys=True) for event in selected)


class SqliteProvenanceLedger:
    """SQLite-backed ledger for local durable runs with serialized append authority."""

    def __init__(self, path: Path | str, *, strict_tenant_queries: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.strict_tenant_queries = strict_tenant_queries
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def __enter__(self) -> SqliteProvenanceLedger:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def append(self, event: ProvenanceEvent) -> ProvenanceEvent:
        _validate_append(self, event)
        previous = self._previous_chain_hash()
        chained = event.with_chain_hash(previous)
        payload = json.dumps(chained.to_dict(), sort_keys=True)
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO events (
                    event_id, event_type, tenant_id, chain_hash, payload_hash, timestamp, event_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chained.event_id,
                    chained.event_type.value,
                    chained.tenant_id,
                    chained.chain_hash,
                    chained.payload_hash,
                    chained.timestamp.isoformat(),
                    payload,
                ),
            )
        return chained

    def events(self) -> tuple[ProvenanceEvent, ...]:
        rows = self._connection.execute(
            "SELECT event_json FROM events ORDER BY position"
        ).fetchall()
        return tuple(_event_from_row(row) for row in rows)

    def get(self, event_id: str) -> ProvenanceEvent | None:
        row = self._connection.execute(
            "SELECT event_json FROM events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return _event_from_row(row) if row is not None else None

    def require(self, event_id: str) -> ProvenanceEvent:
        event = self.get(event_id)
        if event is None:
            raise KeyError(event_id)
        return event

    def query_by_tenant(self, tenant_id: str | None = None) -> list[ProvenanceEvent]:
        if self.strict_tenant_queries and not tenant_id:
            raise PermissionError("tenant_id is required for tenant-scoped ledger query")
        if tenant_id is None:
            return list(self.events())
        rows = self._connection.execute(
            "SELECT event_json FROM events WHERE tenant_id = ? ORDER BY position",
            (tenant_id,),
        ).fetchall()
        return [_event_from_row(row) for row in rows]

    def query_by_source_hash(self, source_hash: str) -> list[ProvenanceEvent]:
        return [
            event
            for event in self.events()
            if event.payload.get("source_hash") == source_hash
            or event.payload.get("source_set_hash") == source_hash
            or source_hash in event.payload.get("source_hashes", [])
        ]

    def find_by_type(self, event_type: EventType) -> list[ProvenanceEvent]:
        rows = self._connection.execute(
            "SELECT event_json FROM events WHERE event_type = ? ORDER BY position",
            (event_type.value,),
        ).fetchall()
        return [_event_from_row(row) for row in rows]

    def export_jsonl(self, events: Iterable[ProvenanceEvent] | None = None) -> str:
        selected = list(events) if events is not None else list(self.events())
        return "\n".join(json.dumps(event.to_dict(), sort_keys=True) for event in selected)

    def close(self) -> None:
        self._connection.close()

    def _ensure_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    position INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    chain_hash TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_json TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_tenant ON events(tenant_id, position)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, position)"
            )
            self._connection.execute("INSERT OR IGNORE INTO schema_version(version) VALUES (1)")

    def _previous_chain_hash(self) -> str:
        row = self._connection.execute(
            "SELECT chain_hash FROM events ORDER BY position DESC LIMIT 1"
        ).fetchone()
        return str(row["chain_hash"]) if row is not None else "GENESIS"


def _event_from_row(row: sqlite3.Row) -> ProvenanceEvent:
    return ProvenanceEvent.from_dict(json.loads(str(row["event_json"])))
