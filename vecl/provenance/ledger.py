from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash


class _AppendValidationLedger(Protocol):
    def get(self, event_id: str) -> ProvenanceEvent | None: ...

    def find_by_type(self, event_type: EventType) -> list[ProvenanceEvent]: ...


class _VerifiedPrefix:
    """Minimal read-only ledger view over an already verified chain prefix."""

    def __init__(self) -> None:
        self._events: list[ProvenanceEvent] = []
        self._by_id: dict[str, ProvenanceEvent] = {}

    def get(self, event_id: str) -> ProvenanceEvent | None:
        return self._by_id.get(event_id)

    def find_by_type(self, event_type: EventType) -> list[ProvenanceEvent]:
        return [event for event in self._events if event.event_type == event_type]

    def add_verified(self, event: ProvenanceEvent) -> None:
        self._events.append(event)
        self._by_id[event.event_id] = event


def _validate_append(ledger: _AppendValidationLedger, event: ProvenanceEvent) -> None:
    if ledger.get(event.event_id) is not None:
        raise ValueError(f"duplicate event_id: {event.event_id}")
    for parent_event_id in event.parent_event_ids:
        parent = ledger.get(parent_event_id)
        if parent is None:
            raise ValueError(f"parent event does not exist: {parent_event_id}")
        if parent.tenant_id != event.tenant_id:
            raise ValueError(f"parent event tenant does not match: {parent_event_id}")
    if event.event_type == EventType.SPARSE_UPDATE_APPLIED:
        _require_event_reference(
            ledger,
            event,
            field_name="learning_event_id",
            expected_type=EventType.LEARNING_EVENT_PREPARED,
        )
    if event.event_type == EventType.LEARNING_EVENT_COMMITTED:
        prepared = _require_event_reference(
            ledger,
            event,
            field_name="learning_event_id",
            expected_type=EventType.LEARNING_EVENT_PREPARED,
        )
        applied = _require_event_reference(
            ledger,
            event,
            field_name="sparse_update_event_id",
            expected_type=EventType.SPARSE_UPDATE_APPLIED,
        )
        if applied.payload.get("learning_event_id") != prepared.event_id:
            raise ValueError(
                "committed sparse update does not reference the prepared learning event"
            )
    if event.event_type == EventType.ROLLBACK_PERFORMED:
        if not (
            event.payload.get("source_id")
            or event.payload.get("target_event_id")
            or event.payload.get("checkpoint_id")
        ):
            raise ValueError("RollbackPerformed must point to a source, event, or checkpoint")


def _require_event_reference(
    ledger: _AppendValidationLedger,
    event: ProvenanceEvent,
    *,
    field_name: str,
    expected_type: EventType,
) -> ProvenanceEvent:
    raw_event_id = event.payload.get(field_name)
    if not isinstance(raw_event_id, str) or not raw_event_id:
        raise ValueError(f"{event.event_type.value} requires non-empty {field_name}")
    referenced = ledger.get(raw_event_id)
    if referenced is None:
        raise ValueError(
            f"{event.event_type.value} requires existing prepared event of type "
            f"{expected_type.value}"
        )
    if referenced.event_type != expected_type:
        raise ValueError(f"{field_name} must reference {expected_type.value}")
    if referenced.tenant_id != event.tenant_id:
        raise ValueError(f"{field_name} tenant does not match event tenant")
    return referenced


def verify_event_chain(events: Sequence[ProvenanceEvent]) -> bool:
    """Verify payload hashes, chain hashes, ordering, and event-id uniqueness."""

    previous_chain_hash = "GENESIS"
    seen_events: dict[str, ProvenanceEvent] = {}
    prefix = _VerifiedPrefix()
    for position, event in enumerate(events):
        if event.event_id in seen_events:
            raise ValueError(f"duplicate event id at position {position}: {event.event_id}")
        for parent_event_id in event.parent_event_ids:
            parent = seen_events.get(parent_event_id)
            if parent is None:
                raise ValueError(
                    f"parent event is missing or not earlier at position {position}: "
                    f"{parent_event_id}"
                )
            if parent.tenant_id != event.tenant_id:
                raise ValueError(
                    f"parent event tenant mismatch at position {position}: {parent_event_id}"
                )
        stable_payload_hash = stable_hash(event.payload)
        if stable_payload_hash != event.payload_hash:
            raise ValueError(f"payload hash mismatch at position {position}: {event.event_id}")
        expected_chain_hash = event.with_chain_hash(previous_chain_hash).chain_hash
        if not event.chain_hash or event.chain_hash != expected_chain_hash:
            raise ValueError(f"chain hash mismatch at position {position}: {event.event_id}")
        _validate_append(prefix, event)
        previous_chain_hash = event.chain_hash
        seen_events[event.event_id] = event
        prefix.add_verified(event)
    return True


@dataclass
class ProvenanceLedger:
    _events: list[ProvenanceEvent] = field(default_factory=list)
    _by_id: dict[str, ProvenanceEvent] = field(default_factory=dict)
    strict_tenant_queries: bool = False

    def append(self, event: ProvenanceEvent) -> ProvenanceEvent:
        if self._events:
            self.verify_integrity()
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

    def verify_integrity(self) -> bool:
        return verify_event_chain(tuple(self._events))


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
        self.verify_integrity()

    def __enter__(self) -> SqliteProvenanceLedger:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def append(self, event: ProvenanceEvent) -> ProvenanceEvent:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self.verify_integrity()
            _validate_append(self, event)
            previous = self._previous_chain_hash()
            chained = event.with_chain_hash(previous)
            payload = json.dumps(chained.to_dict(), sort_keys=True)
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
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
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

    def verify_integrity(self) -> bool:
        rows = self._connection.execute(
            """
            SELECT event_id, event_type, tenant_id, chain_hash, payload_hash, timestamp, event_json
            FROM events
            ORDER BY position
            """
        ).fetchall()
        events = tuple(_event_from_row(row) for row in rows)
        for position, (row, event) in enumerate(zip(rows, events, strict=True)):
            stored_columns = (
                str(row["event_id"]),
                str(row["event_type"]),
                str(row["tenant_id"]),
                str(row["chain_hash"]),
                str(row["payload_hash"]),
                str(row["timestamp"]),
            )
            event_columns = (
                event.event_id,
                event.event_type.value,
                event.tenant_id,
                event.chain_hash,
                event.payload_hash,
                event.timestamp.isoformat(),
            )
            if stored_columns != event_columns:
                raise ValueError(f"event column mismatch at position {position}: {event.event_id}")
        return verify_event_chain(events)

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
