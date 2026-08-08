from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import SqliteProvenanceLedger
from vecl.runtime.checkpoints import DiskCheckpointStore, RollbackService
from vecl.specialists.artifacts import ArtifactRecord, ContentAddressedStore


def test_sqlite_ledger_reopens_with_chain_hash_continuity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger_path = tmp_path / ".vecl" / "provenance.sqlite3"
    ledger = SqliteProvenanceLedger(ledger_path)
    first = ledger.append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant-a", "tester", {"n": 1})
    )
    second = ledger.append(
        ProvenanceEvent(
            EventType.ARTIFACT_PRODUCED,
            "tenant-a",
            "tester",
            {
                "artifact_id": "artifact-a",
                "producer_specialist_id": "tester",
                "producer_version": "v1",
                "input_hash": "input",
                "output_hash": "output",
                "output_path": "/tmp/artifact",
                "output_format": "txt",
                "created_at": "2026-05-20T00:00:00+00:00",
                "parent_event_id": first.event_id,
            },
            parent_event_ids=[first.event_id],
        )
    )
    ledger.close()

    reopened = SqliteProvenanceLedger(ledger_path)
    assert [event.event_id for event in reopened.events()] == [first.event_id, second.event_id]
    assert reopened.require(second.event_id).chain_hash == second.chain_hash
    third = reopened.append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant-a", "tester", {"n": 3})
    )
    assert third.chain_hash != second.chain_hash
    assert reopened.query_by_tenant("tenant-a")[-1].event_id == third.event_id


def test_sqlite_ledger_rejects_tampered_event_history_on_reopen(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger_path = tmp_path / ".vecl" / "provenance.sqlite3"
    with SqliteProvenanceLedger(ledger_path) as ledger:
        event = ledger.append(
            ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant-a", "tester", {"n": 1})
        )

    connection = sqlite3.connect(ledger_path)
    row = connection.execute(
        "SELECT event_json FROM events WHERE event_id = ?", (event.event_id,)
    ).fetchone()
    assert row is not None
    payload = json.loads(str(row[0]))
    payload["payload"]["n"] = 2
    connection.execute(
        "UPDATE events SET event_json = ? WHERE event_id = ?",
        (json.dumps(payload, sort_keys=True), event.event_id),
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="payload hash"):
        SqliteProvenanceLedger(ledger_path)


def test_sqlite_ledger_validates_against_persisted_prepared_event(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger_path = tmp_path / ".vecl" / "provenance.sqlite3"
    ledger = SqliteProvenanceLedger(ledger_path)
    ledger.append(
        ProvenanceEvent(
            EventType.LEARNING_EVENT_PREPARED,
            "tenant-a",
            "trainer",
            {"event_id": "learn-1"},
            event_id="learn-1",
        )
    )
    ledger.close()

    reopened = SqliteProvenanceLedger(ledger_path)
    update = reopened.append(
        ProvenanceEvent(
            EventType.SPARSE_UPDATE_APPLIED,
            "tenant-a",
            "trainer",
            {
                "learning_event_id": "learn-1",
                "selected_slots": [0],
                "eligible_slots": [0],
                "delta_records": [{"slot_id": 0, "delta": -0.1}],
            },
        )
    )

    assert update.event_type == EventType.SPARSE_UPDATE_APPLIED


def test_sqlite_ledger_supports_context_manager(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger_path = tmp_path / ".vecl" / "provenance.sqlite3"
    with SqliteProvenanceLedger(ledger_path) as ledger:
        event = ledger.append(
            ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant-a", "tester", {"n": 1})
        )

    reopened = SqliteProvenanceLedger(ledger_path)
    assert reopened.require(event.event_id).chain_hash == event.chain_hash


def test_sqlite_ledger_rejects_unscoped_tenant_query_when_strict(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger = SqliteProvenanceLedger(
        tmp_path / ".vecl" / "provenance.sqlite3", strict_tenant_queries=True
    )
    with pytest.raises(PermissionError, match="tenant_id"):
        ledger.query_by_tenant()


def test_artifact_restore_accepts_payload_and_detects_tampering(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ContentAddressedStore(tmp_path / "artifacts")
    record = store.write_text(
        "artifact body",
        producer_specialist_id="producer",
        producer_version="v1",
        input_hash=stable_hash({"input": "x"}),
        output_format="txt",
        parent_event_id="evt-parent",
    )

    assert store.restore(record.to_payload()) == b"artifact body"
    restored_record = ArtifactRecord.from_payload(record.to_payload())
    assert store.read_text(restored_record) == "artifact body"
    (tmp_path / "artifacts" / f"{record.output_hash}.txt").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.restore(record.to_payload())


def test_disk_checkpoint_store_reopens_and_detects_tampering(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = DiskCheckpointStore(tmp_path / ".vecl" / "checkpoints")
    checkpoint = store.create_checkpoint(np.array([1.0, 2.5]), "tenant-a", "checkpoint-1")

    reopened = DiskCheckpointStore(tmp_path / ".vecl" / "checkpoints")
    assert reopened.get_checkpoint("checkpoint-1") == checkpoint
    assert np.allclose(reopened.restore_checkpoint("checkpoint-1"), [1.0, 2.5])
    assert [item.checkpoint_id for item in reopened.list_checkpoints("tenant-a")] == [
        "checkpoint-1"
    ]

    values_path = tmp_path / ".vecl" / "checkpoints" / "checkpoint-1" / "memory_values.npy"
    values_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        reopened.restore_checkpoint("checkpoint-1")


def test_rollback_replays_from_sqlite_ledger_and_disk_checkpoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger = SqliteProvenanceLedger(tmp_path / ".vecl" / "provenance.sqlite3")
    checkpoints = DiskCheckpointStore(tmp_path / ".vecl" / "checkpoints")
    checkpoints.create_checkpoint(np.array([1.0]), "tenant-a", "checkpoint-1")
    ledger.append(
        ProvenanceEvent(
            EventType.LEARNING_EVENT_PREPARED,
            "tenant-a",
            "trainer",
            {"event_id": "learn-a"},
            event_id="learn-a",
        )
    )
    learning = ledger.append(
        ProvenanceEvent(
            EventType.SPARSE_UPDATE_APPLIED,
            "tenant-a",
            "trainer",
            {
                "learning_event_id": "learn-a",
                "source_ids": ["source-a"],
                "selected_slots": [0],
                "eligible_slots": [0],
                "delta_records": [{"slot_id": 0, "delta": -0.25}],
            },
        )
    )
    ledger.close()

    reopened = SqliteProvenanceLedger(tmp_path / ".vecl" / "provenance.sqlite3")
    service = RollbackService(reopened, checkpoints)
    result = service.rollback_by_event(learning.event_id, "tenant-a", "checkpoint-1")

    assert result.exact
    assert np.allclose(result.memory_values, [1.0])
    assert reopened.require(result.rollback_event_id).event_type == EventType.ROLLBACK_PERFORMED
