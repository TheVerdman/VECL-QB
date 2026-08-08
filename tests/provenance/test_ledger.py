from dataclasses import replace

import pytest

from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import ProvenanceLedger, verify_event_chain


def test_event_order_is_preserved() -> None:
    ledger = ProvenanceLedger()
    first = ledger.append(ProvenanceEvent(EventType.EVIDENCE_INGESTED, "t", "a", {"n": 1}))
    second = ledger.append(ProvenanceEvent(EventType.EVIDENCE_INGESTED, "t", "a", {"n": 2}))
    assert [event.event_id for event in ledger.events()] == [first.event_id, second.event_id]


def test_chain_hash_changes_if_payload_changes() -> None:
    ledger_a = ProvenanceLedger()
    ledger_b = ProvenanceLedger()
    event_a = ledger_a.append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "t", "a", {"payload": "one"}, event_id="same")
    )
    event_b = ledger_b.append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "t", "a", {"payload": "two"}, event_id="same")
    )
    assert event_a.chain_hash != event_b.chain_hash


def test_committed_event_payload_cannot_drift_from_payload_hash() -> None:
    source_payload = {"nested": {"value": 1}, "items": ["original"]}
    event = ProvenanceLedger().append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant", "actor", source_payload)
    )

    source_payload["nested"]["value"] = 2
    source_payload["items"].append("mutated")

    assert event.payload == {"nested": {"value": 1}, "items": ["original"]}
    assert event.payload_hash == stable_hash(event.payload)
    with pytest.raises(TypeError, match="does not support item assignment"):
        event.payload["nested"]["value"] = 3


def test_chain_integrity_verifier_rejects_tampered_chain_hash() -> None:
    ledger = ProvenanceLedger()
    first = ledger.append(ProvenanceEvent(EventType.EVIDENCE_INGESTED, "t", "a", {"n": 1}))
    second = ledger.append(ProvenanceEvent(EventType.EVIDENCE_INGESTED, "t", "a", {"n": 2}))

    assert ledger.verify_integrity()
    with pytest.raises(ValueError, match="chain hash"):
        verify_event_chain((first, replace(second, chain_hash="tampered")))


def test_append_rejects_missing_or_cross_tenant_parent() -> None:
    ledger = ProvenanceLedger()
    parent = ledger.append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant-a", "actor", {"n": 1})
    )

    with pytest.raises(ValueError, match="parent event does not exist"):
        ledger.append(
            ProvenanceEvent(
                EventType.EVIDENCE_INGESTED,
                "tenant-a",
                "actor",
                {"n": 2},
                parent_event_ids=["missing"],
            )
        )
    with pytest.raises(ValueError, match="parent event tenant"):
        ledger.append(
            ProvenanceEvent(
                EventType.EVIDENCE_INGESTED,
                "tenant-b",
                "actor",
                {"n": 2},
                parent_event_ids=[parent.event_id],
            )
        )


def test_chain_integrity_verifier_rejects_parent_that_is_not_earlier_in_chain() -> None:
    parent = ProvenanceEvent(
        EventType.EVIDENCE_INGESTED,
        "tenant",
        "actor",
        {"n": 1},
        event_id="parent",
    ).with_chain_hash("GENESIS")
    child = ProvenanceEvent(
        EventType.EVIDENCE_INGESTED,
        "tenant",
        "actor",
        {"n": 2},
        event_id="child",
        parent_event_ids=[parent.event_id],
    ).with_chain_hash(parent.chain_hash)

    with pytest.raises(ValueError, match="parent event is missing or not earlier"):
        verify_event_chain((child, parent))


def test_event_rejects_malformed_or_duplicate_parent_ids() -> None:
    with pytest.raises(ValueError, match="parent_event_ids"):
        ProvenanceEvent(
            EventType.EVIDENCE_INGESTED,
            "tenant",
            "actor",
            {},
            parent_event_ids="not-a-sequence-of-ids",
        )
    with pytest.raises(ValueError, match="unique"):
        ProvenanceEvent(
            EventType.EVIDENCE_INGESTED,
            "tenant",
            "actor",
            {},
            parent_event_ids=["same", "same"],
        )


def test_event_rejects_supplied_payload_hash_that_does_not_match_payload() -> None:
    with pytest.raises(ValueError, match="payload hash"):
        ProvenanceEvent(
            EventType.EVIDENCE_INGESTED,
            "tenant",
            "actor",
            {"value": "actual"},
            payload_hash=stable_hash({"value": "different"}),
        )


def test_event_rejects_ambiguous_or_nonfinite_payload_values() -> None:
    with pytest.raises(ValueError, match="string keys"):
        ProvenanceEvent(
            event_type=EventType.EVIDENCE_INGESTED,
            tenant_id="tenant-a",
            actor="test",
            payload={1: "numeric", "1": "string"},
        )
    with pytest.raises(ValueError, match="finite"):
        ProvenanceEvent(
            event_type=EventType.EVIDENCE_INGESTED,
            tenant_id="tenant-a",
            actor="test",
            payload={"score": float("nan")},
        )


def test_chain_integrity_revalidates_learning_event_reference_semantics() -> None:
    wrong_type = ProvenanceEvent(
        event_type=EventType.EVIDENCE_INGESTED,
        event_id="not-a-learning-event",
        tenant_id="tenant-a",
        actor="test",
        payload={"source_hash": "source"},
    ).with_chain_hash("GENESIS")
    applied = ProvenanceEvent(
        event_type=EventType.SPARSE_UPDATE_APPLIED,
        tenant_id="tenant-a",
        actor="test",
        payload={"learning_event_id": wrong_type.event_id},
    ).with_chain_hash(wrong_type.chain_hash)

    with pytest.raises(ValueError, match="LearningEventPrepared"):
        verify_event_chain((wrong_type, applied))


def test_missing_tenant_rejected() -> None:
    try:
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "", "a", {})
    except ValueError as exc:
        assert "tenant_id" in str(exc)
    else:
        raise AssertionError("missing tenant_id was accepted")


def test_cannot_append_sparse_update_without_prepared() -> None:
    ledger = ProvenanceLedger()
    event = ProvenanceEvent(
        EventType.SPARSE_UPDATE_APPLIED,
        "t",
        "a",
        {"learning_event_id": "missing", "delta_records": []},
    )
    try:
        ledger.append(event)
    except ValueError as exc:
        assert "prepared" in str(exc)
    else:
        raise AssertionError("unprepared sparse update committed")


def test_sparse_update_rejects_reference_to_wrong_event_type() -> None:
    ledger = ProvenanceLedger()
    ledger.append(
        ProvenanceEvent(
            EventType.EVIDENCE_INGESTED,
            "tenant",
            "actor",
            {},
            event_id="not-a-learning-event",
        )
    )

    with pytest.raises(ValueError, match="LearningEventPrepared"):
        ledger.append(
            ProvenanceEvent(
                EventType.SPARSE_UPDATE_APPLIED,
                "tenant",
                "actor",
                {"learning_event_id": "not-a-learning-event"},
            )
        )


def test_sparse_update_rejects_cross_tenant_prepared_event_reference() -> None:
    ledger = ProvenanceLedger()
    ledger.append(
        ProvenanceEvent(
            EventType.LEARNING_EVENT_PREPARED,
            "tenant-a",
            "actor",
            {"event_id": "learning-event"},
            event_id="learning-event",
        )
    )

    with pytest.raises(ValueError, match="tenant"):
        ledger.append(
            ProvenanceEvent(
                EventType.SPARSE_UPDATE_APPLIED,
                "tenant-b",
                "actor",
                {"learning_event_id": "learning-event"},
            )
        )


def test_rollback_points_to_source_or_event() -> None:
    ledger = ProvenanceLedger()
    event = ledger.append(
        ProvenanceEvent(EventType.ROLLBACK_PERFORMED, "t", "a", {"source_id": "source-a"})
    )
    assert event.payload["source_id"] == "source-a"
