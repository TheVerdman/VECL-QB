from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger


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


def test_rollback_points_to_source_or_event() -> None:
    ledger = ProvenanceLedger()
    event = ledger.append(
        ProvenanceEvent(EventType.ROLLBACK_PERFORMED, "t", "a", {"source_id": "source-a"})
    )
    assert event.payload["source_id"] == "source-a"
