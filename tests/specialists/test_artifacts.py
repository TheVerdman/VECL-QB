from __future__ import annotations

from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.specialists.artifacts import ContentAddressedStore


def test_content_addressed_store_round_trips_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ContentAddressedStore(tmp_path)

    record = store.write_text(
        "artifact body",
        producer_specialist_id="producer",
        producer_version="v1",
        input_hash=stable_hash({"input": "x"}),
        output_format="txt",
        parent_event_id="evt-parent",
    )

    assert record.artifact_id == f"artifact-{record.output_hash}"
    assert store.read_bytes(record) == b"artifact body"
    assert record.output_path.endswith(f"{record.output_hash}.txt")


def test_artifact_produced_event_chains_in_ledger(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger = ProvenanceLedger()
    parent = ledger.append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant", "tester", {"source_hash": "s"})
    )
    store = ContentAddressedStore(tmp_path)
    record = store.write_text(
        "artifact body",
        producer_specialist_id="producer",
        producer_version="v1",
        input_hash=stable_hash({"input": "x"}),
        output_format="txt",
        parent_event_id=parent.event_id,
    )

    artifact_event = ledger.append(
        ProvenanceEvent(
            EventType.ARTIFACT_PRODUCED,
            "tenant",
            "producer",
            record.to_payload(),
            parent_event_ids=[parent.event_id],
        )
    )

    assert artifact_event.parent_event_ids == [parent.event_id]
    assert artifact_event.payload["artifact_id"] == record.artifact_id
    assert artifact_event.chain_hash != parent.chain_hash


def test_content_addressed_store_detects_tampering(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ContentAddressedStore(tmp_path)
    record = store.write_text(
        "artifact body",
        producer_specialist_id="producer",
        producer_version="v1",
        input_hash=stable_hash({"input": "x"}),
        output_format="txt",
        parent_event_id="evt-parent",
    )
    output_path = tmp_path / f"{record.output_hash}.txt"
    output_path.write_text("tampered")

    try:
        store.read_bytes(record)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered artifact was accepted")
