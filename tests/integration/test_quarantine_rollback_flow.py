import numpy as np

from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.checkpoints import CheckpointStore, RollbackService


def _append_update(
    ledger: ProvenanceLedger, learning_id: str, source_id: str, delta: float
) -> None:
    ledger.append(
        ProvenanceEvent(
            EventType.LEARNING_EVENT_PREPARED,
            "tenant",
            "test",
            {"event_id": learning_id},
            event_id=learning_id,
        )
    )
    ledger.append(
        ProvenanceEvent(
            EventType.SPARSE_UPDATE_APPLIED,
            "tenant",
            "test",
            {
                "learning_event_id": learning_id,
                "source_ids": [source_id],
                "eligible_slots": [0],
                "selected_slots": [0],
                "delta_records": [{"slot_id": 0, "delta": delta}],
            },
        )
    )


def test_adversarial_source_quarantine_and_exact_rollback() -> None:
    ledger = ProvenanceLedger()
    store = CheckpointStore()
    store.create_checkpoint(np.array([1.0]), "tenant", "c0")
    _append_update(ledger, "learn-a", "A", -0.2)
    _append_update(ledger, "learn-b", "B", -0.1)
    ledger.append(
        ProvenanceEvent(
            EventType.SOURCE_QUARANTINED,
            "tenant",
            "trust",
            {"source_id": "A", "reason": "adversarial"},
        )
    )
    result = RollbackService(ledger, store).rollback_by_source("A", "tenant", "c0")
    assert result.exact
    assert np.allclose(result.memory_values, [0.9])
    event_types = [event.event_type for event in ledger.events()]
    assert EventType.SOURCE_QUARANTINED in event_types
    assert EventType.ROLLBACK_PERFORMED in event_types
