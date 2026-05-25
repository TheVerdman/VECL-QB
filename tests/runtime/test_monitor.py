from datetime import UTC, datetime

import numpy as np
import pytest

from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.runtime.tokens import create_learning_event_token
from vecl.sparse.types import SparseMemoryInputs


def _token(min_authority: float = 0.1, min_score: float = 0.1, max_slots: int = 1):
    return create_learning_event_token(
        batch_id="b",
        tenant_id="t",
        source_set_hash="sources",
        provenance_root_hash="root",
        min_authority=min_authority,
        min_score=min_score,
        max_slots=max_slots,
        policy_version="p",
        trust_policy_version="tp",
        now=datetime.now(UTC),
    )


def _inputs(min_authority: float = 0.1, min_score: float = 0.1, max_slots: int = 1):
    return SparseMemoryInputs(
        memory_values=np.array([1.0, 1.0]),
        gradients=np.array([1.0, 1.0]),
        activation=np.array([1.0, 1.0]),
        rarity=np.array([1.0, 1.0]),
        authority=np.array([1.0, 0.0]),
        learning_rate=0.1,
        min_authority=min_authority,
        min_score=min_score,
        max_slots=max_slots,
        quarantined_slots=set(),
    )


def test_successful_commit() -> None:
    ledger = ProvenanceLedger()
    monitor = SparseUpdateMonitor(ledger)
    token = _token()
    result = monitor.apply_sparse_update(token, _inputs())
    assert monitor.committed_memory_values is None
    events = monitor.commit_learning_event(token)
    assert result.selected_slots == [0]
    assert monitor.committed_memory_values is not None
    assert [event.event_type for event in events] == [
        EventType.SPARSE_UPDATE_APPLIED,
        EventType.LEARNING_EVENT_COMMITTED,
    ]


def test_abort_on_ineligible_update() -> None:
    ledger = ProvenanceLedger()
    monitor = SparseUpdateMonitor(ledger)
    token = _token()
    inputs = _inputs()
    result = monitor.apply_sparse_update(token, inputs)
    result.new_memory_values[1] = 0.0
    with pytest.raises(ValueError, match="changed slots"):
        monitor.verify_sparse_update(token, inputs, result)
    aborted = monitor.abort_learning_event(token, "tampered")
    assert aborted.event_type == EventType.LEARNING_EVENT_ABORTED
    assert monitor.committed_memory_values is None


def test_abort_on_tampered_selected_slots() -> None:
    monitor = SparseUpdateMonitor(ProvenanceLedger())
    token = _token()
    inputs = _inputs()
    result = monitor.apply_sparse_update(token, inputs)
    result.selected_slots.append(1)
    with pytest.raises(ValueError, match="subset"):
        monitor.verify_sparse_update(token, inputs, result)


def test_abort_on_token_threshold_mismatch() -> None:
    monitor = SparseUpdateMonitor(ProvenanceLedger())
    token = _token(min_authority=0.5)
    with pytest.raises(ValueError, match="thresholds"):
        monitor.apply_sparse_update(token, _inputs(min_authority=0.1))
