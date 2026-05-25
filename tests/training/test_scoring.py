from __future__ import annotations

import numpy as np
import pytest

from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.runtime.tokens import create_learning_event_token
from vecl.sparse.types import SparseMemoryInputs
from vecl.training.scoring import (
    activation_from_gradient_summaries,
    rarity_from_selection_counts,
    slot_selection_counts_from_ledger,
)


def test_activation_from_gradient_summaries_normalizes_and_handles_zero() -> None:
    assert np.allclose(activation_from_gradient_summaries([0.0, 0.0]), [0.0, 0.0])
    assert np.allclose(activation_from_gradient_summaries([0.0, 2.0, 4.0]), [0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="finite"):
        activation_from_gradient_summaries([1.0, float("nan")])


def test_rarity_from_selection_counts_decreases_with_count() -> None:
    rarity = rarity_from_selection_counts([0, 3, 8])
    assert np.allclose(rarity, [1.0, 0.5, 1.0 / 3.0])
    assert rarity[0] > rarity[1] > rarity[2]
    with pytest.raises(ValueError, match="non-negative"):
        rarity_from_selection_counts([0, -1])


def test_slot_selection_counts_from_ledger_reads_sparse_update_events() -> None:
    ledger = ProvenanceLedger()
    monitor = SparseUpdateMonitor(ledger)
    token = create_learning_event_token(
        batch_id="batch",
        tenant_id="tenant-a",
        source_set_hash="sources",
        provenance_root_hash="root",
        min_authority=0.0,
        min_score=0.0,
        max_slots=2,
        policy_version="p",
        trust_policy_version="tp",
    )
    inputs = SparseMemoryInputs(
        memory_values=np.ones(3),
        gradients=np.ones(3),
        activation=np.array([0.2, 0.9, 0.8]),
        rarity=np.ones(3),
        authority=np.ones(3),
        learning_rate=0.1,
        min_authority=0.0,
        min_score=0.0,
        max_slots=2,
        quarantined_slots=set(),
    )
    result = monitor.apply_sparse_update(token, inputs)
    monitor.commit_learning_event(token)

    counts = slot_selection_counts_from_ledger(ledger, slot_count=3, tenant_id="tenant-a")
    other_tenant_counts = slot_selection_counts_from_ledger(
        ledger, slot_count=3, tenant_id="tenant-b"
    )

    assert result.selected_slots == [1, 2]
    assert np.allclose(counts, [0.0, 1.0, 1.0])
    assert np.allclose(other_tenant_counts, [0.0, 0.0, 0.0])
    assert ledger.find_by_type(EventType.SPARSE_UPDATE_APPLIED)
