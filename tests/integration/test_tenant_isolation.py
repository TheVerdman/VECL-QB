from datetime import UTC, datetime

import numpy as np
import pytest

from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.checkpoints import CheckpointStore, RollbackService
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.runtime.tokens import create_learning_event_token, validate_learning_event_token
from vecl.sleep.consolidation import SleepCycleConsolidator
from vecl.sleep.replay import ReplayCandidate, ReplayPolicy


def _candidate(evidence_id: str, tenant_id: str):
    return ReplayCandidate(
        evidence_id=evidence_id,
        source_id=f"source-{tenant_id}",
        tenant_id=tenant_id,
        activation=1.0,
        rarity=1.0,
        authority=1.0,
        claim_ids=["c"],
        last_accessed_at=datetime.now(UTC),
    )


def test_cross_tenant_replay_fails() -> None:
    consolidator = SleepCycleConsolidator(SparseUpdateMonitor(ProvenanceLedger()))
    with pytest.raises(ValueError, match="mixed"):
        consolidator.builder.build_batch(
            [_candidate("a", "t1"), _candidate("b", "t2")],
            ReplayPolicy(2, 1.0, 0.1),
        )


def test_cross_tenant_token_reuse_fails() -> None:
    token = create_learning_event_token(
        batch_id="b",
        tenant_id="t1",
        source_set_hash="s",
        provenance_root_hash="r",
        min_authority=0.1,
        min_score=0.1,
        max_slots=1,
        policy_version="p",
        trust_policy_version="tp",
    )
    with pytest.raises(ValueError, match="tenant"):
        validate_learning_event_token(token, datetime.now(UTC), "t2")


def test_cross_tenant_ledger_query_without_permission_fails() -> None:
    ledger = ProvenanceLedger(strict_tenant_queries=True)
    with pytest.raises(PermissionError):
        ledger.query_by_tenant()


def test_two_tenants_run_separate_sleep_cycles_without_contamination() -> None:
    ledger = ProvenanceLedger()
    monitor = SparseUpdateMonitor(ledger)
    consolidator = SleepCycleConsolidator(monitor)
    report_a = consolidator.run_cycle(
        [_candidate("a", "t1")],
        ReplayPolicy(1, 1.0, 0.1, tenant_id="t1"),
        np.array([1.0]),
        0.1,
        0.1,
    )
    memory_a = monitor.committed_memory_values.copy()
    report_b = consolidator.run_cycle(
        [_candidate("b", "t2")],
        ReplayPolicy(1, 1.0, 0.1, tenant_id="t2"),
        np.array([2.0]),
        0.1,
        0.1,
    )
    assert report_a.committed and report_b.committed
    assert np.allclose(memory_a, [0.9])
    assert np.allclose(monitor.committed_memory_values, [1.9])


def test_no_cross_tenant_rollback() -> None:
    ledger = ProvenanceLedger()
    store = CheckpointStore()
    store.create_checkpoint(np.array([1.0]), "t1", "c1")
    service = RollbackService(ledger, store)
    with pytest.raises(PermissionError):
        service.rollback_to_checkpoint("c1") if False else service.rollback_by_event(
            "e", "t2", "c1"
        )
