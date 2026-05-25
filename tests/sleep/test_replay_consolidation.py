from datetime import UTC, datetime

import numpy as np

from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.sleep.consolidation import SleepCycleConsolidator
from vecl.sleep.replay import ReplayBatchBuilder, ReplayCandidate, ReplayPolicy


def _candidate(evidence_id: str, source_id: str, authority: float = 1.0, quarantined: bool = False):
    return ReplayCandidate(
        evidence_id=evidence_id,
        source_id=source_id,
        tenant_id="t",
        activation=1.0,
        rarity=1.0,
        authority=authority,
        claim_ids=["c"],
        last_accessed_at=datetime.now(UTC),
        quarantine_status=quarantined,
    )


def test_single_source_flooding_is_capped() -> None:
    candidates = [_candidate(f"e{i}", "a") for i in range(5)] + [_candidate("e-b", "b")]
    batch = ReplayBatchBuilder().build_batch(candidates, ReplayPolicy(4, 0.5, 0.1, tenant_id="t"))
    assert [candidate.source_id for candidate in batch.candidates].count("a") <= 2


def test_quarantined_evidence_excluded() -> None:
    batch = ReplayBatchBuilder().build_batch(
        [_candidate("e1", "a", quarantined=True)], ReplayPolicy(2, 1.0, 0.1, tenant_id="t")
    )
    assert batch.candidates == []


def test_min_authority_filter() -> None:
    batch = ReplayBatchBuilder().build_batch(
        [_candidate("e1", "a", authority=0.01)], ReplayPolicy(2, 1.0, 0.1, tenant_id="t")
    )
    assert batch.candidates == []


def test_deterministic_batch_construction() -> None:
    candidates = [_candidate("e2", "b"), _candidate("e1", "a")]
    policy = ReplayPolicy(2, 1.0, 0.1, tenant_id="t")
    builder = ReplayBatchBuilder()
    assert builder.build_batch(candidates, policy) == builder.build_batch(
        list(reversed(candidates)), policy
    )


def test_successful_sleep_cycle() -> None:
    consolidator = SleepCycleConsolidator(SparseUpdateMonitor(ProvenanceLedger()))
    report = consolidator.run_cycle(
        [_candidate("e1", "a")],
        ReplayPolicy(1, 1.0, 0.1, tenant_id="t"),
        np.array([1.0]),
        learning_rate=0.1,
        min_score=0.1,
    )
    assert report.committed
    assert report.selected_slots == [0]


def test_adversarial_single_source_flood_gets_capped() -> None:
    batch = ReplayBatchBuilder().build_batch(
        [_candidate(f"adv-{i}", "adv") for i in range(10)] + [_candidate("clean", "clean")],
        ReplayPolicy(4, 0.25, 0.1, tenant_id="t"),
    )
    assert [candidate.source_id for candidate in batch.candidates].count("adv") <= 1


def test_low_authority_batch_produces_no_update() -> None:
    consolidator = SleepCycleConsolidator(SparseUpdateMonitor(ProvenanceLedger()))
    report = consolidator.run_cycle(
        [_candidate("e1", "a", authority=0.01)],
        ReplayPolicy(2, 1.0, 0.1, tenant_id="t"),
        np.array([1.0]),
        learning_rate=0.1,
        min_score=0.1,
    )
    assert report.committed
    assert report.selected_slots == []


def test_failed_monitor_verification_aborts() -> None:
    consolidator = SleepCycleConsolidator(SparseUpdateMonitor(ProvenanceLedger()))
    report = consolidator.run_cycle(
        [_candidate("e1", "a")],
        ReplayPolicy(1, 1.0, 0.1, tenant_id="t"),
        np.array([1.0, 1.0]),
        learning_rate=0.1,
        min_score=0.1,
        force_tamper=True,
    )
    assert not report.committed
    assert report.abort_reason
