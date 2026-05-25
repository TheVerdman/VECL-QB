from datetime import UTC, datetime

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.claim_graph import ClaimGraph
from vecl.qb.specialist import SpecialistClaim
from vecl.qb.synthesis import synthesize_claim_graph
from vecl.qb.verifier import VerificationPolicy, VerificationStatus, Verifier
from vecl.runtime.checkpoints import CheckpointStore, RollbackService
from vecl.sleep.replay import ReplayBatchBuilder, ReplayCandidate, ReplayPolicy
from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs
from vecl.trust.updater import BoundedTrustUpdater, LearnedTrustState, TrustUpdateObservation


@given(st.integers(min_value=1, max_value=8), st.integers(min_value=0, max_value=8))
@settings(max_examples=30)
def test_sparse_memory_invariants(slot_count: int, max_slots: int) -> None:
    # BoundedUpdateSize, SelectedSubsetEligible, and NoIneligibleUpdate mirror the TLA update invariants.
    inputs = SparseMemoryInputs(
        memory_values=np.ones(slot_count),
        gradients=np.ones(slot_count),
        activation=np.ones(slot_count),
        rarity=np.ones(slot_count),
        authority=np.linspace(0.0, 1.0, slot_count),
        learning_rate=0.1,
        min_authority=0.2,
        min_score=0.2,
        max_slots=max_slots,
        quarantined_slots=set(),
    )
    result = sparse_update_oracle(inputs, "evt", "root")
    changed = set(np.where(~np.isclose(inputs.memory_values, result.new_memory_values))[0])
    assert len(result.selected_slots) <= max_slots
    assert set(result.selected_slots).issubset(result.eligible_slots)
    assert changed.issubset(result.selected_slots)
    assert all(record.event_id == "evt" for record in result.delta_records)


def test_zero_authority_zero_score_and_norm_bound() -> None:
    # ZeroAuthorityNoUpdate, ZeroScoreNoUpdate, and AuthorityScaledNormBound prevent rank-only trust bypass.
    inputs = SparseMemoryInputs(
        memory_values=np.ones(2),
        gradients=np.ones(2),
        activation=np.array([0.0, 1.0]),
        rarity=np.ones(2),
        authority=np.array([1.0, 0.0]),
        learning_rate=0.1,
        min_authority=0.1,
        min_score=0.1,
        max_slots=2,
        quarantined_slots=set(),
    )
    result = sparse_update_oracle(inputs, "evt", "root")
    assert result.selected_slots == []
    assert np.linalg.norm(result.new_memory_values - inputs.memory_values) <= 0.0


def test_provenance_invariants() -> None:
    # NoCommitWithoutPreparedEvent, LedgerAppendOnly, RollbackEmitsEvent are concrete provenance obligations.
    ledger = ProvenanceLedger()
    before = len(ledger.events())
    try:
        ledger.append(
            ProvenanceEvent(
                EventType.SPARSE_UPDATE_APPLIED,
                "t",
                "a",
                {"learning_event_id": "missing"},
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("commit without prepare accepted")
    assert len(ledger.events()) == before
    store = CheckpointStore()
    store.create_checkpoint(np.array([1.0]), "t", "c0")
    result = RollbackService(ledger, store).rollback_to_checkpoint("c0")
    assert result.exact
    assert ledger.events()[-1].event_type == EventType.ROLLBACK_PERFORMED


def test_trust_invariants() -> None:
    # NoSelfPromotion, RootTrustRequired, and BoundedTrustDrift keep learned trust anchored.
    updater = BoundedTrustUpdater(max_drift_per_update=0.01, max_total_drift=0.2)
    state = LearnedTrustState("s", 0.0, datetime.now(UTC), 0)
    promoted = updater.update(
        0.0, state, [TrustUpdateObservation("s", 1.0, "PASSED", 0.0, self_reported=True)]
    )
    assert updater.effective_trust(0.0, promoted.learned_trust) == 0.0
    bounded = updater.update(
        0.5,
        LearnedTrustState("s", 0.5, datetime.now(UTC), 0),
        [TrustUpdateObservation("s", 1.0, "PASSED", 0.0, independent_confirmation_count=1)],
    )
    assert bounded.learned_trust <= 0.51


def test_qb_invariants() -> None:
    # NoAnswerWithoutClaimGraph, NoConclusionWithoutSupport, and ConflictRequiresReview are QB safety rules.
    graph = ClaimGraph()
    graph.add_claim(
        SpecialistClaim(
            "c1", "s", "Accept.", "decision", 0.9, evidence_ids=["e1"], source_ids=["s1"]
        )
    )
    graph.add_claim(
        SpecialistClaim(
            "c2", "s", "Reject.", "decision", 0.9, evidence_ids=["e2"], source_ids=["s2"]
        )
    )
    graph.add_edge("c1", "c2", "contradicts")
    verification = Verifier().verify_claim_graph(graph, VerificationPolicy())
    synthesis = synthesize_claim_graph(graph, verification.status.value)
    assert verification.status == VerificationStatus.NEEDS_REVIEW
    assert synthesis.provenance_graph_hash
    assert synthesis.supporting_claim_ids == []


def test_tenant_invariants() -> None:
    # NoCrossTenantReplay and NoCrossTenantRollback require tenant identity on replay and checkpoint state.
    now = datetime.now(UTC)
    candidates = [
        ReplayCandidate("e1", "s1", "t1", 1.0, 1.0, 1.0, [], now),
        ReplayCandidate("e2", "s2", "t2", 1.0, 1.0, 1.0, [], now),
    ]
    try:
        ReplayBatchBuilder().build_batch(candidates, ReplayPolicy(2, 1.0, 0.0))
    except ValueError:
        pass
    else:
        raise AssertionError("cross-tenant replay accepted")
    store = CheckpointStore()
    store.create_checkpoint(np.array([1.0]), "t1", "c1")
    try:
        RollbackService(ProvenanceLedger(), store).rollback_by_event("e", "t2", "c1")
    except PermissionError:
        pass
    else:
        raise AssertionError("cross-tenant rollback accepted")
