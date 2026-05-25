from datetime import UTC, datetime

import numpy as np

from vecl.runtime.metrics import adversarial_metrics
from vecl.sleep.replay import ReplayBatchBuilder, ReplayCandidate, ReplayPolicy
from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs
from vecl.trust.updater import BoundedTrustUpdater, LearnedTrustState, TrustUpdateObservation


def _candidate(evidence_id: str, source_id: str, authority: float) -> ReplayCandidate:
    return ReplayCandidate(
        evidence_id=evidence_id,
        source_id=source_id,
        tenant_id="t",
        activation=10.0,
        rarity=10.0,
        authority=authority,
        claim_ids=["c"],
        last_accessed_at=datetime.now(UTC),
    )


def test_adversarial_long_game_bounded_drift(capsys) -> None:
    clean = [_candidate("clean", "clean", 1.0)]
    adversarial = [_candidate(f"adv-{i}", "adv", 0.01) for i in range(10)]
    builder = ReplayBatchBuilder()

    no_protection = builder.build_batch(
        adversarial + clean, ReplayPolicy(10, 1.0, 0.0, tenant_id="t")
    )
    assert [candidate.source_id for candidate in no_protection.candidates].count("adv") > 5

    thresholds_only = builder.build_batch(
        adversarial + clean, ReplayPolicy(10, 1.0, 0.01, tenant_id="t")
    )
    assert [candidate.source_id for candidate in thresholds_only.candidates].count("adv") > 5

    protected = builder.build_batch(adversarial + clean, ReplayPolicy(4, 0.25, 0.1, tenant_id="t"))
    replay_sources = [candidate.source_id for candidate in protected.candidates]

    memory = np.ones(4)
    authority = np.array([candidate.authority for candidate in protected.candidates] + [0.0] * 4)[
        :4
    ]
    inputs = SparseMemoryInputs(
        memory_values=memory,
        gradients=np.ones(4),
        activation=np.array(
            [candidate.activation for candidate in protected.candidates] + [0.0] * 4
        )[:4],
        rarity=np.array([candidate.rarity for candidate in protected.candidates] + [0.0] * 4)[:4],
        authority=authority,
        learning_rate=0.1,
        min_authority=0.1,
        min_score=0.1,
        max_slots=4,
        quarantined_slots=set(),
    )
    result = sparse_update_oracle(inputs, "evt", "root")
    selected_sources = [replay_sources[slot] for slot in result.selected_slots]

    updater = BoundedTrustUpdater()
    before = 0.0
    state = LearnedTrustState("adv", before, datetime.now(UTC), 0)
    after_self = updater.update(
        0.0,
        state,
        [TrustUpdateObservation("adv", 1.0, "PASSED", 0.0, self_reported=True) for _ in range(20)],
    )
    effective_after_self = updater.effective_trust(0.0, after_self.learned_trust)
    after_independent = updater.update(
        0.5,
        LearnedTrustState("adv", 0.5, datetime.now(UTC), 0),
        [TrustUpdateObservation("adv", 1.0, "PASSED", 0.0, independent_confirmation_count=2)],
    )
    assert after_independent.learned_trust > 0.5

    metrics = adversarial_metrics(
        "adv",
        replay_sources,
        selected_sources,
        memory,
        result.new_memory_values,
        before,
        effective_after_self,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    captured = capsys.readouterr().out
    assert "adversarial_source_replay_fraction" in captured
    assert metrics["adversarial_source_replay_fraction"] == 0.0
    assert metrics["adversarial_source_selected_fraction"] == 0.0
    assert metrics["memory_drift_norm"] <= 0.1
    assert metrics["trust_drift_by_source"] == 0.0
