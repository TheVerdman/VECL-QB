from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vecl._compat import UTC


@dataclass(frozen=True)
class LearnedTrustState:
    source_id: str
    learned_trust: float
    last_updated_at: datetime
    evidence_count: int = 0


@dataclass(frozen=True)
class TrustUpdateObservation:
    source_id: str
    outcome_score: float
    verification_status: str
    contradiction_score: float
    independent_confirmation_count: int = 0
    self_reported: bool = False
    governance_approval: bool = False


@dataclass(frozen=True)
class BoundedTrustUpdater:
    max_drift_per_update: float = 0.05
    max_total_drift: float = 0.2

    def update(
        self,
        root_trust: float,
        learned_state: LearnedTrustState,
        observations: list[TrustUpdateObservation],
        now: datetime | None = None,
    ) -> LearnedTrustState:
        now = now or datetime.now(UTC)
        learned = learned_state.learned_trust
        for observation in observations:
            if observation.self_reported and observation.independent_confirmation_count == 0:
                signal = 0.0
            else:
                signal = observation.outcome_score - observation.contradiction_score
                if observation.independent_confirmation_count > 0:
                    signal += 0.05 * observation.independent_confirmation_count
            if observation.verification_status == "FAILED":
                signal -= 0.2
            step = max(-self.max_drift_per_update, min(self.max_drift_per_update, signal * 0.1))
            learned += step
        cap = root_trust + self.max_total_drift
        has_independent = any(obs.independent_confirmation_count > 0 for obs in observations)
        if not has_independent:
            cap = root_trust
        learned = max(0.0, min(1.0, min(learned, cap)))
        return LearnedTrustState(
            source_id=learned_state.source_id,
            learned_trust=learned,
            last_updated_at=now,
            evidence_count=learned_state.evidence_count + len(observations),
        )

    def effective_trust(
        self, root_trust: float, learned_trust: float, governance_approval: bool = False
    ) -> float:
        if root_trust == 0 and not governance_approval:
            return 0.0
        return max(0.0, min(1.0, min(learned_trust, root_trust + self.max_total_drift)))
