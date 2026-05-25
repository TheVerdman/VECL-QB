from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger


@dataclass(frozen=True)
class ReleaseGateEvaluation:
    checkpoint_id: str
    invariant_tests_pass: bool
    regression_tests_pass: bool
    adversarial_simulation_within_bound: bool
    rollback_test_pass: bool
    verification_calibration_not_worse: bool
    no_tenant_isolation_failure: bool
    drift_within_bound: bool = True
    drift_value: float | None = None
    drift_threshold: float | None = None
    drift_report: dict[str, Any] | None = None
    report: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return all(
            [
                self.invariant_tests_pass,
                self.regression_tests_pass,
                self.adversarial_simulation_within_bound,
                self.rollback_test_pass,
                self.verification_calibration_not_worse,
                self.no_tenant_isolation_failure,
                self.drift_within_bound,
            ]
        )


class ReleaseGate:
    def __init__(
        self, ledger: ProvenanceLedger, tenant_id: str, actor: str = "release-gate"
    ) -> None:
        self.ledger = ledger
        self.tenant_id = tenant_id
        self.actor = actor

    def evaluate_candidate_checkpoint(
        self, checkpoint_id: str, **checks: Any
    ) -> ReleaseGateEvaluation:
        report = checks.get("report")
        return ReleaseGateEvaluation(
            checkpoint_id=checkpoint_id,
            invariant_tests_pass=checks.get("invariant_tests_pass", False),
            regression_tests_pass=checks.get("regression_tests_pass", False),
            adversarial_simulation_within_bound=checks.get(
                "adversarial_simulation_within_bound", False
            ),
            rollback_test_pass=checks.get("rollback_test_pass", False),
            verification_calibration_not_worse=checks.get(
                "verification_calibration_not_worse", False
            ),
            no_tenant_isolation_failure=checks.get("no_tenant_isolation_failure", False),
            drift_within_bound=checks.get("drift_within_bound", True),
            drift_value=_optional_float(checks.get("drift_value")),
            drift_threshold=_optional_float(checks.get("drift_threshold")),
            drift_report=(
                checks.get("drift_report") if isinstance(checks.get("drift_report"), dict) else None
            ),
            report=report if isinstance(report, dict) else None,
        )

    def compare_against_baseline(
        self, candidate_score: float, baseline_score: float, tolerance: float = 0.0
    ) -> bool:
        return candidate_score + tolerance >= baseline_score

    def approve(self, evaluation: ReleaseGateEvaluation) -> ProvenanceEvent:
        if not evaluation.passed:
            raise ValueError("candidate checkpoint cannot be promoted without passing gate")
        return self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.RELEASE_APPROVED,
                tenant_id=self.tenant_id,
                actor=self.actor,
                payload={
                    "checkpoint_id": evaluation.checkpoint_id,
                    "evaluation": evaluation.__dict__,
                },
            )
        )

    def reject(self, evaluation: ReleaseGateEvaluation, reason: str) -> ProvenanceEvent:
        return self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.RELEASE_REJECTED,
                tenant_id=self.tenant_id,
                actor=self.actor,
                payload={
                    "checkpoint_id": evaluation.checkpoint_id,
                    "reason": reason,
                    "evaluation": evaluation.__dict__,
                },
            )
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
