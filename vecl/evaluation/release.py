from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from vecl.evaluation.evaluators import ReleaseEvaluator, SummaryEvaluator
from vecl.evaluation.types import MetricThreshold, ReleaseEvaluationReport
from vecl.provenance.events import ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.release_gate import ReleaseGate


def run_release_evaluation(
    candidate_id: str, evaluators: Sequence[ReleaseEvaluator]
) -> ReleaseEvaluationReport:
    return ReleaseEvaluationReport(
        candidate_id=candidate_id,
        results=tuple(evaluator.evaluate() for evaluator in evaluators),
    )


def approve_release_report(
    report: ReleaseEvaluationReport,
    ledger: ProvenanceLedger,
    tenant_id: str,
    actor: str = "release-gate",
) -> ProvenanceEvent:
    gate = ReleaseGate(ledger, tenant_id, actor=actor)
    evaluation = gate.evaluate_candidate_checkpoint(report.candidate_id, **_gate_kwargs(report))
    return gate.approve(evaluation)


def reject_release_report(
    report: ReleaseEvaluationReport,
    ledger: ProvenanceLedger,
    tenant_id: str,
    reason: str,
    actor: str = "release-gate",
) -> ProvenanceEvent:
    gate = ReleaseGate(ledger, tenant_id, actor=actor)
    evaluation = gate.evaluate_candidate_checkpoint(report.candidate_id, **_gate_kwargs(report))
    return gate.reject(evaluation, reason)


def save_release_report(report: ReleaseEvaluationReport, path: Path | str) -> None:
    Path(path).write_text(json.dumps(report.to_payload(), indent=2, sort_keys=True) + "\n")


def load_release_report(path: Path | str) -> ReleaseEvaluationReport:
    return ReleaseEvaluationReport.from_payload(json.loads(Path(path).read_text()))


def _gate_kwargs(report: ReleaseEvaluationReport) -> dict[str, object]:
    kwargs: dict[str, object] = {**report.gate_checks(), "report": report.to_payload()}
    drift_results = [result for result in report.results if result.category in {"fisher", "drift"}]
    if drift_results:
        summary = dict(drift_results[-1].summary)
        kwargs["drift_within_bound"] = bool(drift_results[-1].passed)
        if "drift_value" in summary:
            kwargs["drift_value"] = summary["drift_value"]
        if "drift_threshold" in summary:
            kwargs["drift_threshold"] = summary["drift_threshold"]
        kwargs["drift_report"] = summary
    return kwargs


def default_phase9a_evaluators() -> tuple[ReleaseEvaluator, ...]:
    return (
        SummaryEvaluator(
            name="model_driver_eval",
            category="routing",
            summary={
                "route_success_rate": 1.0,
                "synthesis_success_rate": 1.0,
                "route_parse_failures": 0,
                "synthesis_parse_failures": 0,
            },
            thresholds=(
                MetricThreshold("route_success_rate", min_value=1.0),
                MetricThreshold("synthesis_success_rate", min_value=1.0),
                MetricThreshold("route_parse_failures", equals=0),
                MetricThreshold("synthesis_parse_failures", equals=0),
            ),
        ),
        SummaryEvaluator(
            name="tool_call_payload_eval",
            category="tool_call",
            summary={
                "proposal_success_rate": 1.0,
                "validation_success_rate": 1.0,
                "execution_success_rate": 1.0,
                "synthesis_success_rate": 1.0,
            },
            thresholds=(
                MetricThreshold("proposal_success_rate", min_value=1.0),
                MetricThreshold("validation_success_rate", min_value=1.0),
                MetricThreshold("execution_success_rate", min_value=1.0),
                MetricThreshold("synthesis_success_rate", min_value=1.0),
            ),
        ),
        SummaryEvaluator(
            name="ethics_eval",
            category="ethics",
            summary={
                "parse_failures": 0,
                "unexpected_outcomes": [],
                "blocked_specialist_calls": 0,
            },
            thresholds=(
                MetricThreshold("parse_failures", equals=0),
                MetricThreshold("unexpected_outcomes", equals=[]),
                MetricThreshold("blocked_specialist_calls", equals=0),
            ),
        ),
        SummaryEvaluator(
            name="artifact_persistence_eval",
            category="invariant",
            summary={
                "persisted_artifact_restore_count": 2,
                "artifact_events": 2,
                "checkpoint_tamper_detected": True,
            },
            thresholds=(
                MetricThreshold("checkpoint_tamper_detected", equals=True),
                MetricThreshold("persisted_artifact_restore_count", min_value=1),
            ),
        ),
        SummaryEvaluator(
            name="rollback_drill",
            category="rollback",
            summary={"rollback_exact": True, "memory_hash_reproduced": True},
            thresholds=(
                MetricThreshold("rollback_exact", equals=True),
                MetricThreshold("memory_hash_reproduced", equals=True),
            ),
        ),
        SummaryEvaluator(
            name="tenant_isolation_eval",
            category="tenant_isolation",
            summary={"tenant_crossing_attempt_count": 0},
            thresholds=(MetricThreshold("tenant_crossing_attempt_count", equals=0),),
        ),
        SummaryEvaluator(
            name="verification_calibration_eval",
            category="calibration",
            summary={"verification_calibration_not_worse": True},
            thresholds=(MetricThreshold("verification_calibration_not_worse", equals=True),),
        ),
    )
