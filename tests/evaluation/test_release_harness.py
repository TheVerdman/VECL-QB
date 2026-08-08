from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from vecl.cli import main
from vecl.evaluation.evaluators import (
    DemoSummaryEvaluator,
    FisherDriftEvaluator,
    ScriptEvaluator,
    evaluator_from_config,
    evaluators_from_manifest,
)
from vecl.evaluation.release import (
    approve_release_report,
    demo_phase9a_evaluators,
    load_release_report,
    reject_release_report,
    run_release_evaluation,
    save_release_report,
)
from vecl.evaluation.types import (
    EvaluationEvidence,
    EvaluationResult,
    MetricThreshold,
    ReleaseEvaluationReport,
)
from vecl.provenance.events import EventType, stable_hash
from vecl.provenance.ledger import ProvenanceLedger, SqliteProvenanceLedger
from vecl.runtime.release_gate import ReleaseGate


def _evidence_result(
    name: str,
    category: str,
    summary: dict[str, object] | None = None,
    *,
    candidate_id: str = "candidate-a",
):
    summary = summary or {"ok": True}
    thresholds = (MetricThreshold("ok", equals=True),) if "ok" in summary else ()
    command = (sys.executable, "-m", "pytest", f"tests/{name}.py")
    configuration = {
        "category": category,
        "command": list(command),
        "thresholds": [threshold.to_payload() for threshold in thresholds],
    }
    return EvaluationResult(
        name=name,
        category=category,
        passed=True,
        summary=summary,
        thresholds=thresholds,
        evidence=EvaluationEvidence(
            kind="command",
            evaluator_id=name,
            evaluator_version="test-v1",
            candidate_id=candidate_id,
            configuration=configuration,
            configuration_hash=stable_hash(configuration),
            git_revision="a" * 40,
            command=command,
            artifact_hashes={
                "summary": stable_hash(summary),
                "stdout": stable_hash("ok"),
                "stderr": stable_hash(""),
            },
            working_tree_clean=True,
        ),
    )


def _approvable_report(candidate_id: str = "candidate-a") -> ReleaseEvaluationReport:
    return ReleaseEvaluationReport(
        candidate_id=candidate_id,
        manifest_hash=stable_hash({"name": "test-release-manifest", "version": 1}),
        results=(
            _evidence_result("invariants", "invariant", candidate_id=candidate_id),
            _evidence_result("regressions", "regression", candidate_id=candidate_id),
            _evidence_result("ethics", "ethics", candidate_id=candidate_id),
            _evidence_result("rollback", "rollback", candidate_id=candidate_id),
            _evidence_result("calibration", "calibration", candidate_id=candidate_id),
            _evidence_result("tenant-isolation", "tenant_isolation", candidate_id=candidate_id),
        ),
    )


def test_demo_summary_evaluator_scores_thresholds_but_is_not_approval_evidence() -> None:
    passing = DemoSummaryEvaluator(
        "tool_call_payload",
        "tool_call",
        {"proposal_success_rate": 1.0, "parse_failures": 0},
        (
            MetricThreshold("proposal_success_rate", min_value=1.0),
            MetricThreshold("parse_failures", equals=0),
        ),
    ).evaluate("candidate-demo")
    failing = DemoSummaryEvaluator(
        "tool_call_payload",
        "tool_call",
        {"proposal_success_rate": 0.5, "parse_failures": 1},
        (
            MetricThreshold("proposal_success_rate", min_value=1.0),
            MetricThreshold("parse_failures", equals=0),
        ),
    ).evaluate("candidate-demo")

    assert passing.passed
    assert passing.approval_reasons()
    assert not failing.passed
    assert len(failing.reasons) == 2


def test_script_evaluator_parses_summary_prefix() -> None:
    evaluator = ScriptEvaluator(
        "scripted",
        "regression",
        (
            sys.executable,
            "-c",
            "import json; print('SUMMARY ' + json.dumps({'score': 1.0}))",
        ),
        "SUMMARY ",
        (MetricThreshold("score", min_value=1.0),),
    )

    result = evaluator.evaluate("candidate-script")

    assert result.passed
    assert result.summary["score"] == 1.0
    assert result.summary["returncode"] == 0


def test_script_evaluator_passes_manifest_env() -> None:
    evaluator = ScriptEvaluator(
        "scripted-env",
        "regression",
        (
            sys.executable,
            "-c",
            "import json, os; print('SUMMARY ' + json.dumps({'driver': os.environ['VECL_MODEL_DRIVER']}))",
        ),
        "SUMMARY ",
        (MetricThreshold("driver", equals="openai"),),
        env={"VECL_MODEL_DRIVER": "openai"},
    )

    assert evaluator.evaluate("candidate-script").passed


def test_release_report_maps_categories_to_gate_checks_and_ledger_events() -> None:
    report = _approvable_report()
    ledger = ProvenanceLedger()

    assert report.passed
    assert all(report.gate_checks().values())
    approved = approve_release_report(report, ledger, "tenant")

    assert approved.event_type == EventType.RELEASE_APPROVED
    assert approved.payload["evaluation"]["report"]["candidate_id"] == "candidate-a"


def test_canned_default_evaluators_cannot_approve_arbitrary_candidate() -> None:
    report = run_release_evaluation(
        "arbitrary-candidate",
        demo_phase9a_evaluators(),
        manifest_hash=stable_hash({"kind": "demo"}),
    )

    with pytest.raises(ValueError, match="evidence"):
        approve_release_report(report, ProvenanceLedger(), "tenant")


def test_failed_report_rejects_and_records_reason() -> None:
    report = run_release_evaluation(
        "candidate-b",
        (
            DemoSummaryEvaluator(
                "ethics_eval",
                "ethics",
                {"blocked_specialist_calls": 1},
                (MetricThreshold("blocked_specialist_calls", equals=0),),
            ),
        ),
    )
    ledger = ProvenanceLedger()

    assert not report.passed
    assert not report.gate_checks()["adversarial_simulation_within_bound"]
    with pytest.raises(ValueError):
        approve_release_report(report, ledger, "tenant")
    rejected = reject_release_report(report, ledger, "tenant", "ethics failed")
    assert rejected.event_type == EventType.RELEASE_REJECTED
    assert rejected.payload["reason"] == "ethics failed"


def test_drift_report_maps_to_release_gate_fields() -> None:
    base_report = _approvable_report("candidate-drift")
    drift = _evidence_result(
        "fisher_drift",
        "drift",
        {"drift_value": 0.5, "drift_threshold": 1.0},
        candidate_id=base_report.candidate_id,
    )
    report = ReleaseEvaluationReport(
        candidate_id=base_report.candidate_id,
        results=(*base_report.results, drift),
        manifest_hash=base_report.manifest_hash,
    )
    ledger = ProvenanceLedger()

    assert report.passed
    assert report.gate_checks()["drift_within_bound"]
    approved = approve_release_report(report, ledger, "tenant")

    evaluation = approved.payload["evaluation"]
    assert evaluation["drift_within_bound"] is True
    assert evaluation["drift_value"] == 0.5
    assert evaluation["drift_threshold"] == 1.0


def test_release_report_save_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report = run_release_evaluation(
        "candidate-c",
        demo_phase9a_evaluators(),
        manifest_hash=stable_hash({"kind": "demo"}),
    )

    save_release_report(report, path)
    loaded = load_release_report(path)

    assert loaded.to_payload() == report.to_payload()


def test_release_report_load_rejects_tampered_summary(tmp_path: Path) -> None:
    path = tmp_path / "release-report.json"
    report = run_release_evaluation(
        "candidate-c",
        demo_phase9a_evaluators(),
        manifest_hash=stable_hash({"kind": "demo"}),
    )
    save_release_report(report, path)
    payload = json.loads(path.read_text())
    payload["results"][0]["summary"]["route_success_rate"] = 0.0
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="report hash"):
        load_release_report(path)


def test_evaluator_from_manifest_config() -> None:
    evaluator = evaluator_from_config(
        {
            "type": "demo_summary",
            "name": "demo-routing",
            "category": "routing",
            "summary": {"route_success_rate": 1.0},
            "thresholds": [{"metric": "route_success_rate", "min_value": 1.0}],
        }
    )

    assert evaluator.evaluate("candidate-demo").passed


def test_ambiguous_summary_evaluator_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="demo_summary"):
        evaluator_from_config(
            {
                "type": "summary",
                "name": "routing",
                "category": "routing",
                "summary": {"route_success_rate": 1.0},
            }
        )


def test_fisher_drift_evaluator_from_manifest_config() -> None:
    evaluator = evaluator_from_config(
        {
            "type": "fisher_drift",
            "name": "fisher_drift",
            "category": "drift",
            "approved_snapshot": "/tmp/approved.npz",
            "candidate_snapshot": "/tmp/candidate.npz",
            "drift_threshold": 1.0,
        }
    )

    assert isinstance(evaluator, FisherDriftEvaluator)


def test_checked_in_release_manifests_parse() -> None:
    root = Path(__file__).parents[2]
    for manifest in (
        root / "configs" / "release" / "phase9a-frontier-openai.json",
        root / "configs" / "release" / "phase9a-frontier-anthropic.json",
        root / "configs" / "release" / "phase9a-vertex-gemma.json",
    ):
        evaluators = evaluators_from_manifest(json.loads(manifest.read_text()))
        assert evaluators
        assert all(evaluator.name for evaluator in evaluators)


def test_cli_evaluate_and_approve_with_sqlite_ledger(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "release-report.json"
    ledger_path = tmp_path / "ledger.sqlite"
    manifest_path = tmp_path / "release-manifest.json"
    command = [
        sys.executable,
        "-c",
        "import json; print('SUMMARY ' + json.dumps({'ok': True}))",
    ]
    manifest = {
        "evaluators": [
            {
                "type": "script",
                "name": name,
                "category": category,
                "version": "test-v1",
                "command": command,
                "summary_prefix": "SUMMARY ",
                "thresholds": [{"metric": "ok", "equals": True}],
            }
            for name, category in (
                ("invariants", "invariant"),
                ("regressions", "regression"),
                ("ethics", "ethics"),
                ("rollback", "rollback"),
                ("calibration", "calibration"),
                ("tenant-isolation", "tenant_isolation"),
            )
        ]
    }
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr("vecl.evaluation.evaluators._git_state", lambda: ("a" * 40, True))

    assert (
        main(
            [
                "release",
                "evaluate",
                "candidate-cli",
                "--manifest",
                str(manifest_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    evaluate_output = json.loads(capsys.readouterr().out)
    assert evaluate_output["passed"] is True
    assert report_path.exists()

    assert (
        main(
            [
                "release",
                "approve",
                str(report_path),
                "--ledger-path",
                str(ledger_path),
                "--tenant-id",
                "tenant",
            ]
        )
        == 0
    )
    approve_output = json.loads(capsys.readouterr().out)
    assert approve_output["event_type"] == EventType.RELEASE_APPROVED.value

    with SqliteProvenanceLedger(ledger_path) as ledger:
        events = ledger.find_by_type(EventType.RELEASE_APPROVED)
    assert len(events) == 1


def test_missing_or_malformed_evidence_cannot_approve() -> None:
    valid = _approvable_report()
    missing = ReleaseEvaluationReport(
        candidate_id=valid.candidate_id,
        manifest_hash=valid.manifest_hash,
        results=tuple(replace(result, evidence=None) for result in valid.results),
    )
    first_evidence = valid.results[0].evidence
    assert first_evidence is not None
    malformed = ReleaseEvaluationReport(
        candidate_id=valid.candidate_id,
        manifest_hash=valid.manifest_hash,
        results=(
            replace(
                valid.results[0],
                evidence=replace(first_evidence, artifact_hashes={}),
            ),
            *valid.results[1:],
        ),
    )

    with pytest.raises(ValueError, match="missing evaluator evidence"):
        approve_release_report(missing, ProvenanceLedger(), "tenant")
    with pytest.raises(ValueError, match="artifact hashes"):
        approve_release_report(malformed, ProvenanceLedger(), "tenant")


def test_incomplete_gate_evidence_cannot_approve() -> None:
    valid = _approvable_report()
    incomplete = ReleaseEvaluationReport(
        candidate_id=valid.candidate_id,
        manifest_hash=valid.manifest_hash,
        results=valid.results[:2],
    )

    with pytest.raises(ValueError, match="required gate evidence missing"):
        approve_release_report(incomplete, ProvenanceLedger(), "tenant")


def test_configuration_hash_mismatch_cannot_approve() -> None:
    valid = _approvable_report()
    evidence = valid.results[0].evidence
    assert evidence is not None
    mismatched = ReleaseEvaluationReport(
        candidate_id=valid.candidate_id,
        manifest_hash=valid.manifest_hash,
        results=(
            replace(
                valid.results[0],
                evidence=replace(evidence, configuration={"different": True}),
            ),
            *valid.results[1:],
        ),
    )

    with pytest.raises(ValueError, match="configuration_hash does not match"):
        approve_release_report(mismatched, ProvenanceLedger(), "tenant")


def test_release_evidence_cannot_be_relabelled_for_another_candidate() -> None:
    original = _approvable_report("candidate-a")
    relabelled = ReleaseEvaluationReport(
        candidate_id="candidate-b",
        manifest_hash=original.manifest_hash,
        results=original.results,
    )

    with pytest.raises(ValueError, match="candidate"):
        approve_release_report(relabelled, ProvenanceLedger(), "tenant")


def test_nonfinite_metric_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        MetricThreshold("score", min_value=float("nan"))


def test_gate_recomputes_thresholds_instead_of_trusting_passed_flag() -> None:
    valid = _approvable_report()
    result = valid.results[0]
    evidence = result.evidence
    assert evidence is not None
    failing_summary = {"ok": False}
    forged = ReleaseEvaluationReport(
        candidate_id=valid.candidate_id,
        manifest_hash=valid.manifest_hash,
        results=(
            replace(
                result,
                passed=True,
                summary=failing_summary,
                evidence=replace(
                    evidence,
                    artifact_hashes={
                        **evidence.artifact_hashes,
                        "summary": stable_hash(failing_summary),
                    },
                ),
            ),
            *valid.results[1:],
        ),
    )

    with pytest.raises(ValueError, match="expected True"):
        approve_release_report(forged, ProvenanceLedger(), "tenant")


def test_release_evidence_must_come_from_one_git_revision() -> None:
    valid = _approvable_report()
    evidence = valid.results[0].evidence
    assert evidence is not None
    mixed_revision = ReleaseEvaluationReport(
        candidate_id=valid.candidate_id,
        manifest_hash=valid.manifest_hash,
        results=(
            replace(valid.results[0], evidence=replace(evidence, git_revision="b" * 40)),
            *valid.results[1:],
        ),
    )

    with pytest.raises(ValueError, match="same git revision"):
        approve_release_report(mixed_revision, ProvenanceLedger(), "tenant")


def test_low_level_gate_revalidates_candidate_binding() -> None:
    report = _approvable_report("candidate-a")
    gate = ReleaseGate(ProvenanceLedger(), "tenant")
    evaluation = gate.evaluate_candidate_checkpoint(
        "candidate-b",
        **report.gate_checks(),
        evidence_valid=True,
        report_hash=report.report_hash,
        report=report.to_payload(),
    )

    with pytest.raises(ValueError, match="candidate does not match checkpoint"):
        gate.approve(evaluation)


def test_cli_release_evaluate_requires_manifest() -> None:
    with pytest.raises(SystemExit):
        main(["release", "evaluate", "arbitrary-candidate"])


def test_cli_evaluate_returns_failure_for_synthetic_passing_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "demo-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "evaluators": [
                    {
                        "type": "demo_summary",
                        "name": "demo-invariant",
                        "category": "invariant",
                        "summary": {"ok": True},
                        "thresholds": [{"metric": "ok", "equals": True}],
                    }
                ]
            }
        )
    )

    assert (
        main(["release", "evaluate", "arbitrary-candidate", "--manifest", str(manifest_path)]) == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["approval_eligible"] is False
