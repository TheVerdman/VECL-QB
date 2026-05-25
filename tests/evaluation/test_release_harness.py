from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vecl.cli import main
from vecl.evaluation.evaluators import (
    FisherDriftEvaluator,
    ScriptEvaluator,
    SummaryEvaluator,
    evaluator_from_config,
    evaluators_from_manifest,
)
from vecl.evaluation.release import (
    approve_release_report,
    default_phase9a_evaluators,
    load_release_report,
    reject_release_report,
    run_release_evaluation,
    save_release_report,
)
from vecl.evaluation.types import MetricThreshold
from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger, SqliteProvenanceLedger


def test_summary_evaluator_scores_thresholds() -> None:
    passing = SummaryEvaluator(
        "tool_call_payload",
        "tool_call",
        {"proposal_success_rate": 1.0, "parse_failures": 0},
        (
            MetricThreshold("proposal_success_rate", min_value=1.0),
            MetricThreshold("parse_failures", equals=0),
        ),
    ).evaluate()
    failing = SummaryEvaluator(
        "tool_call_payload",
        "tool_call",
        {"proposal_success_rate": 0.5, "parse_failures": 1},
        (
            MetricThreshold("proposal_success_rate", min_value=1.0),
            MetricThreshold("parse_failures", equals=0),
        ),
    ).evaluate()

    assert passing.passed
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

    result = evaluator.evaluate()

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

    assert evaluator.evaluate().passed


def test_release_report_maps_categories_to_gate_checks_and_ledger_events() -> None:
    report = run_release_evaluation("candidate-a", default_phase9a_evaluators())
    ledger = ProvenanceLedger()

    assert report.passed
    assert all(report.gate_checks().values())
    approved = approve_release_report(report, ledger, "tenant")

    assert approved.event_type == EventType.RELEASE_APPROVED
    assert approved.payload["evaluation"]["report"]["candidate_id"] == "candidate-a"


def test_failed_report_rejects_and_records_reason() -> None:
    report = run_release_evaluation(
        "candidate-b",
        (
            SummaryEvaluator(
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
    report = run_release_evaluation(
        "candidate-drift",
        (
            SummaryEvaluator(
                "fisher_drift",
                "drift",
                {"drift_value": 0.5, "drift_threshold": 1.0},
                (MetricThreshold("drift_value", max_value=1.0),),
            ),
        ),
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
    report = run_release_evaluation("candidate-c", default_phase9a_evaluators())

    save_release_report(report, path)
    loaded = load_release_report(path)

    assert loaded.to_payload() == report.to_payload()


def test_evaluator_from_manifest_config() -> None:
    evaluator = evaluator_from_config(
        {
            "type": "summary",
            "name": "routing",
            "category": "routing",
            "summary": {"route_success_rate": 1.0},
            "thresholds": [{"metric": "route_success_rate", "min_value": 1.0}],
        }
    )

    assert evaluator.evaluate().passed


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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "release-report.json"
    ledger_path = tmp_path / "ledger.sqlite"

    assert main(["release", "evaluate", "candidate-cli", "--output", str(report_path)]) == 0
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
