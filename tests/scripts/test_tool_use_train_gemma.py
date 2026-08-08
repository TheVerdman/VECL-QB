from __future__ import annotations

import json
from pathlib import Path

from scripts.tool_use_train_gemma import (
    _any_probe_improved,
    _coverage_report,
    _diagnostic_loss_summary,
    _domain_filter,
    _expected_domains,
    _interference_report,
    _load_examples,
    _overfit_examples,
    _score_tool_call_generation,
    _sparse_cycle_summary,
    _stratified_sample,
    _summary_passed,
    _tool_call_generation_delta,
    _tool_call_generation_metrics,
    _tool_call_generation_prompt,
    _tool_call_probe_cards,
    _training_examples,
    _training_prompt_builder,
)
from vecl.qb.tool_call import ToolCallValidationConfig
from vecl.training.tool_use_loop import SupervisedToolUseExample


def test_tool_use_training_summary_predicate() -> None:
    summary = {
        "sample_count": 4,
        "torch_seed": 1234,
        "sample_seed": 1107,
        "slot_count": 4,
        "selected_slot_count": 2,
        "learning_event_committed": True,
        "probe_any_improved": True,
        "representative_tensor_delta_norm": 0.25,
        "base_weights_unchanged": True,
        "selected_lora_slots_changed": True,
        "unselected_lora_slots_unchanged": True,
        "ewc_lambda": 0.0,
        "gcs_output_uri": "",
        "selection_diagnostics_hash": "diag-hash",
        "selection_diagnostics_uploaded": False,
        "baseline_id": None,
        "baseline_snapshot_hash": None,
        "baseline_restored": False,
        "baseline_uploaded": False,
        "snapshots_uploaded": False,
        "interference_report": {"passed": True},
        "coverage_report": {"passed": True},
        "candidate_decision": "accepted",
        "expect_rejection": False,
        "inline_drift_required": False,
    }
    assert _summary_passed(summary)
    assert not _summary_passed({**summary, "selected_slot_count": 0})
    assert not _summary_passed({**summary, "probe_any_improved": False})
    assert not _summary_passed({**summary, "representative_tensor_delta_norm": 0.0})
    assert not _summary_passed({**summary, "gcs_output_uri": "gs://bucket/path"})
    assert _summary_passed(
        {**summary, "gcs_output_uri": "gs://bucket/path", "snapshots_uploaded": True}
        | {"selection_diagnostics_uploaded": True}
    )
    assert not _summary_passed({**summary, "selection_diagnostics_hash": None})
    assert not _summary_passed({**summary, "baseline_id": "tool_use_v0"})
    assert not _summary_passed(
        {
            **summary,
            "baseline_id": "tool_use_v0",
            "baseline_snapshot_hash": "hash",
            "baseline_restored": False,
        }
    )
    assert _summary_passed(
        {
            **summary,
            "baseline_id": "tool_use_v0",
            "baseline_snapshot_hash": "hash",
            "baseline_restored": True,
        }
    )
    assert not _summary_passed(
        {**summary, "interference_report": {"passed": False, "violations": [{"domain": "sympy"}]}}
    )
    assert not _summary_passed(
        {**summary, "coverage_report": {"passed": False, "missing_train_domains": ["eda"]}}
    )
    assert not _summary_passed({**summary, "ewc_lambda": 0.1})
    assert not _summary_passed(
        {**summary, "inline_drift_required": True, "ewc_drift_within_bound": None}
    )
    assert _summary_passed(
        {
            **summary,
            "inline_drift_required": True,
            "ewc_approved_snapshot_hash": "hash",
            "ewc_drift_within_bound": True,
        }
    )
    assert not _summary_passed(
        {
            **summary,
            "ewc_lambda": 0.1,
            "ewc_approved_snapshot_hash": "hash",
            "ewc_drift_within_bound": False,
        }
    )
    assert _summary_passed(
        {
            **summary,
            "ewc_lambda": 0.1,
            "ewc_approved_snapshot_hash": "hash",
            "ewc_drift_within_bound": True,
        }
    )
    assert _summary_passed(
        {
            **summary,
            "expect_rejection": True,
            "learning_event_committed": False,
            "probe_any_improved": False,
            "ewc_lambda": 0.1,
            "ewc_approved_snapshot_hash": "hash",
            "ewc_drift_within_bound": False,
            "candidate_decision": "rejected_drift_exceeded",
            "start_snapshot_hash": "start",
            "restored_snapshot_hash": "start",
        }
    )
    assert not _summary_passed(
        {
            **summary,
            "expect_rejection": True,
            "learning_event_committed": False,
            "ewc_lambda": 0.1,
            "ewc_approved_snapshot_hash": "hash",
            "ewc_drift_within_bound": False,
            "candidate_decision": "accepted",
            "start_snapshot_hash": "start",
            "restored_snapshot_hash": "start",
        }
    )


def test_tool_use_training_probe_improvement_finds_nested_metrics() -> None:
    assert _any_probe_improved({"heldout": {"any_improved": True}})
    assert _any_probe_improved({"aggregate": {"improved_count": 1, "any_improved": True}})
    assert not _any_probe_improved({"heldout": {"any_improved": False}})


def test_tool_use_training_interference_report_checks_non_trained_domains() -> None:
    metrics = {
        "aggregate": {
            "by_domain": {
                "stockfish": {
                    "relative_mean_change": -0.02,
                    "mean_before": 2.0,
                    "mean_after": 1.96,
                    "sample_count": 8,
                },
                "sympy": {
                    "relative_mean_change": 0.08,
                    "mean_before": 3.0,
                    "mean_after": 3.24,
                    "sample_count": 8,
                    "paired_t_normal_approx_p_value": 0.03,
                },
            }
        }
    }

    report = _interference_report(metrics, trained_domains={"stockfish"}, threshold=0.05)

    assert report["checked"] is True
    assert report["passed"] is False
    assert report["violations"][0]["domain"] == "sympy"


def test_tool_use_training_interference_report_skips_mixed_domain_runs() -> None:
    report = _interference_report({}, trained_domains=set(), threshold=0.05)

    assert report["checked"] is False
    assert report["passed"] is True


def test_tool_use_training_domain_filter_all_sentinel(monkeypatch) -> None:
    monkeypatch.setenv("VECL_TRAIN_DOMAIN_MIX", "__all__")
    assert _domain_filter() == set()
    monkeypatch.setenv("VECL_TRAIN_DOMAIN_MIX", "stockfish,sympy")
    assert _domain_filter() == {"stockfish", "sympy"}


def test_tool_use_training_expected_domains(monkeypatch) -> None:
    monkeypatch.delenv("VECL_TRAIN_EXPECT_DOMAINS", raising=False)
    assert _expected_domains() == set()
    monkeypatch.setenv("VECL_TRAIN_EXPECT_DOMAINS", "__all__")
    assert {"eda", "stockfish", "terraform"} <= _expected_domains()
    monkeypatch.setenv("VECL_TRAIN_EXPECT_DOMAINS", "stockfish,eda")
    assert _expected_domains() == {"stockfish", "eda"}


def test_tool_use_training_export_loader_and_stratified_sample(tmp_path: Path) -> None:
    examples = [
        SupervisedToolUseExample(
            example_id=f"example-{index}",
            tenant_id="tenant",
            source_id="source",
            authority=0.9,
            prompt=f"prompt {index}",
            target_text=f"target {index}",
            task_kind="tool_call_json",
            metadata={
                "corpus_domain": "stockfish" if index % 2 else "sympy",
                "corpus_category": "tool_call",
            },
        )
        for index in range(6)
    ]
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(example.to_payload(), sort_keys=True) for example in examples) + "\n",
        encoding="utf-8",
    )

    loaded = _load_examples(path)
    sample = _stratified_sample(loaded, count=4, seed=7)

    assert len(loaded) == 6
    assert len(sample) == 4
    assert {example.metadata["corpus_domain"] for example in sample} == {
        "stockfish",
        "sympy",
    }


def test_tool_use_training_source_records_sample_seed(monkeypatch, tmp_path: Path) -> None:
    export_dir = tmp_path / "exports"
    export_dir.mkdir()
    examples = [
        SupervisedToolUseExample(
            example_id=f"example-{index}",
            tenant_id="tenant",
            source_id="source",
            authority=0.9,
            prompt=f"prompt {index}",
            target_text=f"target {index}",
            task_kind="tool_call_json",
            metadata={
                "corpus_domain": "stockfish" if index % 2 else "sympy",
                "corpus_category": "tool_call",
            },
        )
        for index in range(4)
    ]
    payload = "\n".join(json.dumps(example.to_payload(), sort_keys=True) for example in examples)
    for name in ("train.jsonl", "heldout.jsonl", "hard-heldout.jsonl"):
        (export_dir / name).write_text(payload + "\n", encoding="utf-8")
    monkeypatch.setenv("VECL_TRAIN_CORPUS_EXPORT_DIR", str(export_dir))
    monkeypatch.setenv("VECL_TRAIN_SAMPLE_COUNT", "2")
    monkeypatch.setenv("VECL_TRAIN_SMOKE_PROBE_COUNT", "1")
    monkeypatch.setenv("VECL_TRAIN_HELDOUT_PROBE_COUNT", "1")
    monkeypatch.setenv("VECL_TRAIN_HARD_PROBE_COUNT", "1")
    monkeypatch.setenv("VECL_TRAIN_SAMPLE_SEED", "99")

    train, probes, source = _training_examples(tmp_path)

    assert len(train) == 2
    assert set(probes) == {"smoke", "heldout", "hard_heldout"}
    assert source["sample_seed"] == 99


def test_tool_call_generation_scoring_and_metrics() -> None:
    expected = (
        '{"specialist_id":"sympy","task_type":"symbolic_math",'
        '"input_payload":{"operation":"simplify","expression":"(x + 1)^2"},'
        '"confidence":0.9,"reasoning":"math"}'
    )
    exact = expected
    wrong_payload = (
        '{"specialist_id":"sympy","task_type":"symbolic_math",'
        '"input_payload":{"operation":"factor","expression":"(x + 1)^2"},'
        '"confidence":0.9,"reasoning":"math"}'
    )
    cards = _tool_call_probe_cards()
    config = ToolCallValidationConfig(
        tenant_id="test",
        request_id="test",
        terraform_config_dir="/tmp/terraform-fixture",
    )

    exact_row = _score_tool_call_generation(
        generated_text=exact,
        expected_text=expected,
        cards=cards,
        config=config,
    )
    wrong_row = _score_tool_call_generation(
        generated_text=wrong_payload,
        expected_text=expected,
        cards=cards,
        config=config,
    )
    parse_row = _score_tool_call_generation(
        generated_text="not json",
        expected_text=expected,
        cards=cards,
        config=config,
    )

    assert exact_row["all_correct"] is True
    assert wrong_row["validation_ok"] is True
    assert wrong_row["payload_match"] is False
    assert parse_row["parse_ok"] is False
    metrics = _tool_call_generation_metrics([exact_row, wrong_row, parse_row])
    assert metrics["sample_count"] == 3
    assert metrics["parse_success"] == 2
    assert metrics["all_correct"] == 1
    delta = _tool_call_generation_delta([wrong_row], [exact_row])
    assert delta["improved_count"] == 1
    assert delta["exact_tool_call_accuracy_delta"] == 1.0


def test_tool_call_generation_prompt_uses_author_contract() -> None:
    example = SupervisedToolUseExample(
        example_id="example-sympy",
        tenant_id="tenant",
        source_id="source",
        authority=0.9,
        prompt="Simplify (x + 1)^2.",
        target_text=(
            '{"specialist_id":"sympy","task_type":"symbolic_math",'
            '"input_payload":{"operation":"simplify","expression":"(x + 1)^2"},'
            '"confidence":0.9,"reasoning":"math"}'
        ),
        task_kind="tool_call_json",
        metadata={"corpus_domain": "sympy", "corpus_category": "sympy_tool_call"},
    )

    prompt = _tool_call_generation_prompt(example, list(_tool_call_probe_cards().values()))

    assert "You are VECL-QB's tool-call author" in prompt
    assert "Specialist input payload contracts" in prompt
    assert "task_type_hint: symbolic_math" in prompt
    assert "Simplify (x + 1)^2." in prompt


def test_training_prompt_builder_wraps_tool_calls_only(monkeypatch) -> None:
    tool_call = SupervisedToolUseExample(
        example_id="example-sympy",
        tenant_id="tenant",
        source_id="source",
        authority=0.9,
        prompt="Simplify (x + 1)^2.",
        target_text=(
            '{"specialist_id":"sympy","task_type":"symbolic_math",'
            '"input_payload":{"operation":"simplify","expression":"(x + 1)^2"},'
            '"confidence":0.9,"reasoning":"math"}'
        ),
        task_kind="tool_call_json",
        metadata={"corpus_domain": "sympy", "corpus_category": "sympy_tool_call"},
    )
    final_answer = SupervisedToolUseExample(
        example_id="example-answer",
        tenant_id="tenant",
        source_id="source",
        authority=0.9,
        prompt="Use this verified claim.",
        target_text="Grounded answer.",
        task_kind="final_answer",
    )

    monkeypatch.setenv("VECL_TRAIN_PROMPT_MODE", "tool_call_author")
    mode, builder = _training_prompt_builder()

    assert mode == "tool_call_author"
    assert builder is not None
    assert "You are VECL-QB's tool-call author" in builder(tool_call)
    assert builder(final_answer) == final_answer.prompt

    monkeypatch.setenv("VECL_TRAIN_PROMPT_MODE", "raw")
    mode, builder = _training_prompt_builder()

    assert mode == "raw"
    assert builder is None


def test_overfit_example_selection_and_loss_summary() -> None:
    examples = [
        SupervisedToolUseExample(
            example_id=f"example-{index}",
            tenant_id="tenant",
            source_id="source",
            authority=0.9,
            prompt=f"prompt {index}",
            target_text=f"target {index}",
            task_kind="tool_call_json" if index < 3 else "final_answer",
            metadata={"corpus_domain": "stockfish", "corpus_category": "tool_call"},
        )
        for index in range(5)
    ]

    selected = _overfit_examples(examples, sample_seed=7)
    summary = _diagnostic_loss_summary([3.0, 2.0], [1.0, 2.5])

    assert selected
    assert all(example.task_kind == "tool_call_json" for example in selected)
    assert summary["mean_before"] == 2.5
    assert summary["mean_after"] == 1.75
    assert summary["mean_drop"] == 0.75
    assert summary["improved_count"] == 1
    assert summary["worsened_count"] == 1


def test_sparse_cycle_summary_reports_loss_and_delta_norm() -> None:
    summary = _sparse_cycle_summary(
        cycle=2,
        report={
            "committed": True,
            "selected_slots": [1, 3],
            "tensor_delta_records": [
                {"total_delta_norm": 0.25},
                {"total_delta_norm": 0.75},
            ],
            "before_snapshot_hash": "before",
            "after_snapshot_hash": "after",
            "candidate_decision": "accepted",
        },
        before_losses=[2.0, 4.0],
        after_losses=[1.0, 3.0],
    )

    assert summary["cycle"] == 2
    assert summary["committed"] is True
    assert summary["selected_slot_count"] == 2
    assert summary["tensor_delta_norm_sum"] == 1.0
    assert summary["loss_summary"]["mean_drop"] == 1.0


def test_tool_use_training_coverage_report_checks_train_and_probe_domains() -> None:
    examples = [
        SupervisedToolUseExample(
            example_id=f"example-{domain}",
            tenant_id="tenant",
            source_id="source",
            authority=0.9,
            prompt=f"prompt {domain}",
            target_text=f"target {domain}",
            task_kind="tool_call_json",
            metadata={"corpus_domain": domain, "corpus_category": "tool_call"},
        )
        for domain in ("stockfish", "eda")
    ]

    report = _coverage_report(
        train_examples=examples,
        probe_sets={"heldout": examples},
        expected_domains={"stockfish", "eda"},
    )
    assert report["passed"] is True

    missing = _coverage_report(
        train_examples=examples[:1],
        probe_sets={"heldout": examples[:1]},
        expected_domains={"stockfish", "eda"},
    )
    assert missing["passed"] is False
    assert missing["missing_train_domains"] == ["eda"]
    assert missing["missing_probe_domains"] == {"heldout": ["eda"]}


def test_tool_use_training_coverage_report_ignores_smoke_by_default() -> None:
    examples = [
        SupervisedToolUseExample(
            example_id=f"example-{domain}",
            tenant_id="tenant",
            source_id="source",
            authority=0.9,
            prompt=f"prompt {domain}",
            target_text=f"target {domain}",
            task_kind="tool_call_json",
            metadata={"corpus_domain": domain, "corpus_category": "tool_call"},
        )
        for domain in ("stockfish", "eda")
    ]

    report = _coverage_report(
        train_examples=examples,
        probe_sets={"smoke": examples[:1], "heldout": examples},
        expected_domains={"stockfish", "eda"},
    )

    assert report["passed"] is True
    assert report["required_probe_names"] == ["heldout"]
