from __future__ import annotations

import json
from pathlib import Path

from scripts.tool_use_train_gemma import (
    _any_probe_improved,
    _coverage_report,
    _domain_filter,
    _expected_domains,
    _interference_report,
    _load_examples,
    _stratified_sample,
    _summary_passed,
    _training_examples,
)
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
