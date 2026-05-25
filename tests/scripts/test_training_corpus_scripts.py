from __future__ import annotations

import json
from pathlib import Path

import scripts.generate_training_corpus as generate_script
from scripts.export_training_examples import main as export_main
from scripts.generate_training_corpus import main as generate_main
from scripts.report_training_corpus import main as report_main
from scripts.validate_training_corpus import main as validate_main
from vecl.qb.model_driver import ModelDriverResult
from vecl.training.corpus_schema import read_llm_cache_jsonl


def test_training_corpus_scripts_generate_validate_and_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "sample-corpus"
    report_path = tmp_path / "report.md"

    assert (
        generate_main(
            [
                "--size",
                "sample",
                "--count",
                "64",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert (output_dir / "corpus.jsonl").exists()
    assert validate_main([str(output_dir), "--min-records", "64", "--max-records", "64"]) == 0
    assert report_main([str(output_dir), "--output", str(report_path)]) == 0
    assert export_main([str(output_dir), "--output-dir", str(tmp_path / "exports")]) == 0
    assert "# VECL-QB Synthetic Corpus" in report_path.read_text()
    assert (tmp_path / "exports" / "train.jsonl").exists()


def test_v1_hard_training_corpus_script_path_exports_hard_heldout(tmp_path: Path) -> None:
    output_dir = tmp_path / "v1-hard"
    export_dir = tmp_path / "v1-exports"

    assert (
        generate_main(
            [
                "--size",
                "v1-hard",
                "--count",
                "200",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert validate_main([str(output_dir), "--min-records", "200", "--max-records", "200"]) == 0
    assert export_main([str(output_dir), "--output-dir", str(export_dir)]) == 0
    assert (export_dir / "hard-heldout.jsonl").exists()


def test_generate_script_checkpoint_cache_resumes_without_provider_call(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "checkpointed"
    args = [
        "--size",
        "sample",
        "--count",
        "64",
        "--output-dir",
        str(output_dir),
        "--llm-augment",
        "--llm-driver",
        "fake",
        "--llm-limit",
        "1",
        "--checkpoint-cache",
    ]

    first_driver = _ScriptCheckpointDriver()
    monkeypatch.setattr(generate_script, "resolve_model_driver", lambda _name: first_driver)
    assert generate_main(args) == 0
    assert first_driver.calls == 1
    assert len(read_llm_cache_jsonl(output_dir / "llm_raw_cache.jsonl")) == 1

    resumed_driver = _ScriptExplodingDriver()
    monkeypatch.setattr(generate_script, "resolve_model_driver", lambda _name: resumed_driver)
    assert generate_main(args) == 0
    assert resumed_driver.calls == 0
    assert len(read_llm_cache_jsonl(output_dir / "llm_raw_cache.jsonl")) == 1


class _ScriptCheckpointDriver:
    provider = "fake"
    model_id = "fake-augmentor"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, prompt: str) -> ModelDriverResult:
        self.calls += 1
        seed = json.loads(prompt.split("Seed record:\n", maxsplit=1)[1])
        candidate = {
            "prompt": f"Checkpoint paraphrase for {seed['domain']}.",
            "target_text": seed["target_text"],
            "task_kind": seed["task_kind"],
            "target_format": seed["target_format"],
            "executable": seed["executable"],
            "specialist_id": seed.get("specialist_id"),
            "task_type": seed.get("task_type"),
            "metadata": {"target_source": "copied_seed_target"},
        }
        return ModelDriverResult(
            provider=self.provider,
            model_id=self.model_id,
            raw_text=json.dumps([candidate], sort_keys=True),
            usage={"input_tokens": 100, "output_tokens": 50},
            metadata={"estimated_cost_usd": 0.01},
        )


class _ScriptExplodingDriver:
    provider = "fake"
    model_id = "fake-augmentor"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, prompt: str) -> ModelDriverResult:
        self.calls += 1
        raise AssertionError("checkpoint resume should not call synthesize")
