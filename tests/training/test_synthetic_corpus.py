from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from vecl.qb.model_driver import ModelDriverResult
from vecl.training.corpus_factory import (
    SyntheticCorpusConfig,
    augment_records_with_llm,
    generate_deterministic_records,
    generate_synthetic_corpus,
)
from vecl.training.corpus_metrics import semantic_diversity_metrics, training_export_records
from vecl.training.corpus_schema import (
    JSON_TASK_KINDS,
    LLMRawCacheEntry,
    ToolUseCorpusRecord,
    append_llm_cache_jsonl,
    corpus_hash,
    read_corpus_jsonl,
    read_llm_cache_jsonl,
    stable_example_id,
    stable_split_for_id,
    supervised_examples_from_corpus,
    write_corpus_jsonl,
    write_llm_cache_jsonl,
)
from vecl.training.corpus_validation import (
    single_record_acceptor,
    validate_corpus_records,
    validate_single_record,
)


def test_corpus_schema_roundtrip_json_and_conversion(tmp_path: Path) -> None:
    records = generate_deterministic_records(SyntheticCorpusConfig(size="sample", target_count=64))
    result = validate_corpus_records(records)
    assert result.valid
    assert result.total_records == 64
    assert result.supervised_records == sum(1 for record in records if record.executable)

    path = tmp_path / "corpus.jsonl"
    write_corpus_jsonl(records, path)
    reloaded = read_corpus_jsonl(path)

    assert [record.to_json_obj() for record in reloaded] == [
        record.to_json_obj() for record in sorted(records, key=lambda item: item.example_id)
    ]
    for record in reloaded:
        if record.task_kind in JSON_TASK_KINDS:
            assert isinstance(json.loads(record.target_text), dict)
    supervised = supervised_examples_from_corpus(reloaded)
    assert supervised
    assert all(example.prompt and example.target_text for example in supervised)


def test_deterministic_generation_dedupe_and_split_integrity() -> None:
    config = SyntheticCorpusConfig(size="v0-small", target_count=420, seed=123)
    first = generate_deterministic_records(config)
    second = generate_deterministic_records(config)

    assert corpus_hash(first) == corpus_hash(second)
    assert len({record.example_id for record in first}) == len(first)
    assert len({(record.prompt, record.target_text) for record in first}) == len(first)
    assert {record.split for record in first} == {"train", "heldout"}
    for record in first:
        assert record.example_id == stable_example_id(record)
        assert record.split == stable_split_for_id(record.example_id)


def test_v1_hard_generation_has_difficulty_tags_and_hard_heldout() -> None:
    records = generate_deterministic_records(
        SyntheticCorpusConfig(size="v1-hard", target_count=200, seed=321)
    )
    result = validate_corpus_records(records)

    assert result.valid
    assert result.split_counts["hard-heldout"] > 0
    assert all(record.dataset_version == "tool_use_v1_hard" for record in records)
    assert all(record.metadata["difficulty_tags"] for record in records)
    assert any("ethics_boundary" in record.metadata["difficulty_tags"] for record in records)


def test_eda_corpus_generation_has_real_yosys_openroad_and_chain_examples() -> None:
    records = generate_deterministic_records(
        SyntheticCorpusConfig(size="v1-hard", target_count=500, seed=321)
    )
    result = validate_corpus_records(records)
    eda_records = [record for record in records if record.domain == "eda"]
    categories = {record.category for record in eda_records}
    chain_records = [
        record for record in eda_records if record.category == "eda_yosys_openroad_chain"
    ]
    exportable = training_export_records(records)

    assert result.valid
    assert "yosys_openroad_placeholder" not in categories
    assert {
        "eda_yosys_tool_call",
        "eda_openroad_tool_call",
        "eda_yosys_openroad_chain",
        "eda_final_answer",
        "eda_negative",
    } <= categories
    assert any(
        record.specialist_id == "yosys" and record.executable
        for record in eda_records
        if record.category == "eda_yosys_tool_call"
    )
    assert any(
        record.specialist_id == "openroad" and record.executable
        for record in eda_records
        if record.category == "eda_openroad_tool_call"
    )
    assert chain_records
    for record in chain_records:
        payload = record.target_json()
        assert payload["specialist_id"] == "openroad"
        assert payload["task_type"] == "eda_flow"
        assert payload["input_payload"]["netlist_from"] == "synthesize"
        assert record.metadata["chain"] == ["yosys-cli", "openroad-cli"]
        assert record.metadata["chain_plan_id"] == "eda-yosys-openroad"
        assert "synthesize" in {step["step_id"] for step in record.metadata["chain_steps"]}
    assert any(record.category == "eda_final_answer" for record in exportable)
    assert not any(record.category == "eda_negative" for record in exportable)


def test_validator_rejects_terraform_mutation_and_secret_like_text() -> None:
    terraform = next(
        record
        for record in generate_deterministic_records(SyntheticCorpusConfig(target_count=64))
        if record.category == "terraform_plan_tool_call"
    )
    payload = terraform.target_json()
    payload["input_payload"]["operation"] = "apply"
    bad_terraform = _with_updates(
        terraform,
        target_text=json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )

    safety = _with_updates(terraform, prompt="Contact jane@example.com before planning.")

    terraform_issues = validate_single_record(bad_terraform)
    safety_issues = validate_single_record(safety)

    assert any(issue.check == "terraform" for issue in terraform_issues)
    assert any(issue.check == "safety" for issue in safety_issues)


def test_mocked_llm_augmentation_caches_raw_output_and_uses_validator() -> None:
    seed_record = next(
        record
        for record in generate_deterministic_records(SyntheticCorpusConfig(target_count=64))
        if record.category == "stockfish_tool_call"
    )
    candidate = {
        "prompt": "Please send this synthetic chess FEN to Stockfish at the requested depth.",
        "target_text": seed_record.target_text,
        "task_kind": "tool_call_json",
        "target_format": "json",
        "executable": True,
        "specialist_id": "stockfish",
        "task_type": "chess_eval",
    }

    augmented, cache = augment_records_with_llm(
        seed_records=[seed_record],
        driver=_FakeDriver([candidate]),
        limit=1,
        seed=999,
        record_validator=single_record_acceptor,
    )

    assert len(augmented) == 1
    assert validate_corpus_records([seed_record, *augmented], require_category_coverage=False).valid
    assert len(cache) == 1
    assert cache[0].provider == "fake"
    assert cache[0].model_id == "fake-augmentor"
    assert cache[0].estimated_cost_usd == 0.0123
    assert cache[0].usage["input_tokens"] == 100
    assert cache[0].accepted_example_ids == (augmented[0].example_id,)
    assert cache[0].validator_outcome == "accepted"


def test_llm_augmentation_caches_provider_errors_and_continues() -> None:
    seed_records = [
        record
        for record in generate_deterministic_records(SyntheticCorpusConfig(target_count=64))
        if record.category == "stockfish_tool_call"
    ][:2]

    augmented, cache = augment_records_with_llm(
        seed_records=seed_records,
        driver=_FlakyDriver(seed_records[1].target_text),
        limit=2,
        seed=999,
        record_validator=single_record_acceptor,
    )

    assert len(augmented) == 1
    assert len(cache) == 2
    assert cache[0].validator_outcome == "provider_error"
    assert "TimeoutError" in cache[0].rejected_reasons[0]
    assert cache[0].prompt
    assert cache[1].validator_outcome == "accepted"


def test_llm_cache_checkpoint_append_and_resume(tmp_path: Path) -> None:
    seed_record = next(
        record
        for record in generate_deterministic_records(SyntheticCorpusConfig(target_count=64))
        if record.category == "stockfish_tool_call"
    )
    candidate = {
        "prompt": "Checkpoint paraphrase: send this synthetic chess FEN to Stockfish.",
        "target_text": seed_record.target_text,
        "task_kind": "tool_call_json",
        "target_format": "json",
        "executable": True,
        "specialist_id": "stockfish",
        "task_type": "chess_eval",
    }
    checkpoint_path = tmp_path / "llm_raw_cache.jsonl"

    first_augmented, first_cache = augment_records_with_llm(
        seed_records=[seed_record],
        driver=_FakeDriver([candidate]),
        limit=1,
        seed=999,
        record_validator=single_record_acceptor,
        cache_entry_checkpoint=lambda entry: append_llm_cache_jsonl(entry, checkpoint_path),
    )
    resumed_driver = _ExplodingDriver()
    resumed_augmented, resumed_cache = augment_records_with_llm(
        seed_records=[seed_record],
        driver=resumed_driver,
        limit=1,
        seed=999,
        record_validator=single_record_acceptor,
        existing_cache_entries=read_llm_cache_jsonl(checkpoint_path),
        cache_entry_checkpoint=lambda entry: append_llm_cache_jsonl(entry, checkpoint_path),
    )

    assert [entry.to_json_obj() for entry in read_llm_cache_jsonl(checkpoint_path)] == [
        entry.to_json_obj() for entry in first_cache
    ]
    assert [record.example_id for record in resumed_augmented] == [
        record.example_id for record in first_augmented
    ]
    assert [entry.to_json_obj() for entry in resumed_cache] == [
        entry.to_json_obj() for entry in first_cache
    ]
    assert resumed_driver.calls == 0


def test_llm_cache_read_can_ignore_truncated_tail(tmp_path: Path) -> None:
    entry = LLMRawCacheEntry(
        provider="fake",
        model_id="fake-augmentor",
        prompt_hash="abc123",
        prompt="synthetic prompt",
        raw_output="[]",
        parsed_output=[],
        seed=1,
        topic_id="topic",
        validator_outcome="rejected",
    )
    checkpoint_path = tmp_path / "cache.jsonl"
    append_llm_cache_jsonl(entry, checkpoint_path)
    checkpoint_path.write_text(
        checkpoint_path.read_text(encoding="utf-8") + '{"provider":',
        encoding="utf-8",
    )

    assert read_llm_cache_jsonl(checkpoint_path, allow_truncated_tail=True) == [entry]


def test_synthetic_corpus_build_tracks_llm_cost_and_cache_roundtrip(tmp_path: Path) -> None:
    seed_record = next(
        record
        for record in generate_deterministic_records(
            SyntheticCorpusConfig(size="v1-hard", target_count=64)
        )
        if record.category == "stockfish_tool_call"
    )
    candidate = {
        "prompt": "Hard paraphrase: route only the chess FEN to Stockfish.",
        "target_text": seed_record.target_text,
        "task_kind": "tool_call_json",
        "target_format": "json",
        "executable": True,
        "specialist_id": "stockfish",
        "task_type": "chess_eval",
        "metadata": {
            "target_source": "copied_seed_target",
            "difficulty_tags": ["paraphrase", "ambiguous"],
        },
    }
    build = generate_synthetic_corpus(
        SyntheticCorpusConfig(size="v1-hard", target_count=64),
        llm_driver=_FakeDriver([candidate]),
        llm_limit=1,
        record_validator=single_record_acceptor,
    )

    assert build.cost_by_provider == {"fake": 0.0123}
    assert len(build.llm_admitted_records) == 1
    assert build.llm_cache_entries[0].estimated_cost_usd == 0.0123
    cache_path = tmp_path / "cache.jsonl"
    write_llm_cache_jsonl(build.llm_cache_entries, cache_path)
    assert read_llm_cache_jsonl(cache_path)[0].estimated_cost_usd == 0.0123


def test_semantic_diversity_metrics_are_deterministic() -> None:
    records = generate_deterministic_records(
        SyntheticCorpusConfig(size="v1-hard", target_count=200, seed=777)
    )
    first = semantic_diversity_metrics(records)
    second = semantic_diversity_metrics(records)

    assert first == second
    assert first["record_count"] == 200
    assert first["unique_prompt_ratio"] == 1.0
    assert first["unique_target_ratio"] <= 1.0
    assert first["top_target_share"] > 0.0
    assert first["semantic_signature_count"] > 10
    assert 0.0 <= first["sampled_mean_prompt_jaccard"] <= 1.0
    assert 0.0 <= first["sampled_mean_target_jaccard"] <= 1.0


def test_training_export_records_match_default_export_policy() -> None:
    records = generate_deterministic_records(
        SyntheticCorpusConfig(size="v1-hard", target_count=200, seed=888)
    )
    exportable = training_export_records(records)

    assert exportable
    assert all(record.executable for record in exportable)
    assert all(
        record.metadata.get("training_export_eligible") is not False for record in exportable
    )


class _FakeDriver:
    provider = "fake"
    model_id = "fake-augmentor"

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def synthesize(self, prompt: str) -> ModelDriverResult:
        assert "validator decides" in prompt
        return ModelDriverResult(
            provider=self.provider,
            model_id=self.model_id,
            raw_text=json.dumps(self.payload, sort_keys=True),
            usage={"input_tokens": 100, "output_tokens": 50},
            metadata={"estimated_cost_usd": 0.0123},
        )


class _FlakyDriver:
    provider = "fake"
    model_id = "fake-augmentor"

    def __init__(self, target_text: str) -> None:
        self.target_text = target_text
        self.calls = 0

    def synthesize(self, prompt: str) -> ModelDriverResult:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("synthetic provider timeout")
        return ModelDriverResult(
            provider=self.provider,
            model_id=self.model_id,
            raw_text=json.dumps(
                [
                    {
                        "prompt": "Please route this second synthetic chess FEN to Stockfish.",
                        "target_text": self.target_text,
                        "task_kind": "tool_call_json",
                        "target_format": "json",
                        "executable": True,
                        "specialist_id": "stockfish",
                        "task_type": "chess_eval",
                    }
                ],
                sort_keys=True,
            ),
            usage={"input_tokens": 80, "output_tokens": 40},
            metadata={"estimated_cost_usd": 0.01},
        )


class _ExplodingDriver:
    provider = "fake"
    model_id = "fake-augmentor"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, prompt: str) -> ModelDriverResult:
        self.calls += 1
        raise AssertionError("checkpoint resume should not call synthesize")


def _with_updates(record: ToolUseCorpusRecord, **updates: object) -> ToolUseCorpusRecord:
    draft = replace(record, **updates)
    example_id = stable_example_id(draft)
    return replace(draft, example_id=example_id, split=stable_split_for_id(example_id))
