from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.specialists.stockfish import (
    STOCKFISH_SOURCE_ID,
    create_stockfish_trust_anchor,
)
from vecl.substrate.lora_memory import LoRAMemorySubstrate
from vecl.training.synthetic import SOURCE_AUTHORITY, stockfish_tool_use_examples
from vecl.training.tool_use_loop import (
    SupervisedToolUseExample,
    ToolUseTrainer,
    ToolUseTrainingConfig,
    _probe_metrics,
    _report_probe_metrics,
    dataset_hash,
    snapshot_file_hash,
    supervised_loss_for_example,
    validate_training_examples,
)

torch = pytest.importorskip("torch")
pytest.importorskip("peft")


class _TinyMLP(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class _TinyBlock(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = _TinyMLP(hidden_size)


class _TinyBackbone(torch.nn.Module):
    def __init__(self, layer_count: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyBlock(hidden_size) for _ in range(layer_count)])


class _TinyTrainingModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 8, layer_count: int = 3) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.model = _TinyBackbone(layer_count, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = torch.tanh(layer.mlp.up_proj(hidden))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


class _TinyTokenizer:
    def __call__(
        self, text: str, *, add_special_tokens: bool = True, return_tensors: str = "pt"
    ) -> dict[str, torch.Tensor]:
        del return_tensors
        prefix = [1] if add_special_tokens else []
        ids = prefix + [2 + (ord(char) % 50) for char in text[:64]]
        if not ids:
            ids = [1]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


def _trainer(tmp_path: Path) -> tuple[ToolUseTrainer, LoRAMemorySubstrate, ProvenanceLedger]:
    torch.manual_seed(1234)
    model = _TinyTrainingModel()
    substrate = LoRAMemorySubstrate.attach(model, rank=2, last_n_layers=2)
    ledger = ProvenanceLedger()
    monitor = SparseUpdateMonitor(ledger, actor="tool-use-trainer")
    trainer = ToolUseTrainer(
        model=substrate.model,
        processor=_TinyTokenizer(),
        substrate=substrate,
        monitor=monitor,
        output_dir=tmp_path,
        config=ToolUseTrainingConfig(min_score=0.0, learning_rate=0.01),
    )
    return trainer, substrate, ledger


def test_training_example_validation_and_dataset_hash() -> None:
    examples = stockfish_tool_use_examples()
    assert dataset_hash(examples) == dataset_hash(examples)
    validate_training_examples(examples)
    anchor = create_stockfish_trust_anchor("/tmp/stockfish", "Stockfish 18")
    assert all(example.source_id == STOCKFISH_SOURCE_ID for example in examples)
    assert all(example.authority == anchor.trust_value == SOURCE_AUTHORITY for example in examples)
    with pytest.raises(ValueError, match="one tenant"):
        validate_training_examples(
            [
                examples[0],
                SupervisedToolUseExample(
                    example_id="other",
                    tenant_id="other-tenant",
                    source_id="source",
                    authority=0.5,
                    prompt="prompt",
                    target_text="target",
                    task_kind="tool_call_json",
                ),
            ]
        )
    with pytest.raises(ValueError, match="source_id"):
        SupervisedToolUseExample(
            example_id="bad",
            tenant_id="tenant",
            source_id="",
            authority=0.5,
            prompt="prompt",
            target_text="target",
            task_kind="tool_call_json",
        )
    with pytest.raises(ValueError, match="ewc_lambda"):
        ToolUseTrainingConfig(ewc_lambda=-0.1)
    with pytest.raises(ValueError, match="ewc_drift_threshold"):
        ToolUseTrainingConfig(ewc_drift_threshold=-0.1)
    validate_training_examples(
        [
            examples[0],
            SupervisedToolUseExample(
                example_id="low-authority",
                tenant_id=examples[0].tenant_id,
                source_id=examples[0].source_id,
                authority=0.4,
                prompt="prompt",
                target_text="target",
                task_kind="tool_call_json",
            ),
        ]
    )


def test_tool_use_trainer_commits_lora_sparse_update(tmp_path: Path) -> None:
    trainer, substrate, ledger = _trainer(tmp_path)
    examples = stockfish_tool_use_examples()
    before_base = {
        name: parameter.detach().clone()
        for name, parameter in substrate.model.named_parameters()
        if "lora_A" not in name and "lora_B" not in name
    }

    report = trainer.run_cycle(
        examples, probe_sets={"smoke": examples[:1], "heldout": examples[1:2]}
    )

    assert report.committed
    assert report.selected_slots
    assert report.tensor_delta_records
    assert Path(report.before_snapshot_path).exists()
    assert Path(report.after_snapshot_path).exists()
    assert report.before_snapshot_hash != report.after_snapshot_hash
    assert report.gradient_accumulation_steps == 1
    assert report.selection_diagnostics_path is not None
    assert report.selection_diagnostics_hash is not None
    diagnostics = json.loads(Path(report.selection_diagnostics_path).read_text())
    assert diagnostics["slot_count"] == substrate.slot_count
    assert len(diagnostics["slots"]) == substrate.slot_count
    assert {slot["slot_id"] for slot in diagnostics["slots"] if slot["selected"]} == set(
        report.selected_slots
    )
    assert all("score" in slot and "gradient_summary" in slot for slot in diagnostics["slots"])
    assert "aggregate" in report.probe_metrics
    assert "heldout" in report.probe_metrics
    assert "by_domain" in report.probe_metrics["aggregate"]
    assert report.probe_metrics["aggregate"]["sample_count"] == 2
    assert "sign_test_improvement_p_value" in report.probe_metrics["aggregate"]
    assert "paired_t_statistic" in report.probe_metrics["aggregate"]
    applied = ledger.find_by_type(EventType.SPARSE_UPDATE_APPLIED)[0]
    assert applied.payload["substrate"] == "lora"
    assert applied.payload["dataset_hash"] == report.dataset_hash
    assert applied.payload["ewc_lambda"] == 0.0
    assert applied.payload["ewc_penalty_value"] == 0.0
    assert applied.payload["gradient_accumulation_steps"] == 1
    assert applied.payload["selection_diagnostics_hash"] == report.selection_diagnostics_hash
    assert applied.payload["tensor_delta_records"]
    batch_event = ledger.find_by_type(EventType.TOOL_USE_TRAINING_BATCH_PREPARED)[0]
    assert batch_event.payload["trust_anchor_source_ids"] == [STOCKFISH_SOURCE_ID]
    assert batch_event.payload["trust_anchor_kinds"] == ["VERIFIED_OPERATIONAL_RECORD"]
    for name, parameter in substrate.model.named_parameters():
        if name in before_base:
            assert torch.allclose(before_base[name], parameter.detach())


def test_tool_use_trainer_uses_training_prompt_builder(tmp_path: Path) -> None:
    trainer, _substrate, _ledger = _trainer(tmp_path)
    example = stockfish_tool_use_examples()[0]

    trainer.training_prompt_builder = lambda item: f"wrapped::{item.prompt}"

    assert trainer._training_prompt(example).startswith("wrapped::")
    loss = supervised_loss_for_example(
        model=trainer.model,
        processor=trainer.processor,
        example=example,
        torch=torch,
        training_prompt_builder=trainer.training_prompt_builder,
    )
    assert torch.isfinite(loss)


def test_tool_use_trainer_aborts_and_restores_on_lora_apply_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, substrate, ledger = _trainer(tmp_path)
    before = substrate.flatten().copy()

    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced apply failure")

    monkeypatch.setattr(substrate, "apply_update", _raise)
    with pytest.raises(RuntimeError, match="forced apply failure"):
        trainer.run_cycle(stockfish_tool_use_examples())

    assert np.allclose(substrate.flatten(), before)
    aborted = ledger.find_by_type(EventType.LEARNING_EVENT_ABORTED)
    assert aborted
    assert "forced apply failure" in aborted[0].payload["reason"]


def test_report_payload_is_json_serializable(tmp_path: Path) -> None:
    trainer, _substrate, _ledger = _trainer(tmp_path)
    report = trainer.run_cycle(stockfish_tool_use_examples())
    json.dumps(report.to_payload(), sort_keys=True)


def test_tool_use_trainer_restores_baseline_before_training(tmp_path: Path) -> None:
    first, substrate, _ledger = _trainer(tmp_path / "first")
    baseline_path = tmp_path / "tool_use_v0.npz"
    substrate.snapshot(baseline_path)
    baseline_hash = snapshot_file_hash(baseline_path)
    del first

    trainer, restored_substrate, ledger = _trainer(tmp_path / "second")
    trainer.baseline_id = "tool_use_v0"
    trainer.baseline_snapshot_path = baseline_path
    report = trainer.run_cycle(stockfish_tool_use_examples())

    assert report.baseline_id == "tool_use_v0"
    assert report.baseline_restored is True
    assert report.baseline_snapshot_hash == baseline_hash
    assert report.before_snapshot_hash == baseline_hash
    assert report.after_snapshot_hash != baseline_hash
    applied = ledger.find_by_type(EventType.SPARSE_UPDATE_APPLIED)[0]
    assert applied.payload["baseline_id"] == "tool_use_v0"
    assert applied.payload["baseline_snapshot_hash"] == report.baseline_snapshot_hash
    restored_substrate.restore(baseline_path)
    assert np.allclose(restored_substrate.flatten(), substrate.flatten())


def test_tool_use_trainer_reports_ewc_penalty_and_drift(tmp_path: Path) -> None:
    trainer, substrate, ledger = _trainer(tmp_path)
    approved_path = tmp_path / "approved-fisher.npz"
    substrate.snapshot(
        approved_path,
        fisher_diagonal=np.ones(substrate.slot_count, dtype=np.float64),
        fisher_metadata={
            "fisher_status": "accumulated",
            "sample_count": 1,
            "dataset_hash": "test",
            "loss_sum": 1.0,
        },
    )
    first_lora_a = next(
        parameter for name, parameter in substrate.model.named_parameters() if "lora_A" in name
    )
    with torch.no_grad():
        first_lora_a.add_(0.01)
    trainer.config = ToolUseTrainingConfig(
        min_score=0.0,
        learning_rate=0.01,
        ewc_lambda=0.5,
        ewc_drift_threshold=10.0,
    )
    trainer.ewc_approved_snapshot_path = approved_path

    report = trainer.run_cycle(stockfish_tool_use_examples())

    assert report.ewc_penalty_value > 0.0
    assert report.ewc_approved_snapshot_hash == snapshot_file_hash(approved_path)
    assert report.ewc_drift_report is not None
    assert report.ewc_drift_report["drift_within_bound"] is True
    applied = ledger.find_by_type(EventType.SPARSE_UPDATE_APPLIED)[0]
    assert applied.payload["ewc_lambda"] == 0.5
    assert applied.payload["ewc_penalty_value"] == report.ewc_penalty_value
    assert applied.payload["ewc_approved_snapshot_hash"] == report.ewc_approved_snapshot_hash
    assert applied.payload["ewc_drift_report"]["drift_within_bound"] is True
    accepted = ledger.find_by_type(EventType.LEARNING_CANDIDATE_ACCEPTED)
    assert accepted
    assert accepted[0].payload["candidate_snapshot_hash"] == report.candidate_snapshot_hash


def test_tool_use_trainer_rejects_over_bound_candidate_and_restores(tmp_path: Path) -> None:
    trainer, substrate, ledger = _trainer(tmp_path)
    approved_path = tmp_path / "approved-fisher.npz"
    substrate.snapshot(
        approved_path,
        fisher_diagonal=np.ones(substrate.slot_count, dtype=np.float64),
        fisher_metadata={
            "fisher_status": "accumulated",
            "sample_count": 1,
            "dataset_hash": "test",
            "loss_sum": 1.0,
        },
    )
    before = substrate.flatten().copy()
    trainer.config = ToolUseTrainingConfig(
        min_score=0.0,
        learning_rate=0.01,
        ewc_lambda=0.1,
        ewc_drift_threshold=0.0,
    )
    trainer.ewc_approved_snapshot_path = approved_path

    report = trainer.run_cycle(stockfish_tool_use_examples())

    assert report.committed is False
    assert report.candidate_decision == "rejected_drift_exceeded"
    assert report.ewc_drift_report is not None
    assert report.ewc_drift_report["drift_within_bound"] is False
    assert report.restored_snapshot_hash == report.start_snapshot_hash
    assert Path(report.after_snapshot_path).exists()
    assert report.after_snapshot_hash != report.before_snapshot_hash
    assert np.allclose(substrate.flatten(), before)
    assert not ledger.find_by_type(EventType.SPARSE_UPDATE_APPLIED)
    assert not ledger.find_by_type(EventType.LEARNING_EVENT_COMMITTED)
    assert ledger.find_by_type(EventType.LEARNING_EVENT_ABORTED)
    assert ledger.find_by_type(EventType.LEARNING_CANDIDATE_PREPARED)
    assert ledger.find_by_type(EventType.LEARNING_CANDIDATE_EVALUATED)
    rejected = ledger.find_by_type(EventType.LEARNING_CANDIDATE_REJECTED)
    assert rejected
    assert rejected[0].payload["candidate_quarantined"] is True
    restored = ledger.find_by_type(EventType.LEARNING_CANDIDATE_RESTORED)
    assert restored
    assert restored[0].payload["restored_snapshot_hash"] == report.start_snapshot_hash


def test_ewc_approved_snapshot_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    trainer, substrate, _ledger = _trainer(tmp_path)
    approved_path = tmp_path / "approved-fisher.npz"
    substrate.snapshot(
        approved_path,
        fisher_diagonal=np.ones(substrate.slot_count, dtype=np.float64),
        fisher_metadata={
            "fisher_status": "accumulated",
            "sample_count": 1,
            "dataset_hash": "test",
            "loss_sum": 1.0,
        },
    )
    trainer.config = ToolUseTrainingConfig(ewc_lambda=0.1)
    trainer.ewc_approved_snapshot_path = approved_path
    trainer.expected_ewc_approved_snapshot_hash = "not-the-right-hash"

    with pytest.raises(ValueError, match="approved snapshot hash"):
        trainer.run_cycle(stockfish_tool_use_examples())


def test_ewc_penalty_requires_approved_snapshot(tmp_path: Path) -> None:
    trainer, _substrate, _ledger = _trainer(tmp_path)
    trainer.config = ToolUseTrainingConfig(ewc_lambda=0.1)

    with pytest.raises(ValueError, match="approved LoRA snapshot"):
        trainer.run_cycle(stockfish_tool_use_examples())


def test_tool_use_config_rejects_invalid_gradient_accumulation_steps() -> None:
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        ToolUseTrainingConfig(gradient_accumulation_steps=0)


def test_probe_metrics_include_sign_test_and_paired_delta_stats() -> None:
    metrics = _probe_metrics([1.0, 2.0, 3.0, 4.0], [0.9, 1.9, 3.1, 4.2])

    assert metrics["sample_count"] == 4
    assert metrics["improved_count"] == 2
    assert metrics["worsened_count"] == 2
    assert metrics["mean_delta"] == pytest.approx(0.025)
    assert metrics["sign_test_improvement_p_value"] == pytest.approx(0.6875)
    assert metrics["paired_t_degrees_of_freedom"] == 3
    assert metrics["paired_t_normal_approx_p_value"] is not None


def test_report_probe_metrics_breaks_down_by_domain_category_and_task_kind() -> None:
    examples = [
        SupervisedToolUseExample(
            example_id="stockfish-1",
            tenant_id="tenant",
            source_id="source",
            authority=0.9,
            prompt="p",
            target_text="t",
            task_kind="tool_call_json",
            metadata={"corpus_domain": "stockfish", "corpus_category": "chess"},
        ),
        SupervisedToolUseExample(
            example_id="sympy-1",
            tenant_id="tenant",
            source_id="source",
            authority=0.95,
            prompt="p",
            target_text="t",
            task_kind="final_answer",
            metadata={"corpus_domain": "sympy", "corpus_category": "math"},
        ),
    ]

    metrics = _report_probe_metrics(
        {"heldout": [2.0, 4.0]},
        {"heldout": [1.5, 4.4]},
        examples_by_name={"heldout": examples},
        return_named=True,
    )

    assert metrics["heldout"]["by_domain"]["stockfish"]["mean_delta"] == pytest.approx(-0.5)
    assert metrics["heldout"]["by_domain"]["sympy"]["mean_delta"] == pytest.approx(0.4)
    assert metrics["aggregate"]["by_category"]["chess"]["improved_count"] == 1
    assert metrics["aggregate"]["by_task_kind"]["final_answer"]["worsened_count"] == 1
