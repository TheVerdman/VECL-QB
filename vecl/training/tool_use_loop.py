from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vecl._compat import UTC
from vecl.evaluation.fisher import compute_lora_snapshot_drift
from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.runtime.tokens import create_learning_event_token
from vecl.sparse.types import SparseMemoryInputs
from vecl.substrate.lora_memory import LoRAMemorySubstrate
from vecl.training.scoring import (
    activation_from_gradient_summaries,
    rarity_from_selection_counts,
    slot_selection_counts_from_ledger,
)

TaskKind = Literal["tool_call_json", "final_answer"]


@dataclass(frozen=True)
class SupervisedToolUseExample:
    example_id: str
    tenant_id: str
    source_id: str
    authority: float
    prompt: str
    target_text: str
    task_kind: TaskKind
    artifact_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id or not self.tenant_id or not self.source_id:
            raise ValueError("example_id, tenant_id, and source_id must be non-empty")
        if not self.prompt or not self.target_text:
            raise ValueError("prompt and target_text must be non-empty")
        if self.task_kind not in {"tool_call_json", "final_answer"}:
            raise ValueError("task_kind must be tool_call_json or final_answer")
        if not np.isfinite(self.authority) or self.authority < 0:
            raise ValueError("authority must be finite and non-negative")
        object.__setattr__(self, "artifact_ids", list(self.artifact_ids))
        object.__setattr__(self, "claim_ids", list(self.claim_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["authority"] = float(self.authority)
        return payload


@dataclass(frozen=True)
class ToolUseTrainingConfig:
    learning_rate: float = 1e-4
    min_authority: float = 0.1
    min_score: float = 1e-9
    max_slots: int = 2
    gradient_accumulation_steps: int = 1
    policy_version: str = "tool-use-training-v0"
    trust_policy_version: str = "trust-v0"
    ewc_lambda: float = 0.0
    ewc_drift_threshold: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.ewc_lambda) or self.ewc_lambda < 0:
            raise ValueError("ewc_lambda must be finite and non-negative")
        if self.ewc_drift_threshold is not None and (
            not np.isfinite(self.ewc_drift_threshold) or self.ewc_drift_threshold < 0
        ):
            raise ValueError("ewc_drift_threshold must be finite and non-negative")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be >= 0")
        if self.min_authority < 0 or self.min_score < 0:
            raise ValueError("min_authority and min_score must be >= 0")
        if self.max_slots < 0:
            raise ValueError("max_slots must be >= 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if not self.policy_version or not self.trust_policy_version:
            raise ValueError("policy versions must be non-empty")


@dataclass(frozen=True)
class ToolUseTrainingReport:
    batch_id: str
    dataset_hash: str
    loss_before_update: float
    selected_slots: list[int]
    eligible_slots: list[int]
    tensor_delta_records: list[dict[str, Any]]
    before_snapshot_path: str
    before_snapshot_hash: str
    after_snapshot_path: str
    after_snapshot_hash: str
    learning_event_id: str
    committed: bool
    task_loss_before_update: float | None = None
    total_loss_before_update: float | None = None
    ewc_penalty_value: float = 0.0
    ewc_approved_snapshot_path: str | None = None
    ewc_approved_snapshot_hash: str | None = None
    ewc_drift_report: dict[str, Any] | None = None
    baseline_id: str | None = None
    baseline_snapshot_path: str | None = None
    baseline_snapshot_hash: str | None = None
    baseline_restored: bool = False
    gradient_accumulation_steps: int = 1
    selection_diagnostics_path: str | None = None
    selection_diagnostics_hash: str | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)
    probe_metrics: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class ToolUseTrainer:
    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        substrate: LoRAMemorySubstrate,
        monitor: SparseUpdateMonitor,
        output_dir: str | Path,
        config: ToolUseTrainingConfig | None = None,
        actor: str = "tool-use-trainer",
        baseline_id: str | None = None,
        baseline_snapshot_path: str | Path | None = None,
        expected_baseline_snapshot_hash: str | None = None,
        ewc_approved_snapshot_path: str | Path | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.substrate = substrate
        self.monitor = monitor
        self.output_dir = Path(output_dir)
        self.config = config or ToolUseTrainingConfig()
        self.actor = actor
        self.baseline_id = baseline_id
        self.baseline_snapshot_path = (
            Path(baseline_snapshot_path) if baseline_snapshot_path is not None else None
        )
        self.expected_baseline_snapshot_hash = expected_baseline_snapshot_hash
        self.ewc_approved_snapshot_path = (
            Path(ewc_approved_snapshot_path) if ewc_approved_snapshot_path is not None else None
        )
        self.run_metadata = dict(run_metadata or {})

    def run_cycle(
        self,
        train_examples: list[SupervisedToolUseExample],
        probe_examples: list[SupervisedToolUseExample] | None = None,
        probe_sets: Mapping[str, list[SupervisedToolUseExample]] | None = None,
    ) -> ToolUseTrainingReport:
        validate_training_examples(train_examples)
        named_probe_sets = dict(probe_sets or {})
        if probe_examples:
            named_probe_sets.setdefault("probe", probe_examples)
        if probe_examples:
            validate_training_examples(
                probe_examples, expected_tenant_id=train_examples[0].tenant_id
            )
        for examples in named_probe_sets.values():
            if examples:
                validate_training_examples(examples, expected_tenant_id=train_examples[0].tenant_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data_hash = dataset_hash(train_examples)
        batch_id = f"tool-use:{data_hash[:16]}"
        cycle_dir = self.output_dir / batch_id.replace(":", "-")
        cycle_dir.mkdir(parents=True, exist_ok=True)
        tenant_id = train_examples[0].tenant_id
        batch_authority = min(float(example.authority) for example in train_examples)

        batch_event = self._append_batch_event(batch_id, data_hash, train_examples)
        before_path = cycle_dir / "before-lora.npz"
        after_path = cycle_dir / "after-lora.npz"
        baseline_hash = self._restore_baseline_if_configured()
        ewc_snapshot_path = self._ewc_snapshot_path()
        ewc_snapshot_hash = snapshot_file_hash(ewc_snapshot_path) if ewc_snapshot_path else None
        self.substrate.snapshot(before_path)
        before_hash = snapshot_file_hash(before_path)
        token = None
        mutated = False
        try:
            probe_before = self._losses_for_probe_sets(named_probe_sets)
            task_loss, ewc_penalty, total_loss = self._backward_loss(train_examples)
            raw_gradients = self.substrate.raw_gradients_from_current_grads()
            gradient_summaries = self.substrate.gradient_summaries(raw_gradients)
            inputs = self._sparse_inputs(tenant_id, gradient_summaries, batch_authority)
            token = create_learning_event_token(
                batch_id=batch_id,
                tenant_id=tenant_id,
                source_set_hash=stable_hash(
                    sorted({example.source_id for example in train_examples})
                ),
                provenance_root_hash=stable_hash(
                    {"training_batch_event_id": batch_event.event_id, "dataset_hash": data_hash}
                ),
                min_authority=self.config.min_authority,
                min_score=self.config.min_score,
                max_slots=inputs.max_slots,
                policy_version=self.config.policy_version,
                trust_policy_version=self.config.trust_policy_version,
                now=datetime.now(UTC),
            )
            result = self.monitor.apply_sparse_update(token, inputs)
            self.monitor.verify_sparse_update(token, inputs, result)
            diagnostics_path = cycle_dir / "selection-diagnostics.json"
            selection_diagnostics = _selection_diagnostics(
                inputs=inputs,
                gradient_summaries=gradient_summaries,
                scores=result.scores,
                result_selected_slots=result.selected_slots,
                slot_metadata=[asdict(metadata) for metadata in self.substrate.slot_metadata],
                batch_id=batch_id,
                dataset_hash=data_hash,
            )
            diagnostics_path.write_text(
                json.dumps(selection_diagnostics, indent=2, sort_keys=True), encoding="utf-8"
            )
            diagnostics_hash = snapshot_file_hash(diagnostics_path)
            tensor_records = self.substrate.apply_update(
                result, raw_gradients, learning_rate=self.config.learning_rate
            )
            mutated = True
            self.substrate.snapshot(after_path)
            after_hash = snapshot_file_hash(after_path)
            drift_report = self._ewc_drift_report(after_path)
            extra_payload = {
                "substrate": "lora",
                "tensor_delta_records": [asdict(record) for record in tensor_records],
                "before_snapshot_hash": before_hash,
                "after_snapshot_hash": after_hash,
                "dataset_hash": data_hash,
                "loss_value": float(total_loss),
                "task_loss_value": float(task_loss),
                "ewc_penalty_value": float(ewc_penalty),
                "activation_policy": "gradient_summary_normalized_v0",
                "rarity_policy": "inverse_sqrt_selection_count_v0",
                "authority_policy": "batch_min_authority_v0",
                "ewc_lambda": self.config.ewc_lambda,
                "ewc_approved_snapshot_hash": ewc_snapshot_hash,
                "ewc_drift_threshold": self.config.ewc_drift_threshold,
                "ewc_drift_report": drift_report,
                "baseline_id": self.baseline_id,
                "baseline_snapshot_hash": baseline_hash,
                "baseline_restored": self.baseline_snapshot_path is not None,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "selection_diagnostics_hash": diagnostics_hash,
                **self.run_metadata,
            }
            self.monitor.commit_learning_event(token, extra_payload=extra_payload)
            probe_after = self._losses_for_probe_sets(named_probe_sets)
            return ToolUseTrainingReport(
                batch_id=batch_id,
                dataset_hash=data_hash,
                loss_before_update=float(task_loss),
                selected_slots=list(result.selected_slots),
                eligible_slots=list(result.eligible_slots),
                tensor_delta_records=[asdict(record) for record in tensor_records],
                before_snapshot_path=str(before_path),
                before_snapshot_hash=before_hash,
                after_snapshot_path=str(after_path),
                after_snapshot_hash=after_hash,
                learning_event_id=token.event_id,
                committed=True,
                task_loss_before_update=float(task_loss),
                total_loss_before_update=float(total_loss),
                ewc_penalty_value=float(ewc_penalty),
                ewc_approved_snapshot_path=str(ewc_snapshot_path) if ewc_snapshot_path else None,
                ewc_approved_snapshot_hash=ewc_snapshot_hash,
                ewc_drift_report=drift_report,
                baseline_id=self.baseline_id,
                baseline_snapshot_path=(
                    str(self.baseline_snapshot_path) if self.baseline_snapshot_path else None
                ),
                baseline_snapshot_hash=baseline_hash,
                baseline_restored=self.baseline_snapshot_path is not None,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                selection_diagnostics_path=str(diagnostics_path),
                selection_diagnostics_hash=diagnostics_hash,
                run_metadata=dict(self.run_metadata),
                probe_metrics=_report_probe_metrics(
                    probe_before,
                    probe_after,
                    examples_by_name=named_probe_sets,
                    return_named=probe_sets is not None,
                ),
            )
        except Exception as exc:
            if mutated:
                self.substrate.restore(before_path)
            if token is not None:
                self.monitor.abort_learning_event(
                    token,
                    str(exc),
                    payload={
                        "dataset_hash": data_hash,
                        "before_snapshot_hash": before_hash,
                        "baseline_id": self.baseline_id,
                        "baseline_snapshot_hash": baseline_hash,
                        "ewc_approved_snapshot_hash": ewc_snapshot_hash,
                        **self.run_metadata,
                    },
                )
            raise

    def _restore_baseline_if_configured(self) -> str | None:
        if self.baseline_snapshot_path is None:
            return None
        if not self.baseline_snapshot_path.exists():
            raise ValueError(f"baseline snapshot not found: {self.baseline_snapshot_path}")
        baseline_hash = snapshot_file_hash(self.baseline_snapshot_path)
        if (
            self.expected_baseline_snapshot_hash
            and baseline_hash != self.expected_baseline_snapshot_hash
        ):
            raise ValueError("baseline snapshot hash does not match expected hash")
        self.substrate.restore(self.baseline_snapshot_path)
        return baseline_hash

    def _ewc_snapshot_path(self) -> Path | None:
        snapshot_path = self.ewc_approved_snapshot_path or (
            self.baseline_snapshot_path if self.config.ewc_lambda > 0 else None
        )
        if self.config.ewc_lambda > 0 and snapshot_path is None:
            raise ValueError("ewc_lambda > 0 requires an approved LoRA snapshot")
        if snapshot_path is not None and not snapshot_path.exists():
            raise ValueError(f"EWC approved snapshot not found: {snapshot_path}")
        return snapshot_path

    def _append_batch_event(
        self, batch_id: str, data_hash: str, examples: list[SupervisedToolUseExample]
    ) -> ProvenanceEvent:
        authorities = [float(example.authority) for example in examples]
        trust_anchor_sources = sorted(
            {
                str(example.metadata["trust_anchor_source_id"])
                for example in examples
                if "trust_anchor_source_id" in example.metadata
            }
        )
        trust_anchor_kinds = sorted(
            {
                str(example.metadata["trust_anchor_kind"])
                for example in examples
                if "trust_anchor_kind" in example.metadata
            }
        )
        return self.monitor.ledger.append(
            ProvenanceEvent(
                event_type=EventType.TOOL_USE_TRAINING_BATCH_PREPARED,
                tenant_id=examples[0].tenant_id,
                actor=self.actor,
                payload={
                    "batch_id": batch_id,
                    "dataset_hash": data_hash,
                    "example_ids": [example.example_id for example in examples],
                    "source_set_hash": stable_hash(
                        sorted({example.source_id for example in examples})
                    ),
                    "artifact_ids": sorted(
                        {item for example in examples for item in example.artifact_ids}
                    ),
                    "claim_ids": sorted(
                        {item for example in examples for item in example.claim_ids}
                    ),
                    "task_kinds": sorted({example.task_kind for example in examples}),
                    "trust_anchor_source_ids": trust_anchor_sources,
                    "trust_anchor_kinds": trust_anchor_kinds,
                    "authority_summary": {
                        "min": min(authorities),
                        "max": max(authorities),
                        "mean": float(np.mean(authorities)),
                    },
                },
            )
        )

    def _sparse_inputs(
        self, tenant_id: str, gradient_summaries: np.ndarray, batch_authority: float
    ) -> SparseMemoryInputs:
        if gradient_summaries.shape != (self.substrate.slot_count,):
            raise ValueError("gradient_summaries length must match substrate slot_count")
        counts = slot_selection_counts_from_ledger(
            self.monitor.ledger, slot_count=self.substrate.slot_count, tenant_id=tenant_id
        )
        authority = np.full(self.substrate.slot_count, batch_authority, dtype=np.float64)
        return SparseMemoryInputs(
            memory_values=self.substrate.flatten(),
            gradients=gradient_summaries,
            activation=activation_from_gradient_summaries(gradient_summaries),
            rarity=rarity_from_selection_counts(counts),
            authority=authority,
            learning_rate=self.config.learning_rate,
            min_authority=self.config.min_authority,
            min_score=self.config.min_score,
            max_slots=min(self.config.max_slots, self.substrate.slot_count),
            quarantined_slots=set(),
        )

    def _backward_loss(
        self, examples: list[SupervisedToolUseExample]
    ) -> tuple[float, float, float]:
        torch = _import_torch()
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        total = 0.0
        scale = 1.0 / len(examples)
        step_count = min(self.config.gradient_accumulation_steps, len(examples))
        for microbatch in np.array_split(np.array(examples, dtype=object), step_count):
            for example in microbatch.tolist():
                loss = _supervised_loss(
                    model=self.model,
                    processor=self.processor,
                    prompt=example.prompt,
                    target=example.target_text,
                    torch=torch,
                )
                total += float(loss.detach().cpu().item())
                (loss * scale).backward()
        task_loss = total * scale
        ewc_penalty = 0.0
        if self.config.ewc_lambda > 0:
            snapshot_path = self._ewc_snapshot_path()
            if snapshot_path is None:  # pragma: no cover - _ewc_snapshot_path already checks.
                raise ValueError("ewc_lambda > 0 requires an approved LoRA snapshot")
            penalty = self.substrate.fisher_weighted_drift_penalty(snapshot_path)
            ewc_penalty = float(penalty.detach().cpu().item())
            (penalty * float(self.config.ewc_lambda)).backward()
        total_loss = task_loss + float(self.config.ewc_lambda) * ewc_penalty
        return task_loss, ewc_penalty, total_loss

    def _ewc_drift_report(self, candidate_snapshot: Path) -> dict[str, Any] | None:
        snapshot_path = self._ewc_snapshot_path()
        if snapshot_path is None or self.config.ewc_drift_threshold is None:
            return None
        report = compute_lora_snapshot_drift(
            snapshot_path,
            candidate_snapshot,
            drift_threshold=self.config.ewc_drift_threshold,
        )
        return report.to_payload()

    def _losses_for_examples(self, examples: list[SupervisedToolUseExample]) -> list[float]:
        if not examples:
            return []
        torch = _import_torch()
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for example in examples:
                loss = _supervised_loss(
                    model=self.model,
                    processor=self.processor,
                    prompt=example.prompt,
                    target=example.target_text,
                    torch=torch,
                )
                losses.append(float(loss.detach().cpu().item()))
        return losses

    def _losses_for_probe_sets(
        self, probe_sets: Mapping[str, list[SupervisedToolUseExample]]
    ) -> dict[str, list[float]]:
        return {
            name: self._losses_for_examples(examples)
            for name, examples in sorted(probe_sets.items())
            if examples
        }


def validate_training_examples(
    examples: list[SupervisedToolUseExample], *, expected_tenant_id: str | None = None
) -> None:
    if not examples:
        raise ValueError("at least one training example is required")
    tenant_id = expected_tenant_id or examples[0].tenant_id
    for example in examples:
        if example.tenant_id != tenant_id:
            raise ValueError("training examples must belong to one tenant")
        if not example.source_id:
            raise ValueError("training example source_id must be non-empty")
        if not np.isfinite(example.authority):
            raise ValueError("training example authority must be finite")


def dataset_hash(examples: list[SupervisedToolUseExample]) -> str:
    validate_training_examples(examples)
    return stable_hash([example.to_payload() for example in examples])


def snapshot_file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_diagnostics(
    *,
    inputs: SparseMemoryInputs,
    gradient_summaries: np.ndarray,
    scores: np.ndarray,
    result_selected_slots: list[int],
    slot_metadata: list[dict[str, Any]],
    batch_id: str,
    dataset_hash: str,
) -> dict[str, Any]:
    if gradient_summaries.shape != inputs.memory_values.shape:
        raise ValueError("gradient_summaries length must match sparse inputs")
    if len(slot_metadata) != len(inputs.memory_values):
        raise ValueError("slot_metadata length must match sparse inputs")
    selected = set(result_selected_slots)
    if scores.shape != inputs.memory_values.shape:
        raise ValueError("scores length must match sparse inputs")
    ranked = sorted(
        ((float(scores[index]), int(index)) for index in range(len(scores))),
        key=lambda item: (-item[0], item[1]),
    )
    ranks = {slot_id: rank for rank, (_score, slot_id) in enumerate(ranked, start=1)}
    selected_scores = [float(scores[slot_id]) for slot_id in result_selected_slots]
    unselected_scores = [
        float(scores[index]) for index in range(len(scores)) if index not in selected
    ]
    cutoff_score = min(selected_scores) if selected_scores else None
    next_score = max(unselected_scores) if unselected_scores else None
    return {
        "batch_id": batch_id,
        "dataset_hash": dataset_hash,
        "slot_count": int(len(inputs.memory_values)),
        "max_slots": int(inputs.max_slots),
        "selected_slots": list(result_selected_slots),
        "cutoff_score": cutoff_score,
        "next_unselected_score": next_score,
        "cutoff_margin": (
            float(cutoff_score - next_score)
            if cutoff_score is not None and next_score is not None
            else None
        ),
        "slots": [
            {
                **slot_metadata[index],
                "gradient_summary": float(gradient_summaries[index]),
                "activation": float(inputs.activation[index]),
                "rarity": float(inputs.rarity[index]),
                "authority": float(inputs.authority[index]),
                "score": float(scores[index]),
                "selected": index in selected,
                "rank": ranks[index],
            }
            for index in range(len(inputs.memory_values))
        ],
    }


def _supervised_loss(*, model: Any, processor: Any, prompt: str, target: str, torch: Any) -> Any:
    prompt_inputs = _prompt_inputs(processor, prompt, torch)
    tokenizer = getattr(processor, "tokenizer", processor)
    target_inputs = tokenizer(target, add_special_tokens=False, return_tensors="pt")
    target_ids = target_inputs["input_ids"]
    if target_ids.shape[-1] == 0:
        raise ValueError("target tokenization produced no tokens")
    input_ids = torch.cat([prompt_inputs["input_ids"], target_ids], dim=-1)
    attention_mask = torch.cat(
        [
            prompt_inputs.get("attention_mask", torch.ones_like(prompt_inputs["input_ids"])),
            torch.ones_like(target_ids),
        ],
        dim=-1,
    )
    labels = input_ids.clone()
    labels[:, : prompt_inputs["input_ids"].shape[-1]] = -100
    device = _model_input_device(model, torch)
    output = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=labels.to(device),
    )
    return output.loss


def _prompt_inputs(processor: Any, prompt: str, torch: Any) -> dict[str, Any]:
    if hasattr(processor, "apply_chat_template"):
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return dict(inputs)
    inputs = processor(prompt, add_special_tokens=True, return_tensors="pt")
    result = dict(inputs)
    result.setdefault("attention_mask", torch.ones_like(result["input_ids"]))
    return result


def _model_input_device(model: Any, torch: Any) -> Any:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _probe_metrics(before: list[float], after: list[float]) -> dict[str, Any]:
    if not before and not after:
        return {}
    if len(before) != len(after):
        raise ValueError("probe before/after losses must have the same length")
    deltas = [
        after_value - before_value for before_value, after_value in zip(before, after, strict=True)
    ]
    improved = [
        after_value < before_value for before_value, after_value in zip(before, after, strict=True)
    ]
    worsened = [
        after_value > before_value for before_value, after_value in zip(before, after, strict=True)
    ]
    mean_before = float(np.mean(before)) if before else None
    mean_after = float(np.mean(after)) if after else None
    mean_delta = float(np.mean(deltas)) if deltas else None
    median_delta = float(np.median(deltas)) if deltas else None
    return {
        "sample_count": len(before),
        "before_losses": before,
        "after_losses": after,
        "loss_deltas": deltas,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "relative_mean_change": _relative_mean_change(mean_before, mean_after),
        "improved_count": int(sum(improved)),
        "worsened_count": int(sum(worsened)),
        "tied_count": int(len(before) - sum(improved) - sum(worsened)),
        "any_improved": bool(any(improved)),
        "sign_test_improvement_p_value": _sign_test_p_value(
            successes=int(sum(improved)), failures=int(sum(worsened))
        ),
        "sign_test_degradation_p_value": _sign_test_p_value(
            successes=int(sum(worsened)), failures=int(sum(improved))
        ),
        "paired_t_statistic": _paired_t_statistic(deltas),
        "paired_t_degrees_of_freedom": len(deltas) - 1 if len(deltas) > 1 else 0,
        "paired_t_normal_approx_p_value": _paired_t_normal_approx_p_value(deltas),
    }


def _report_probe_metrics(
    before_by_name: Mapping[str, list[float]],
    after_by_name: Mapping[str, list[float]],
    *,
    examples_by_name: Mapping[str, list[SupervisedToolUseExample]] | None = None,
    return_named: bool,
) -> dict[str, Any]:
    if not before_by_name and not after_by_name:
        return {}
    if not return_named and len(before_by_name) <= 1:
        name = next(iter(before_by_name), "")
        metrics = _probe_metrics(before_by_name.get(name, []), after_by_name.get(name, []))
        if examples_by_name is not None:
            metrics.update(
                _probe_breakdowns(
                    examples_by_name.get(name, []),
                    before_by_name.get(name, []),
                    after_by_name.get(name, []),
                )
            )
        return metrics
    metrics = {
        name: _probe_metrics(before_by_name.get(name, []), after_by_name.get(name, []))
        for name in sorted(set(before_by_name) | set(after_by_name))
    }
    if examples_by_name is not None:
        for name, values in metrics.items():
            values.update(
                _probe_breakdowns(
                    examples_by_name.get(name, []),
                    before_by_name.get(name, []),
                    after_by_name.get(name, []),
                )
            )
    aggregate_before = [value for name in sorted(before_by_name) for value in before_by_name[name]]
    aggregate_after = [value for name in sorted(after_by_name) for value in after_by_name[name]]
    metrics["aggregate"] = _probe_metrics(aggregate_before, aggregate_after)
    if examples_by_name is not None:
        aggregate_examples = [
            example for name in sorted(before_by_name) for example in examples_by_name.get(name, [])
        ]
        metrics["aggregate"].update(
            _probe_breakdowns(aggregate_examples, aggregate_before, aggregate_after)
        )
    return metrics


def _probe_breakdowns(
    examples: list[SupervisedToolUseExample], before: list[float], after: list[float]
) -> dict[str, Any]:
    if not examples:
        return {}
    if len(examples) != len(before) or len(before) != len(after):
        raise ValueError("probe examples and losses must have the same length")
    return {
        "by_domain": _group_probe_metrics(examples, before, after, "corpus_domain"),
        "by_category": _group_probe_metrics(examples, before, after, "corpus_category"),
        "by_task_kind": _group_probe_metrics(examples, before, after, "task_kind"),
    }


def _group_probe_metrics(
    examples: list[SupervisedToolUseExample],
    before: list[float],
    after: list[float],
    key: str,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for example, before_value, after_value in zip(examples, before, after, strict=True):
        label: str
        if key == "task_kind":
            label = example.task_kind
        else:
            label = str(example.metadata.get(key) or "unknown")
        bucket = grouped.setdefault(label, {"before": [], "after": []})
        bucket["before"].append(before_value)
        bucket["after"].append(after_value)
    return {
        label: _probe_metrics(values["before"], values["after"])
        for label, values in sorted(grouped.items())
    }


def _relative_mean_change(mean_before: float | None, mean_after: float | None) -> float | None:
    if mean_before is None or mean_after is None or mean_before == 0.0:
        return None
    return float((mean_after - mean_before) / abs(mean_before))


def _sign_test_p_value(*, successes: int, failures: int) -> float | None:
    """One-sided exact sign-test p-value under p=0.5.

    Ties are excluded before this function is called. A small p-value for
    improvement means "at least this many improvements by chance" is unlikely;
    the degradation variant is computed by swapping successes/failures.
    """

    trials = successes + failures
    if trials == 0:
        return None
    if successes <= 0:
        return 1.0
    tail = 0.0
    for count in range(successes, trials + 1):
        tail += math.comb(trials, count)
    return float(tail / (2**trials))


def _paired_t_statistic(deltas: list[float]) -> float | None:
    if len(deltas) < 2:
        return None
    std = float(np.std(deltas, ddof=1))
    if std == 0.0:
        return None
    return float(np.mean(deltas) / (std / math.sqrt(len(deltas))))


def _paired_t_normal_approx_p_value(deltas: list[float]) -> float | None:
    statistic = _paired_t_statistic(deltas)
    if statistic is None:
        return None
    return float(math.erfc(abs(statistic) / math.sqrt(2.0)))


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "ToolUseTrainer requires the substrate extra with torch installed"
        ) from exc
    return torch
