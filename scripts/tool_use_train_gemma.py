#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from vecl._gemma import has_allowed_cuda_device, resolve_gemma_model_class, resolve_torch_dtype
from vecl.provenance.events import EventType, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import SpecialistRequest
from vecl.qb.specialist_contracts import SPECIALIST_PAYLOAD_CONTRACTS
from vecl.qb.tool_call import (
    ToolCallParseError,
    ToolCallValidationConfig,
    ToolCallValidationError,
    build_tool_call_prompt,
    parse_tool_call_response,
    validate_tool_call,
)
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.substrate.lora_memory import LoRAMemorySubstrate
from vecl.training.synthetic import stockfish_final_answer_examples, stockfish_tool_use_examples
from vecl.training.tool_use_loop import (
    SupervisedToolUseExample,
    ToolUseTrainer,
    ToolUseTrainingConfig,
    TrainingPromptBuilder,
    snapshot_file_hash,
    supervised_loss_for_example,
)

DEFAULT_TRAIN_MODEL_ID = "google/gemma-4-31B-it"
DEFAULT_ALLOWED_CUDA_DEVICES = "A100,H100"
DEFAULT_EXPANDED_CORPUS_DOMAINS = frozenset(
    {"blast", "cross", "eda", "stockfish", "sympy", "terraform", "timesfm"}
)


def main() -> int:
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        print(f"Missing optional substrate dependency: {exc}", flush=True)
        return 2

    allowed_devices = os.environ.get("GEMMA_ALLOWED_CUDA_DEVICES", DEFAULT_ALLOWED_CUDA_DEVICES)
    require_cuda = os.environ.get("GEMMA_REQUIRE_CUDA", "true").lower() not in {"0", "false", "no"}
    if require_cuda and not has_allowed_cuda_device(torch, allowed_devices):
        print("No approved CUDA device detected for tool-use training eval.", flush=True)
        return 2

    model_cls = resolve_gemma_model_class()
    if model_cls is None:
        print("No supported Gemma 4 model class is available in Transformers.", flush=True)
        return 2

    torch_seed = _int_env("VECL_TRAIN_TORCH_SEED", 1234)
    torch.manual_seed(torch_seed)
    model_id = os.environ.get("VECL_TRAIN_MODEL_ID", DEFAULT_TRAIN_MODEL_ID)
    token = os.environ.get("HF_TOKEN")
    dtype = resolve_torch_dtype(torch, os.environ.get("GEMMA_DTYPE", "bfloat16"))
    device_map = os.environ.get("GEMMA_DEVICE_MAP", "auto")
    rank = int(os.environ.get("VECL_TRAIN_LORA_RANK", "2"))
    last_n_layers = int(os.environ.get("VECL_TRAIN_LAST_N_LAYERS", "2"))
    max_slots = int(os.environ.get("VECL_TRAIN_MAX_SLOTS", "2"))
    learning_rate = float(os.environ.get("VECL_TRAIN_LEARNING_RATE", "0.001"))
    gradient_accumulation_steps = _int_env("VECL_TRAIN_GRADIENT_ACCUMULATION_STEPS", 1)
    ewc_lambda = float(os.environ.get("VECL_TRAIN_EWC_LAMBDA", "0.0"))
    ewc_drift_threshold = _optional_float_env("VECL_TRAIN_EWC_DRIFT_THRESHOLD")
    expect_rejection = _bool_env("VECL_TRAIN_EXPECT_REJECTION", False)
    require_inline_drift = _bool_env("VECL_TRAIN_REQUIRE_INLINE_DRIFT", False)
    interference_threshold = float(os.environ.get("VECL_TRAIN_INTERFERENCE_THRESHOLD", "0.05"))
    prompt_mode, training_prompt_builder = _training_prompt_builder()
    diagnostic_mode = os.environ.get("VECL_TRAIN_DIAGNOSTIC_MODE", "").strip()
    if diagnostic_mode and diagnostic_mode not in {"tiny_overfit", "sparse_overfit"}:
        raise ValueError(
            "VECL_TRAIN_DIAGNOSTIC_MODE must be empty, tiny_overfit, or sparse_overfit"
        )
    output_dir = Path(os.environ.get("VECL_TRAIN_OUTPUT_DIR", "/tmp/vecl-tool-use-train"))
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model_kwargs: dict[str, Any] = {"device_map": device_map, "token": token}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = model_cls.from_pretrained(model_id, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    substrate = LoRAMemorySubstrate.attach(model, rank=rank, last_n_layers=last_n_layers)
    base_weights_unchanged = _base_weights_frozen(substrate.model)
    baseline = _prepare_baseline_snapshot(
        substrate,
        output_dir,
        model_id=model_id,
        rank=rank,
        last_n_layers=last_n_layers,
        torch_seed=torch_seed,
    )
    approved_snapshot = _prepare_ewc_approved_snapshot(
        output_dir,
        fallback_path=baseline["baseline_snapshot_path"] if ewc_lambda > 0 else None,
        fallback_hash=baseline["baseline_snapshot_hash"] if ewc_lambda > 0 else None,
    )
    if require_inline_drift:
        if not approved_snapshot["approved_snapshot_path"]:
            raise ValueError(
                "VECL_TRAIN_REQUIRE_INLINE_DRIFT=1 requires "
                "VECL_TRAIN_EWC_APPROVED_SNAPSHOT_URI or "
                "VECL_TRAIN_EWC_APPROVED_SNAPSHOT_PATH"
            )
        if ewc_drift_threshold is None:
            raise ValueError(
                "VECL_TRAIN_REQUIRE_INLINE_DRIFT=1 requires VECL_TRAIN_EWC_DRIFT_THRESHOLD"
            )

    ledger = ProvenanceLedger()
    monitor = SparseUpdateMonitor(ledger, actor="tool-use-trainer")
    train_examples, probe_sets, training_source = _training_examples(output_dir)
    sample_seed = training_source.get("sample_seed")
    if diagnostic_mode == "tiny_overfit":
        overfit_summary = _tiny_overfit_diagnostic(
            model=substrate.model,
            processor=processor,
            substrate=substrate,
            train_examples=train_examples,
            baseline=baseline,
            output_dir=output_dir,
            model_id=model_id,
            torch_seed=torch_seed,
            sample_seed=int(sample_seed or 0),
            training_source=training_source,
            prompt_mode=prompt_mode,
            training_prompt_builder=training_prompt_builder,
        )
        summary_path = output_dir / "overfit-summary.json"
        summary_path.write_text(
            json.dumps(overfit_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        _upload_outputs([summary_path], os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", ""))
        print("TOOL_USE_OVERFIT_SUMMARY " + json.dumps(overfit_summary, sort_keys=True), flush=True)
        return 0 if overfit_summary["passed"] else 1
    if diagnostic_mode == "sparse_overfit":
        sparse_summary = _sparse_overfit_diagnostic(
            model=substrate.model,
            processor=processor,
            substrate=substrate,
            train_examples=train_examples,
            baseline=baseline,
            output_dir=output_dir,
            model_id=model_id,
            torch_seed=torch_seed,
            sample_seed=int(sample_seed or 0),
            training_source=training_source,
            prompt_mode=prompt_mode,
            training_prompt_builder=training_prompt_builder,
        )
        summary_path = output_dir / "sparse-overfit-summary.json"
        summary_path.write_text(
            json.dumps(sparse_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        _upload_outputs([summary_path], os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", ""))
        print(
            "TOOL_USE_SPARSE_OVERFIT_SUMMARY " + json.dumps(sparse_summary, sort_keys=True),
            flush=True,
        )
        return 0 if sparse_summary["passed"] else 1
    trainer = ToolUseTrainer(
        model=substrate.model,
        processor=processor,
        substrate=substrate,
        monitor=monitor,
        output_dir=output_dir,
        config=ToolUseTrainingConfig(
            learning_rate=learning_rate,
            max_slots=max_slots,
            gradient_accumulation_steps=gradient_accumulation_steps,
            ewc_lambda=ewc_lambda,
            ewc_drift_threshold=ewc_drift_threshold,
        ),
        baseline_id=baseline["baseline_id"],
        baseline_snapshot_path=baseline["baseline_snapshot_path"],
        expected_baseline_snapshot_hash=baseline["baseline_snapshot_hash"],
        ewc_approved_snapshot_path=approved_snapshot["approved_snapshot_path"],
        expected_ewc_approved_snapshot_hash=approved_snapshot["approved_snapshot_hash"],
        run_metadata={
            "torch_seed": torch_seed,
            "sample_seed": sample_seed,
            "training_prompt_mode": prompt_mode,
        },
        training_prompt_builder=training_prompt_builder,
    )
    report = trainer.run_cycle(train_examples, probe_sets=probe_sets)
    tool_call_generation_probe = _tool_call_generation_probe(
        model=substrate.model,
        processor=processor,
        substrate=substrate,
        report=report,
        probe_sets=probe_sets,
        sample_seed=sample_seed,
    )
    trained_domains = set(training_source.get("domain_filter") or [])
    interference_report = _interference_report(
        report.probe_metrics,
        trained_domains=trained_domains,
        threshold=interference_threshold,
    )
    coverage_report = _coverage_report(
        train_examples=train_examples,
        probe_sets=probe_sets,
        expected_domains=_expected_domains(),
    )

    slot_changes = _slot_change_summary(
        before_snapshot=Path(report.before_snapshot_path),
        after_snapshot=Path(report.after_snapshot_path),
        selected_slots=set(report.selected_slots),
    )
    representative_delta = report.tensor_delta_records[0] if report.tensor_delta_records else {}
    summary = {
        "model_id": model_id,
        "torch_seed": torch_seed,
        "sample_seed": sample_seed,
        "training_source": training_source,
        "sample_count": len(train_examples),
        "train_example_counts": _example_counts(train_examples),
        "slot_count": substrate.slot_count,
        "selected_slot_count": len(report.selected_slots),
        "eligible_slot_count": len(report.eligible_slots),
        "learning_event_committed": report.committed,
        "learning_event_id": report.learning_event_id,
        "dataset_hash": report.dataset_hash,
        "baseline_id": report.baseline_id,
        "baseline_snapshot": report.baseline_snapshot_path,
        "baseline_snapshot_hash": report.baseline_snapshot_hash,
        "baseline_restored": report.baseline_restored,
        "baseline_source": baseline["baseline_source"],
        "baseline_metadata_path": baseline["baseline_metadata_path"],
        "gradient_accumulation_steps": report.gradient_accumulation_steps,
        "training_prompt_mode": prompt_mode,
        "loss_before_update": report.loss_before_update,
        "task_loss_before_update": report.task_loss_before_update,
        "total_loss_before_update": report.total_loss_before_update,
        "ewc_penalty_value": report.ewc_penalty_value,
        "ewc_approved_snapshot": report.ewc_approved_snapshot_path,
        "ewc_approved_snapshot_hash": report.ewc_approved_snapshot_hash,
        "ewc_drift_report": report.ewc_drift_report,
        "ewc_drift_value": (
            report.ewc_drift_report.get("drift_value") if report.ewc_drift_report else None
        ),
        "ewc_drift_threshold": (
            report.ewc_drift_report.get("drift_threshold") if report.ewc_drift_report else None
        ),
        "ewc_drift_within_bound": (
            report.ewc_drift_report.get("drift_within_bound") if report.ewc_drift_report else None
        ),
        "inline_drift_required": require_inline_drift,
        "candidate_id": report.candidate_id,
        "candidate_decision": report.candidate_decision,
        "candidate_rejection_reason": report.candidate_rejection_reason,
        "candidate_snapshot": report.candidate_snapshot_path,
        "candidate_snapshot_hash": report.candidate_snapshot_hash,
        "start_snapshot": report.start_snapshot_path,
        "start_snapshot_hash": report.start_snapshot_hash,
        "restored_snapshot_hash": report.restored_snapshot_hash,
        "drift_bound_enforced": report.drift_bound_enforced,
        "expect_rejection": expect_rejection,
        "probe_metrics": report.probe_metrics,
        "probe_any_improved": _any_probe_improved(report.probe_metrics),
        "tool_call_generation_probe": tool_call_generation_probe,
        "interference_report": interference_report,
        "coverage_report": coverage_report,
        "probe_example_counts": {
            name: _example_counts(examples) for name, examples in sorted(probe_sets.items())
        },
        "representative_tensor_delta_record": representative_delta,
        "representative_tensor_delta_norm": float(
            representative_delta.get("total_delta_norm", 0.0)
        ),
        "base_weights_unchanged": base_weights_unchanged,
        "selected_lora_slots_changed": slot_changes["selected_changed"]
        == len(report.selected_slots),
        "unselected_lora_slots_unchanged": slot_changes["unselected_changed"] == 0,
        "before_snapshot": report.before_snapshot_path,
        "before_snapshot_hash": report.before_snapshot_hash,
        "after_snapshot": report.after_snapshot_path,
        "after_snapshot_hash": report.after_snapshot_hash,
        "selection_diagnostics": report.selection_diagnostics_path,
        "selection_diagnostics_hash": report.selection_diagnostics_hash,
        "ewc_lambda": ewc_lambda,
        "gcs_output_uri": os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", ""),
        "baseline_uploaded": False,
        "selection_diagnostics_uploaded": False,
        "snapshots_uploaded": False,
    }
    summary_path = output_dir / "summary.json"
    summary["baseline_uploaded"] = _upload_outputs(
        [
            path
            for path in (
                Path(str(report.baseline_snapshot_path)) if report.baseline_snapshot_path else None,
                Path(str(baseline["baseline_metadata_path"]))
                if baseline["baseline_metadata_path"]
                else None,
            )
            if path is not None
        ],
        os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", ""),
    )
    summary["selection_diagnostics_uploaded"] = _upload_outputs(
        [Path(report.selection_diagnostics_path)] if report.selection_diagnostics_path else [],
        os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", ""),
    )
    summary["snapshots_uploaded"] = _upload_outputs(
        [
            Path(report.before_snapshot_path),
            Path(report.after_snapshot_path),
        ],
        os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", ""),
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _upload_outputs([summary_path], os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", ""))
    print("TOOL_USE_TRAINING_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if _summary_passed(summary) else 1


def _base_weights_frozen(model: Any) -> bool:
    return all(
        (not parameter.requires_grad)
        for name, parameter in model.named_parameters()
        if "lora_A" not in name and "lora_B" not in name
    )


def _slot_change_summary(
    *, before_snapshot: Path, after_snapshot: Path, selected_slots: set[int]
) -> dict[str, int]:
    import numpy as np

    selected_changed = 0
    unselected_changed = 0
    with (
        np.load(before_snapshot, allow_pickle=False) as before,
        np.load(after_snapshot, allow_pickle=False) as after,
    ):
        metadata = json.loads(str(before["metadata_json"].item()))
        for item in metadata["slot_metadata"]:
            slot_id = int(item["slot_id"])
            module_name = str(item["module_name"])
            rank_index = int(item["rank_index"])
            module_index = list(metadata["target_modules"]).index(module_name)
            before_a = before[f"module_{module_index}_lora_a"][rank_index, :]
            after_a = after[f"module_{module_index}_lora_a"][rank_index, :]
            before_b = before[f"module_{module_index}_lora_b"][:, rank_index]
            after_b = after[f"module_{module_index}_lora_b"][:, rank_index]
            changed = not (np.allclose(before_a, after_a) and np.allclose(before_b, after_b))
            if changed and slot_id in selected_slots:
                selected_changed += 1
            elif changed:
                unselected_changed += 1
    return {"selected_changed": selected_changed, "unselected_changed": unselected_changed}


def _upload_outputs(paths: list[Path], gcs_output_uri: str) -> bool:
    if not paths:
        return False
    if not gcs_output_uri:
        return False
    for path in paths:
        subprocess.run(
            ["gsutil", "-q", "cp", str(path), gcs_output_uri.rstrip("/") + "/"], check=True
        )
    return True


def _summary_passed(summary: dict[str, Any]) -> bool:
    upload_required = bool(summary.get("gcs_output_uri"))
    expect_rejection = bool(summary.get("expect_rejection"))
    inline_drift_required = bool(summary.get("inline_drift_required"))
    interference = summary.get("interference_report") or {}
    interference_passed = bool(interference.get("passed", True))
    coverage = summary.get("coverage_report") or {}
    coverage_passed = bool(coverage.get("passed", True))
    baseline_required = bool(summary.get("baseline_id") or summary.get("baseline_snapshot"))
    diagnostics_required = bool(summary.get("selection_diagnostics_hash"))
    common_passed = (
        summary["sample_count"] > 0
        and summary.get("torch_seed") is not None
        and summary.get("sample_seed") is not None
        and summary["slot_count"] > 0
        and summary["selected_slot_count"] > 0
        and summary["representative_tensor_delta_norm"] > 0.0
        and summary["base_weights_unchanged"]
        and summary["selected_lora_slots_changed"]
        and summary["unselected_lora_slots_unchanged"]
        and summary["ewc_lambda"] >= 0.0
        and (not baseline_required or bool(summary.get("baseline_snapshot_hash")))
        and (not baseline_required or bool(summary.get("baseline_restored")))
        and (not inline_drift_required or bool(summary.get("ewc_approved_snapshot_hash")))
        and (not inline_drift_required or summary.get("ewc_drift_within_bound") is not None)
        and diagnostics_required
        and interference_passed
        and coverage_passed
        and (not upload_required or summary["snapshots_uploaded"])
        and (not upload_required or summary["selection_diagnostics_uploaded"])
        and (not upload_required or not baseline_required or summary["baseline_uploaded"])
    )
    if not common_passed:
        return False
    if expect_rejection:
        return (
            summary.get("candidate_decision") == "rejected_drift_exceeded"
            and not summary["learning_event_committed"]
            and bool(summary.get("ewc_approved_snapshot_hash"))
            and summary.get("ewc_drift_within_bound") is False
            and summary.get("restored_snapshot_hash") == summary.get("start_snapshot_hash")
        )
    return (
        summary["learning_event_committed"]
        and summary.get("candidate_decision", "accepted") == "accepted"
        and summary["probe_any_improved"]
        and (
            summary["ewc_lambda"] == 0.0
            or (
                bool(summary.get("ewc_approved_snapshot_hash"))
                and summary.get("ewc_drift_within_bound") is True
            )
        )
    )


def _prepare_baseline_snapshot(
    substrate: LoRAMemorySubstrate,
    output_dir: Path,
    *,
    model_id: str,
    rank: int,
    last_n_layers: int,
    torch_seed: int,
) -> dict[str, Any]:
    baseline_id = os.environ.get("VECL_TRAIN_BASELINE_ID", "").strip()
    baseline_uri = os.environ.get("VECL_TRAIN_BASELINE_SNAPSHOT_URI", "").strip()
    baseline_path_raw = os.environ.get("VECL_TRAIN_BASELINE_SNAPSHOT_PATH", "").strip()
    expected_hash = os.environ.get("VECL_TRAIN_BASELINE_SNAPSHOT_HASH", "").strip()
    if baseline_uri:
        path = output_dir / "baseline-lora.npz"
        _download_gcs_file(baseline_uri, path)
        source = "gcs"
    elif baseline_path_raw:
        path = Path(baseline_path_raw)
        source = "local"
    elif baseline_id:
        path = output_dir / f"{baseline_id}-baseline-lora.npz"
        substrate.snapshot(path)
        source = "generated"
    else:
        return {
            "baseline_id": None,
            "baseline_snapshot_path": None,
            "baseline_snapshot_hash": None,
            "baseline_metadata_path": None,
            "baseline_source": "none",
        }
    if not path.exists():
        raise ValueError(f"baseline snapshot not found: {path}")
    baseline_hash = snapshot_file_hash(path)
    if expected_hash and baseline_hash != expected_hash:
        raise ValueError("baseline snapshot hash does not match VECL_TRAIN_BASELINE_SNAPSHOT_HASH")
    metadata_path = output_dir / f"{baseline_id or 'baseline'}-baseline-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "baseline_id": baseline_id or None,
                "baseline_snapshot_hash": baseline_hash,
                "baseline_source": source,
                "model_id": model_id,
                "lora_rank": rank,
                "last_n_layers": last_n_layers,
                "torch_seed": torch_seed,
                "slot_count": substrate.slot_count,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "baseline_id": baseline_id or None,
        "baseline_snapshot_path": str(path),
        "baseline_snapshot_hash": baseline_hash,
        "baseline_metadata_path": str(metadata_path),
        "baseline_source": source,
    }


def _prepare_ewc_approved_snapshot(
    output_dir: Path, *, fallback_path: str | None, fallback_hash: str | None
) -> dict[str, Any]:
    approved_uri = os.environ.get("VECL_TRAIN_EWC_APPROVED_SNAPSHOT_URI", "").strip()
    approved_path_raw = os.environ.get("VECL_TRAIN_EWC_APPROVED_SNAPSHOT_PATH", "").strip()
    expected_hash = os.environ.get("VECL_TRAIN_EWC_APPROVED_SNAPSHOT_HASH", "").strip()
    if approved_uri:
        path = output_dir / "ewc-approved-lora.npz"
        _download_gcs_file(approved_uri, path)
    elif approved_path_raw:
        path = Path(approved_path_raw)
    elif fallback_path:
        path = Path(fallback_path)
        expected_hash = expected_hash or str(fallback_hash or "")
    else:
        return {"approved_snapshot_path": None, "approved_snapshot_hash": None}
    if not path.exists():
        raise ValueError(f"EWC approved snapshot not found: {path}")
    actual_hash = snapshot_file_hash(path)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            "EWC approved snapshot hash does not match VECL_TRAIN_EWC_APPROVED_SNAPSHOT_HASH"
        )
    return {"approved_snapshot_path": str(path), "approved_snapshot_hash": actual_hash}


def _training_prompt_builder() -> tuple[str, TrainingPromptBuilder | None]:
    mode = os.environ.get("VECL_TRAIN_PROMPT_MODE", "tool_call_author").strip().lower()
    if mode in {"", "raw"}:
        return "raw", None
    if mode != "tool_call_author":
        raise ValueError("VECL_TRAIN_PROMPT_MODE must be raw or tool_call_author")
    cards = list(_tool_call_probe_cards().values())

    def _builder(example: SupervisedToolUseExample) -> str:
        if example.task_kind != "tool_call_json":
            return example.prompt
        return _tool_call_generation_prompt(example, cards)

    return mode, _builder


def _tiny_overfit_diagnostic(
    *,
    model: Any,
    processor: Any,
    substrate: LoRAMemorySubstrate,
    train_examples: list[SupervisedToolUseExample],
    baseline: dict[str, Any],
    output_dir: Path,
    model_id: str,
    torch_seed: int,
    sample_seed: int,
    training_source: dict[str, Any],
    prompt_mode: str,
    training_prompt_builder: TrainingPromptBuilder | None,
) -> dict[str, Any]:
    import torch

    examples = _overfit_examples(train_examples, sample_seed=sample_seed)
    if not examples:
        raise ValueError("tiny overfit diagnostic selected no examples")
    baseline_path = baseline.get("baseline_snapshot_path")
    if baseline_path:
        substrate.restore(str(baseline_path))
    before_path = output_dir / "overfit-before-lora.npz"
    after_path = output_dir / "overfit-after-lora.npz"
    substrate.snapshot(before_path)
    before_hash = snapshot_file_hash(before_path)
    before_losses = _diagnostic_losses(
        model=model,
        processor=processor,
        examples=examples,
        training_prompt_builder=training_prompt_builder,
    )
    generation_count = _int_env("VECL_TRAIN_OVERFIT_GENERATION_PROBE_COUNT", 0)
    generation_examples = [
        example for example in examples if example.task_kind == "tool_call_json"
    ][:generation_count]
    cards = _tool_call_probe_cards()
    validation_config = ToolCallValidationConfig(
        tenant_id="tiny-overfit-diagnostic",
        request_id="tiny-overfit-diagnostic",
        terraform_config_dir=os.environ.get(
            "VECL_TRAIN_TERRAFORM_CONFIG_DIR", "/tmp/vecl-terraform-fixture"
        ),
    )
    before_generation_rows = (
        _generate_tool_call_rows(
            model=model,
            processor=processor,
            examples=generation_examples,
            cards=cards,
            config=validation_config,
            max_new_tokens=_int_env("VECL_TRAIN_TOOL_CALL_GENERATION_MAX_NEW_TOKENS", 256),
        )
        if generation_examples
        else []
    )
    _empty_cuda_cache(torch)

    trainable_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
    ]
    if not trainable_parameters:
        raise ValueError("tiny overfit diagnostic found no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(os.environ.get("VECL_TRAIN_OVERFIT_LEARNING_RATE", "0.01")),
    )
    steps = _int_env("VECL_TRAIN_OVERFIT_STEPS", 25)
    if steps <= 0:
        raise ValueError("VECL_TRAIN_OVERFIT_STEPS must be > 0")
    losses_by_step: list[float] = []
    model.train()
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for example in examples:
            loss = supervised_loss_for_example(
                model=model,
                processor=processor,
                example=example,
                torch=torch,
                training_prompt_builder=training_prompt_builder,
            )
            total_loss += float(loss.detach().cpu().item())
            (loss / len(examples)).backward()
        mean_loss_value = total_loss / len(examples)
        losses_by_step.append(mean_loss_value)
        optimizer.step()
        _empty_cuda_cache(torch)

    after_losses = _diagnostic_losses(
        model=model,
        processor=processor,
        examples=examples,
        training_prompt_builder=training_prompt_builder,
    )
    substrate.snapshot(after_path)
    after_hash = snapshot_file_hash(after_path)
    after_generation_rows = (
        _generate_tool_call_rows(
            model=model,
            processor=processor,
            examples=generation_examples,
            cards=cards,
            config=validation_config,
            max_new_tokens=_int_env("VECL_TRAIN_TOOL_CALL_GENERATION_MAX_NEW_TOKENS", 256),
        )
        if generation_examples
        else []
    )
    loss_summary = _diagnostic_loss_summary(before_losses, after_losses)
    min_drop = float(os.environ.get("VECL_TRAIN_OVERFIT_MIN_CE_DROP", "0.05"))
    passed = loss_summary["mean_drop"] >= min_drop
    summary = {
        "diagnostic_mode": "tiny_overfit",
        "model_id": model_id,
        "torch_seed": torch_seed,
        "sample_seed": sample_seed,
        "training_source": training_source,
        "training_prompt_mode": prompt_mode,
        "sample_count": len(examples),
        "example_ids": [example.example_id for example in examples],
        "example_counts": _example_counts(examples),
        "steps": steps,
        "learning_rate": float(os.environ.get("VECL_TRAIN_OVERFIT_LEARNING_RATE", "0.01")),
        "min_ce_drop": min_drop,
        "loss_summary": loss_summary,
        "losses_by_step": losses_by_step,
        "baseline_id": baseline.get("baseline_id"),
        "baseline_snapshot_hash": baseline.get("baseline_snapshot_hash"),
        "baseline_restored": bool(baseline_path),
        "before_snapshot": str(before_path),
        "before_snapshot_hash": before_hash,
        "after_snapshot": str(after_path),
        "after_snapshot_hash": after_hash,
        "snapshot_changed": before_hash != after_hash,
        "generation_probe": {
            "enabled": bool(generation_examples),
            "sample_count": len(generation_examples),
            "before": _tool_call_generation_metrics(before_generation_rows),
            "after": _tool_call_generation_metrics(after_generation_rows),
            "delta": _tool_call_generation_delta(before_generation_rows, after_generation_rows)
            if generation_examples
            else {},
            "rows": [
                {
                    "example_id": example.example_id,
                    "domain": example.metadata.get("corpus_domain"),
                    "category": example.metadata.get("corpus_category"),
                    "before": before_row,
                    "after": after_row,
                }
                for example, before_row, after_row in zip(
                    generation_examples,
                    before_generation_rows,
                    after_generation_rows,
                    strict=True,
                )
            ],
        },
        "gcs_output_uri": os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", ""),
        "snapshots_uploaded": False,
        "passed": passed,
    }
    summary["snapshots_uploaded"] = _upload_outputs(
        [before_path, after_path], os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", "")
    )
    return summary


def _sparse_overfit_diagnostic(
    *,
    model: Any,
    processor: Any,
    substrate: LoRAMemorySubstrate,
    train_examples: list[SupervisedToolUseExample],
    baseline: dict[str, Any],
    output_dir: Path,
    model_id: str,
    torch_seed: int,
    sample_seed: int,
    training_source: dict[str, Any],
    prompt_mode: str,
    training_prompt_builder: TrainingPromptBuilder | None,
) -> dict[str, Any]:
    import torch

    examples = _overfit_examples(train_examples, sample_seed=sample_seed)
    if not examples:
        raise ValueError("sparse overfit diagnostic selected no examples")
    baseline_path = baseline.get("baseline_snapshot_path")
    if baseline_path:
        substrate.restore(str(baseline_path))
    before_path = output_dir / "sparse-overfit-before-lora.npz"
    after_path = output_dir / "sparse-overfit-after-lora.npz"
    substrate.snapshot(before_path)
    before_hash = snapshot_file_hash(before_path)
    before_losses = _diagnostic_losses(
        model=model,
        processor=processor,
        examples=examples,
        training_prompt_builder=training_prompt_builder,
    )
    ledger = ProvenanceLedger()
    monitor = SparseUpdateMonitor(ledger, actor="sparse-overfit-diagnostic")
    cycles = _int_env("VECL_TRAIN_SPARSE_OVERFIT_CYCLES", 25)
    if cycles <= 0:
        raise ValueError("VECL_TRAIN_SPARSE_OVERFIT_CYCLES must be > 0")
    learning_rate = float(os.environ.get("VECL_TRAIN_SPARSE_OVERFIT_LEARNING_RATE", "0.01"))
    max_slots = _int_env("VECL_TRAIN_SPARSE_OVERFIT_MAX_SLOTS", _int_env("VECL_TRAIN_MAX_SLOTS", 8))
    gradient_accumulation_steps = _int_env(
        "VECL_TRAIN_SPARSE_OVERFIT_GRADIENT_ACCUMULATION_STEPS",
        min(len(examples), _int_env("VECL_TRAIN_GRADIENT_ACCUMULATION_STEPS", 8)),
    )
    cycle_summaries: list[dict[str, Any]] = []
    first_baseline_path = str(baseline_path) if baseline_path else None
    first_baseline_hash = str(baseline.get("baseline_snapshot_hash") or "")
    for cycle in range(cycles):
        cycle_dir = output_dir / "sparse-overfit-cycles" / f"cycle-{cycle + 1:03d}"
        trainer = ToolUseTrainer(
            model=substrate.model,
            processor=processor,
            substrate=substrate,
            monitor=monitor,
            output_dir=cycle_dir,
            config=ToolUseTrainingConfig(
                learning_rate=learning_rate,
                max_slots=max_slots,
                gradient_accumulation_steps=gradient_accumulation_steps,
                ewc_lambda=0.0,
                ewc_drift_threshold=None,
                enforce_drift_bound=False,
            ),
            actor="sparse-overfit-diagnostic",
            baseline_id=baseline.get("baseline_id"),
            baseline_snapshot_path=first_baseline_path if cycle == 0 else None,
            expected_baseline_snapshot_hash=first_baseline_hash if cycle == 0 else None,
            run_metadata={
                "diagnostic_mode": "sparse_overfit",
                "diagnostic_cycle": cycle + 1,
                "torch_seed": torch_seed,
                "sample_seed": sample_seed,
                "training_prompt_mode": prompt_mode,
            },
            training_prompt_builder=training_prompt_builder,
        )
        report = trainer.run_cycle(examples, probe_sets={})
        after_cycle_losses = _diagnostic_losses(
            model=model,
            processor=processor,
            examples=examples,
            training_prompt_builder=training_prompt_builder,
        )
        cycle_summaries.append(
            _sparse_cycle_summary(
                cycle=cycle + 1,
                report=report.to_payload(),
                before_losses=before_losses,
                after_losses=after_cycle_losses,
            )
        )
        model.zero_grad(set_to_none=True)
        _empty_cuda_cache(torch)
    substrate.snapshot(after_path)
    after_hash = snapshot_file_hash(after_path)
    after_losses = _diagnostic_losses(
        model=model,
        processor=processor,
        examples=examples,
        training_prompt_builder=training_prompt_builder,
    )
    loss_summary = _diagnostic_loss_summary(before_losses, after_losses)
    min_drop = float(os.environ.get("VECL_TRAIN_SPARSE_OVERFIT_MIN_CE_DROP", "0.05"))
    passed = loss_summary["mean_drop"] >= min_drop
    summary = {
        "diagnostic_mode": "sparse_overfit",
        "model_id": model_id,
        "torch_seed": torch_seed,
        "sample_seed": sample_seed,
        "training_source": training_source,
        "training_prompt_mode": prompt_mode,
        "sample_count": len(examples),
        "example_ids": [example.example_id for example in examples],
        "example_counts": _example_counts(examples),
        "cycles": cycles,
        "learning_rate": learning_rate,
        "max_slots": max_slots,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "min_ce_drop": min_drop,
        "loss_summary": loss_summary,
        "cycle_summaries": cycle_summaries,
        "baseline_id": baseline.get("baseline_id"),
        "baseline_snapshot_hash": baseline.get("baseline_snapshot_hash"),
        "baseline_restored": bool(baseline_path),
        "before_snapshot": str(before_path),
        "before_snapshot_hash": before_hash,
        "after_snapshot": str(after_path),
        "after_snapshot_hash": after_hash,
        "snapshot_changed": before_hash != after_hash,
        "ledger_event_count": len(ledger.events()),
        "learning_commit_count": len(ledger.find_by_type(EventType.LEARNING_EVENT_COMMITTED)),
        "gcs_output_uri": os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", ""),
        "snapshots_uploaded": False,
        "passed": passed,
    }
    summary["snapshots_uploaded"] = _upload_outputs(
        [before_path, after_path], os.environ.get("VECL_TRAIN_GCS_OUTPUT_URI", "")
    )
    return summary


def _sparse_cycle_summary(
    *,
    cycle: int,
    report: dict[str, Any],
    before_losses: list[float],
    after_losses: list[float],
) -> dict[str, Any]:
    delta_records = list(report.get("tensor_delta_records") or [])
    return {
        "cycle": cycle,
        "committed": bool(report.get("committed")),
        "selected_slots": list(report.get("selected_slots") or []),
        "selected_slot_count": len(report.get("selected_slots") or []),
        "tensor_delta_norm_sum": sum(
            float(record.get("total_delta_norm", 0.0)) for record in delta_records
        ),
        "before_snapshot_hash": report.get("before_snapshot_hash"),
        "after_snapshot_hash": report.get("after_snapshot_hash"),
        "candidate_decision": report.get("candidate_decision"),
        "loss_summary": _diagnostic_loss_summary(before_losses, after_losses),
    }


def _empty_cuda_cache(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _overfit_examples(
    examples: list[SupervisedToolUseExample], *, sample_seed: int
) -> list[SupervisedToolUseExample]:
    task_kind = os.environ.get("VECL_TRAIN_OVERFIT_TASK_KIND", "tool_call_json").strip()
    pool = [example for example in examples if not task_kind or example.task_kind == task_kind]
    if not pool:
        pool = examples
    return _stratified_sample(
        pool,
        count=_int_env("VECL_TRAIN_OVERFIT_SAMPLE_COUNT", 8),
        seed=sample_seed + 301,
    )


def _diagnostic_losses(
    *,
    model: Any,
    processor: Any,
    examples: list[SupervisedToolUseExample],
    training_prompt_builder: TrainingPromptBuilder | None,
) -> list[float]:
    import torch

    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for example in examples:
            loss = supervised_loss_for_example(
                model=model,
                processor=processor,
                example=example,
                torch=torch,
                training_prompt_builder=training_prompt_builder,
            )
            losses.append(float(loss.detach().cpu().item()))
    return losses


def _diagnostic_loss_summary(
    before_losses: list[float], after_losses: list[float]
) -> dict[str, Any]:
    if len(before_losses) != len(after_losses):
        raise ValueError("diagnostic loss arrays must have the same length")
    if not before_losses:
        return {
            "sample_count": 0,
            "mean_before": None,
            "mean_after": None,
            "mean_delta": None,
            "mean_drop": None,
            "relative_mean_change": None,
            "improved_count": 0,
            "worsened_count": 0,
            "tied_count": 0,
            "before_losses": [],
            "after_losses": [],
            "deltas": [],
        }
    deltas = [after - before for before, after in zip(before_losses, after_losses, strict=True)]
    mean_before = sum(before_losses) / len(before_losses)
    mean_after = sum(after_losses) / len(after_losses)
    return {
        "sample_count": len(before_losses),
        "mean_before": mean_before,
        "mean_after": mean_after,
        "mean_delta": mean_after - mean_before,
        "mean_drop": mean_before - mean_after,
        "relative_mean_change": (mean_after - mean_before) / mean_before if mean_before else 0.0,
        "improved_count": sum(int(delta < 0) for delta in deltas),
        "worsened_count": sum(int(delta > 0) for delta in deltas),
        "tied_count": sum(int(delta == 0) for delta in deltas),
        "before_losses": before_losses,
        "after_losses": after_losses,
        "deltas": deltas,
    }


def _tool_call_generation_probe(
    *,
    model: Any,
    processor: Any,
    substrate: LoRAMemorySubstrate,
    report: Any,
    probe_sets: dict[str, list[SupervisedToolUseExample]],
    sample_seed: int | None,
) -> dict[str, Any]:
    count = _int_env("VECL_TRAIN_TOOL_CALL_GENERATION_PROBE_COUNT", 0)
    if count <= 0:
        return {"enabled": False, "sample_count": 0}
    if not report.committed:
        return {
            "enabled": True,
            "completed": False,
            "sample_count": 0,
            "reason": "generation probe skipped for uncommitted candidate",
        }
    heldout = [
        example
        for example in probe_sets.get("heldout", [])
        if example.task_kind == "tool_call_json"
    ]
    examples = _stratified_sample(
        heldout,
        count=count,
        seed=(sample_seed or 0) + 101,
    )
    if not examples:
        return {
            "enabled": True,
            "completed": False,
            "sample_count": 0,
            "reason": "no heldout tool-call examples available",
        }
    max_new_tokens = _int_env("VECL_TRAIN_TOOL_CALL_GENERATION_MAX_NEW_TOKENS", 256)
    cards = _tool_call_probe_cards()
    validation_config = ToolCallValidationConfig(
        tenant_id="tool-call-generation-probe",
        request_id="tool-call-generation-probe",
        terraform_config_dir=os.environ.get(
            "VECL_TRAIN_TERRAFORM_CONFIG_DIR", "/tmp/vecl-terraform-fixture"
        ),
    )

    substrate.restore(report.before_snapshot_path)
    before_rows = _generate_tool_call_rows(
        model=model,
        processor=processor,
        examples=examples,
        cards=cards,
        config=validation_config,
        max_new_tokens=max_new_tokens,
    )
    substrate.restore(report.after_snapshot_path)
    after_rows = _generate_tool_call_rows(
        model=model,
        processor=processor,
        examples=examples,
        cards=cards,
        config=validation_config,
        max_new_tokens=max_new_tokens,
    )
    return {
        "enabled": True,
        "completed": True,
        "sample_count": len(examples),
        "max_new_tokens": max_new_tokens,
        "before": _tool_call_generation_metrics(before_rows),
        "after": _tool_call_generation_metrics(after_rows),
        "delta": _tool_call_generation_delta(before_rows, after_rows),
        "rows": [
            {
                "example_id": example.example_id,
                "domain": example.metadata.get("corpus_domain"),
                "category": example.metadata.get("corpus_category"),
                "before": before_row,
                "after": after_row,
            }
            for example, before_row, after_row in zip(
                examples, before_rows, after_rows, strict=True
            )
        ],
    }


def _generate_tool_call_rows(
    *,
    model: Any,
    processor: Any,
    examples: list[SupervisedToolUseExample],
    cards: dict[str, SpecialistCard],
    config: ToolCallValidationConfig,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    return [
        _score_tool_call_generation(
            generated_text=_generate_text(
                model=model,
                processor=processor,
                prompt=_tool_call_generation_prompt(example, list(cards.values())),
                max_new_tokens=max_new_tokens,
            ),
            expected_text=example.target_text,
            cards=cards,
            config=config,
        )
        for example in examples
    ]


def _tool_call_generation_prompt(
    example: SupervisedToolUseExample, cards: list[SpecialistCard]
) -> str:
    try:
        expected = parse_tool_call_response(example.target_text)
        task_type = expected.task_type or "tool_call_json"
    except ToolCallParseError:
        task_type = "tool_call_json"
    request = SpecialistRequest(
        example.example_id,
        example.tenant_id,
        task_type,
        {"query": example.prompt},
        {},
        {},
    )
    return build_tool_call_prompt(request, cards)


def _generate_text(
    *,
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    model.eval()
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(processor, "apply_chat_template"):
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = dict(inputs)
    else:
        inputs = dict(processor(prompt, add_special_tokens=True, return_tensors="pt"))
    device = _model_input_device(model, torch)
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }
    input_length = int(model_inputs["input_ids"].shape[-1])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    generated_ids = outputs[0, input_length:]
    return str(tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()


def _model_input_device(model: Any, torch: Any) -> Any:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _score_tool_call_generation(
    *,
    generated_text: str,
    expected_text: str,
    cards: dict[str, SpecialistCard],
    config: ToolCallValidationConfig,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "generated_text": generated_text[:1000],
        "parse_ok": False,
        "validation_ok": False,
        "specialist_match": False,
        "task_type_match": False,
        "payload_match": False,
        "all_correct": False,
    }
    try:
        expected = parse_tool_call_response(expected_text)
        expected_validated = validate_tool_call(expected, cards=cards, config=config)
    except (ToolCallParseError, ToolCallValidationError, ValueError) as exc:
        return {**row, "expected_error": str(exc)}
    row["expected_specialist_id"] = expected.specialist_id
    row["expected_task_type"] = expected.task_type
    try:
        generated = parse_tool_call_response(generated_text)
    except ToolCallParseError as exc:
        return {**row, "parse_error": str(exc)}
    row["parse_ok"] = True
    row["generated_specialist_id"] = generated.specialist_id
    row["generated_task_type"] = generated.task_type
    row["specialist_match"] = generated.specialist_id == expected.specialist_id
    row["task_type_match"] = generated.task_type == expected.task_type
    try:
        generated_validated = validate_tool_call(generated, cards=cards, config=config)
    except ToolCallValidationError as exc:
        return {**row, "validation_error": str(exc)}
    row["validation_ok"] = True
    row["payload_match"] = (
        generated_validated.request.input_payload == expected_validated.request.input_payload
    )
    row["all_correct"] = bool(
        row["specialist_match"] and row["task_type_match"] and row["payload_match"]
    )
    return row


def _tool_call_generation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "sample_count": total,
        "parse_success": _count_rows(rows, "parse_ok"),
        "validation_success": _count_rows(rows, "validation_ok"),
        "specialist_correct": _count_rows(rows, "specialist_match"),
        "task_type_correct": _count_rows(rows, "task_type_match"),
        "payload_correct": _count_rows(rows, "payload_match"),
        "all_correct": _count_rows(rows, "all_correct"),
        "parse_success_rate": _rate(_count_rows(rows, "parse_ok"), total),
        "validation_success_rate": _rate(_count_rows(rows, "validation_ok"), total),
        "specialist_accuracy": _rate(_count_rows(rows, "specialist_match"), total),
        "task_type_accuracy": _rate(_count_rows(rows, "task_type_match"), total),
        "payload_accuracy": _rate(_count_rows(rows, "payload_match"), total),
        "exact_tool_call_accuracy": _rate(_count_rows(rows, "all_correct"), total),
    }


def _tool_call_generation_delta(
    before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    before = _tool_call_generation_metrics(before_rows)
    after = _tool_call_generation_metrics(after_rows)
    return {
        "exact_tool_call_accuracy_delta": after["exact_tool_call_accuracy"]
        - before["exact_tool_call_accuracy"],
        "payload_accuracy_delta": after["payload_accuracy"] - before["payload_accuracy"],
        "specialist_accuracy_delta": after["specialist_accuracy"] - before["specialist_accuracy"],
        "improved_count": sum(
            int((not before_row.get("all_correct")) and bool(after_row.get("all_correct")))
            for before_row, after_row in zip(before_rows, after_rows, strict=True)
        ),
        "worsened_count": sum(
            int(bool(before_row.get("all_correct")) and not after_row.get("all_correct"))
            for before_row, after_row in zip(before_rows, after_rows, strict=True)
        ),
        "unchanged_count": sum(
            int(bool(before_row.get("all_correct")) == bool(after_row.get("all_correct")))
            for before_row, after_row in zip(before_rows, after_rows, strict=True)
        ),
    }


def _count_rows(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(bool(row.get(key))) for row in rows)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _tool_call_probe_cards() -> dict[str, SpecialistCard]:
    cards: dict[str, SpecialistCard] = {}
    for specialist_id, contract in sorted(SPECIALIST_PAYLOAD_CONTRACTS.items()):
        cards[specialist_id] = SpecialistCard(
            specialist_id,
            set(contract["task_types"]),
            {"description": f"{specialist_id} training probe specialist"},
            cost_hint=1.0,
            latency_hint=1.0,
            version="training-probe",
            effective_trust=0.9,
            description=f"{specialist_id} training probe specialist",
        )
    return cards


def _training_examples(
    output_dir: Path,
) -> tuple[
    list[SupervisedToolUseExample],
    dict[str, list[SupervisedToolUseExample]],
    dict[str, Any],
]:
    export_dir = _resolve_corpus_export_dir(output_dir)
    if export_dir is None:
        train_examples = [*stockfish_tool_use_examples(), *stockfish_final_answer_examples()]
        smoke_examples = stockfish_tool_use_examples()[:1]
        return (
            train_examples,
            {"smoke": smoke_examples},
            {"kind": "synthetic_stockfish_fallback"},
        )

    all_train = _load_examples(export_dir / "train.jsonl")
    all_heldout = _load_examples(export_dir / "heldout.jsonl")
    all_hard = _load_examples(export_dir / "hard-heldout.jsonl")
    domains = _domain_filter()
    train_pool = _filter_domains(all_train, domains)
    heldout_pool = _filter_domains(all_heldout, domains)
    hard_pool = _filter_domains(all_hard, domains)
    sample_seed = _int_env("VECL_TRAIN_SAMPLE_SEED", 1107)
    train_examples = _stratified_sample(
        train_pool,
        count=_int_env("VECL_TRAIN_SAMPLE_COUNT", 64),
        seed=sample_seed,
    )
    probe_sets = {
        "smoke": _stratified_sample(
            train_pool,
            count=_int_env("VECL_TRAIN_SMOKE_PROBE_COUNT", 8),
            seed=sample_seed + 1,
        ),
        "heldout": _stratified_sample(
            heldout_pool,
            count=_int_env("VECL_TRAIN_HELDOUT_PROBE_COUNT", 24),
            seed=sample_seed + 2,
        ),
        "hard_heldout": _stratified_sample(
            hard_pool,
            count=_int_env("VECL_TRAIN_HARD_PROBE_COUNT", 24),
            seed=sample_seed + 3,
        ),
    }
    probe_sets = {name: examples for name, examples in probe_sets.items() if examples}
    if not train_examples:
        raise ValueError(f"no training examples selected from {export_dir}")
    return (
        train_examples,
        probe_sets,
        {
            "kind": "corpus_export",
            "export_dir": str(export_dir),
            "export_uri": os.environ.get("VECL_TRAIN_CORPUS_EXPORT_URI", ""),
            "sample_seed": sample_seed,
            "domain_filter": sorted(domains) if domains else [],
            "available_counts": {
                "train": _example_counts(train_pool),
                "heldout": _example_counts(heldout_pool),
                "hard_heldout": _example_counts(hard_pool),
            },
        },
    )


def _resolve_corpus_export_dir(output_dir: Path) -> Path | None:
    export_uri = os.environ.get("VECL_TRAIN_CORPUS_EXPORT_URI", "").strip()
    if export_uri:
        destination = output_dir / "corpus-exports"
        _download_gcs_prefix(export_uri, destination)
        return destination
    raw = os.environ.get("VECL_TRAIN_CORPUS_EXPORT_DIR", "data/synthetic/v1-hard/exports")
    path = Path(raw)
    return path if (path / "train.jsonl").exists() else None


def _download_gcs_prefix(gcs_uri: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gsutil", "-q", "-m", "cp", "-r", gcs_uri.rstrip("/") + "/*", str(destination) + "/"],
        check=True,
    )


def _download_gcs_file(gcs_uri: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gsutil", "-q", "cp", gcs_uri, str(destination)], check=True)


def _load_examples(path: Path) -> list[SupervisedToolUseExample]:
    if not path.exists():
        return []
    examples: list[SupervisedToolUseExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(
                SupervisedToolUseExample(
                    example_id=str(payload["example_id"]),
                    tenant_id=str(payload["tenant_id"]),
                    source_id=str(payload["source_id"]),
                    authority=float(payload["authority"]),
                    prompt=str(payload["prompt"]),
                    target_text=str(payload["target_text"]),
                    task_kind=str(payload["task_kind"]),  # type: ignore[arg-type]
                    artifact_ids=list(payload.get("artifact_ids") or []),
                    claim_ids=list(payload.get("claim_ids") or []),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
    return examples


def _domain_filter() -> set[str]:
    raw = os.environ.get("VECL_TRAIN_DOMAIN_MIX", "").strip()
    if raw.lower() in {"", "__all__", "all", "*"}:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _expected_domains() -> set[str]:
    raw = os.environ.get("VECL_TRAIN_EXPECT_DOMAINS", "").strip()
    if not raw:
        return set()
    if raw.lower() in {"__all__", "all", "*"}:
        return set(DEFAULT_EXPANDED_CORPUS_DOMAINS)
    return {item.strip() for item in raw.split(",") if item.strip()}


def _filter_domains(
    examples: list[SupervisedToolUseExample], domains: set[str]
) -> list[SupervisedToolUseExample]:
    if not domains:
        return examples
    return [
        example
        for example in examples
        if str(example.metadata.get("corpus_domain") or "") in domains
    ]


def _stratified_sample(
    examples: list[SupervisedToolUseExample], *, count: int, seed: int
) -> list[SupervisedToolUseExample]:
    if count <= 0 or not examples:
        return []
    buckets: dict[str, list[SupervisedToolUseExample]] = {}
    for example in examples:
        domain = str(example.metadata.get("corpus_domain") or "unknown")
        category = str(example.metadata.get("corpus_category") or "unknown")
        buckets.setdefault(f"{domain}:{category}:{example.task_kind}", []).append(example)
    for key, bucket in buckets.items():
        bucket.sort(key=lambda item: stable_hash({"seed": seed, "key": key, "id": item.example_id}))
    selected: list[SupervisedToolUseExample] = []
    bucket_names = sorted(buckets)
    while len(selected) < count and bucket_names:
        next_names: list[str] = []
        for name in bucket_names:
            bucket = buckets[name]
            if bucket and len(selected) < count:
                selected.append(bucket.pop(0))
            if bucket:
                next_names.append(name)
        bucket_names = next_names
    return selected


def _example_counts(examples: list[SupervisedToolUseExample]) -> dict[str, Any]:
    domains = Counter(
        str(example.metadata.get("corpus_domain") or "unknown") for example in examples
    )
    categories = Counter(
        str(example.metadata.get("corpus_category") or "unknown") for example in examples
    )
    task_kinds = Counter(example.task_kind for example in examples)
    authorities = [float(example.authority) for example in examples]
    return {
        "total": len(examples),
        "domains": dict(sorted(domains.items())),
        "categories": dict(sorted(categories.items())),
        "task_kinds": dict(sorted(task_kinds.items())),
        "authority_min": min(authorities) if authorities else None,
        "authority_max": max(authorities) if authorities else None,
    }


def _coverage_report(
    *,
    train_examples: list[SupervisedToolUseExample],
    probe_sets: dict[str, list[SupervisedToolUseExample]],
    expected_domains: set[str],
) -> dict[str, Any]:
    if not expected_domains:
        return {
            "checked": False,
            "passed": True,
            "reason": "no expected domains configured",
            "expected_domains": [],
            "missing_train_domains": [],
            "missing_probe_domains": {},
        }
    train_domains = _domains_in_examples(train_examples)
    required_probe_names = _coverage_probe_names(probe_sets)
    probe_missing = {
        name: sorted(expected_domains - _domains_in_examples(examples))
        for name, examples in sorted(probe_sets.items())
        if examples and name in required_probe_names
    }
    missing_train = sorted(expected_domains - train_domains)
    missing_probe = {name: missing for name, missing in probe_missing.items() if missing}
    return {
        "checked": True,
        "passed": not missing_train and not missing_probe,
        "expected_domains": sorted(expected_domains),
        "train_domains": sorted(train_domains),
        "probe_domains": {
            name: sorted(_domains_in_examples(examples))
            for name, examples in sorted(probe_sets.items())
            if examples
        },
        "required_probe_names": sorted(required_probe_names),
        "missing_train_domains": missing_train,
        "missing_probe_domains": missing_probe,
    }


def _coverage_probe_names(probe_sets: dict[str, list[SupervisedToolUseExample]]) -> set[str]:
    raw = os.environ.get("VECL_TRAIN_COVERAGE_PROBE_NAMES", "").strip()
    if raw:
        return {item.strip() for item in raw.split(",") if item.strip()}
    return {name for name in ("heldout", "hard_heldout") if probe_sets.get(name)}


def _domains_in_examples(examples: list[SupervisedToolUseExample]) -> set[str]:
    return {
        str(example.metadata.get("corpus_domain") or "unknown")
        for example in examples
        if example.metadata.get("corpus_domain")
    }


def _any_probe_improved(metrics: dict[str, Any]) -> bool:
    if bool(metrics.get("any_improved")):
        return True
    for value in metrics.values():
        if isinstance(value, dict) and _any_probe_improved(value):
            return True
    return False


def _interference_report(
    metrics: dict[str, Any], *, trained_domains: set[str], threshold: float
) -> dict[str, Any]:
    if threshold < 0:
        raise ValueError("interference threshold must be non-negative")
    aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), dict) else metrics
    by_domain = aggregate.get("by_domain") if isinstance(aggregate, dict) else None
    if not trained_domains:
        return {
            "checked": False,
            "passed": True,
            "reason": "no focused training domain filter",
            "threshold": threshold,
            "trained_domains": [],
            "violations": [],
        }
    if not isinstance(by_domain, dict):
        return {
            "checked": False,
            "passed": True,
            "reason": "domain probe metrics unavailable",
            "threshold": threshold,
            "trained_domains": sorted(trained_domains),
            "violations": [],
        }
    violations = []
    for domain, domain_metrics in sorted(by_domain.items()):
        if domain in trained_domains or not isinstance(domain_metrics, dict):
            continue
        relative_change = domain_metrics.get("relative_mean_change")
        if relative_change is None:
            continue
        if float(relative_change) > threshold:
            violations.append(
                {
                    "domain": domain,
                    "relative_mean_change": float(relative_change),
                    "mean_before": domain_metrics.get("mean_before"),
                    "mean_after": domain_metrics.get("mean_after"),
                    "sample_count": domain_metrics.get("sample_count"),
                    "paired_t_normal_approx_p_value": domain_metrics.get(
                        "paired_t_normal_approx_p_value"
                    ),
                    "sign_test_degradation_p_value": domain_metrics.get(
                        "sign_test_degradation_p_value"
                    ),
                }
            )
    return {
        "checked": True,
        "passed": not violations,
        "threshold": threshold,
        "trained_domains": sorted(trained_domains),
        "violations": violations,
    }


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    return None if raw is None or raw == "" else float(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


if __name__ == "__main__":
    raise SystemExit(main())
