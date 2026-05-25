#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from vecl._gemma import has_allowed_cuda_device, resolve_gemma_model_class, resolve_torch_dtype
from vecl.provenance.events import stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.monitor import SparseUpdateMonitor
from vecl.substrate.lora_memory import LoRAMemorySubstrate
from vecl.training.synthetic import stockfish_final_answer_examples, stockfish_tool_use_examples
from vecl.training.tool_use_loop import (
    SupervisedToolUseExample,
    ToolUseTrainer,
    ToolUseTrainingConfig,
    snapshot_file_hash,
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
    interference_threshold = float(os.environ.get("VECL_TRAIN_INTERFERENCE_THRESHOLD", "0.05"))
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

    ledger = ProvenanceLedger()
    monitor = SparseUpdateMonitor(ledger, actor="tool-use-trainer")
    train_examples, probe_sets, training_source = _training_examples(output_dir)
    sample_seed = training_source.get("sample_seed")
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
        ewc_approved_snapshot_path=baseline["baseline_snapshot_path"] if ewc_lambda > 0 else None,
        run_metadata={
            "torch_seed": torch_seed,
            "sample_seed": sample_seed,
        },
    )
    report = trainer.run_cycle(train_examples, probe_sets=probe_sets)
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
        "probe_metrics": report.probe_metrics,
        "probe_any_improved": _any_probe_improved(report.probe_metrics),
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
    interference = summary.get("interference_report") or {}
    interference_passed = bool(interference.get("passed", True))
    coverage = summary.get("coverage_report") or {}
    coverage_passed = bool(coverage.get("passed", True))
    baseline_required = bool(summary.get("baseline_id") or summary.get("baseline_snapshot"))
    diagnostics_required = bool(summary.get("selection_diagnostics_hash"))
    return (
        summary["sample_count"] > 0
        and summary.get("torch_seed") is not None
        and summary.get("sample_seed") is not None
        and summary["slot_count"] > 0
        and summary["selected_slot_count"] > 0
        and summary["learning_event_committed"]
        and summary["probe_any_improved"]
        and summary["representative_tensor_delta_norm"] > 0.0
        and summary["base_weights_unchanged"]
        and summary["selected_lora_slots_changed"]
        and summary["unselected_lora_slots_unchanged"]
        and summary["ewc_lambda"] >= 0.0
        and (
            summary["ewc_lambda"] == 0.0
            or (
                bool(summary.get("ewc_approved_snapshot_hash"))
                and summary.get("ewc_drift_within_bound") is True
            )
        )
        and (not baseline_required or bool(summary.get("baseline_snapshot_hash")))
        and (not baseline_required or bool(summary.get("baseline_restored")))
        and diagnostics_required
        and interference_passed
        and coverage_passed
        and (not upload_required or summary["snapshots_uploaded"])
        and (not upload_required or summary["selection_diagnostics_uploaded"])
        and (not upload_required or not baseline_required or summary["baseline_uploaded"])
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


if __name__ == "__main__":
    raise SystemExit(main())
