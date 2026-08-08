#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from vecl._gemma import has_allowed_cuda_device, resolve_gemma_model_class, resolve_torch_dtype
from vecl._paths import environment_directory
from vecl.evaluation.fisher import accumulate_lora_fisher, compute_lora_snapshot_drift
from vecl.substrate.lora_memory import LoRAMemorySubstrate
from vecl.training.tool_use_loop import snapshot_file_hash

DEFAULT_FISHER_MODEL_ID = "google/gemma-4-31B-it"
DEFAULT_ALLOWED_CUDA_DEVICES = "A100,H100"

DEFAULT_TOOL_CALL_EXAMPLES = (
    {
        "prompt": (
            "Choose the VECL specialist and exact JSON payload for: Analyze FEN "
            "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2 at depth 12."
        ),
        "target": json.dumps(
            {
                "specialist_id": "stockfish",
                "task_type": "chess_eval",
                "input_payload": {
                    "fen": "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",
                    "depth": 12,
                },
            },
            sort_keys=True,
        ),
    },
    {
        "prompt": "Choose the VECL specialist and exact JSON payload for: simplify (x + 1)^2.",
        "target": json.dumps(
            {
                "specialist_id": "sympy",
                "task_type": "symbolic_math",
                "input_payload": {"operation": "simplify", "expression": "(x + 1)^2"},
            },
            sort_keys=True,
        ),
    },
)

DEFAULT_FINAL_ANSWER_EXAMPLES = (
    {
        "prompt": (
            "Using verified context: stockfish claim bestmove=d7d5 eval_cp=3 depth=4 pv=d7d5 c2c3. "
            "Answer the user: What should Black play after 1.d4 e6?"
        ),
        "target": "Black should play 1...d5. Stockfish evaluates the position as essentially equal at +0.03 from White's perspective, with principal variation 1...d5 2.c3.",
    },
    {
        "prompt": (
            "Using verified context: timesfm forecast_sum=420 current_inventory=600. "
            "Answer whether the inventory team needs to reorder."
        ),
        "target": "The forecasted demand is 420 units against 600 units on hand, so no reorder is needed under this simple inventory policy.",
    },
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
        print("No approved CUDA device detected for Fisher eval.", flush=True)
        return 2

    model_cls = resolve_gemma_model_class()
    if model_cls is None:
        print("No supported Gemma 4 model class is available in Transformers.", flush=True)
        return 2

    torch_seed = int(os.environ.get("VECL_FISHER_TORCH_SEED", "1234"))
    torch.manual_seed(torch_seed)
    model_id = os.environ.get("VECL_FISHER_MODEL_ID", DEFAULT_FISHER_MODEL_ID)
    token = os.environ.get("HF_TOKEN")
    dtype = resolve_torch_dtype(torch, os.environ.get("GEMMA_DTYPE", "bfloat16"))
    device_map = os.environ.get("GEMMA_DEVICE_MAP", "auto")
    rank = int(os.environ.get("VECL_FISHER_LORA_RANK", "2"))
    last_n_layers = int(os.environ.get("VECL_FISHER_LAST_N_LAYERS", "2"))
    drift_threshold = float(os.environ.get("VECL_FISHER_DRIFT_THRESHOLD", "1.0"))
    output_dir = environment_directory("VECL_FISHER_OUTPUT_DIR", prefix="vecl-fisher-eval-")

    processor = AutoProcessor.from_pretrained(model_id, token=token)
    model_kwargs: dict[str, Any] = {"device_map": device_map, "token": token}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = model_cls.from_pretrained(model_id, **model_kwargs)
    substrate = LoRAMemorySubstrate.attach(model, rank=rank, last_n_layers=last_n_layers)
    baseline = _restore_baseline_if_configured(substrate, output_dir)
    examples = [*DEFAULT_TOOL_CALL_EXAMPLES, *DEFAULT_FINAL_ANSWER_EXAMPLES]

    gradient_batches = []
    loss_sum = 0.0
    for example in examples:
        model.zero_grad(set_to_none=True)
        loss = _supervised_loss(
            model=model,
            processor=processor,
            prompt=str(example["prompt"]),
            target=str(example["target"]),
            torch=torch,
        )
        loss_sum += float(loss.detach().cpu().item())
        loss.backward()
        gradient_batches.append(substrate.raw_gradients_from_current_grads())

    estimate = accumulate_lora_fisher(
        gradient_batches,
        slot_count=substrate.slot_count,
        dataset_items=examples,
        loss_sum=loss_sum,
        metadata={"model_id": model_id, "rank": rank, "last_n_layers": last_n_layers},
    )
    approved_path = output_dir / "approved-lora-fisher.npz"
    candidate_path = output_dir / "candidate-lora-fisher.npz"
    exceeded_path = output_dir / "candidate-lora-fisher-exceeded.npz"
    fisher_metadata = {
        "sample_count": estimate.sample_count,
        "dataset_hash": estimate.dataset_hash,
        "loss_sum": estimate.loss_sum,
        "model_id": model_id,
        "rank": rank,
        "last_n_layers": last_n_layers,
    }
    substrate.snapshot(
        approved_path,
        fisher_diagonal=estimate.slot_fisher,
        fisher_metadata=fisher_metadata,
    )
    approved_hash = snapshot_file_hash(approved_path)
    substrate.snapshot(candidate_path)
    under = compute_lora_snapshot_drift(
        approved_path, candidate_path, drift_threshold=drift_threshold
    )
    _perturb_first_slot(substrate, torch)
    substrate.snapshot(exceeded_path)
    exceeded = compute_lora_snapshot_drift(approved_path, exceeded_path, drift_threshold=0.0)

    summary = {
        "model_id": model_id,
        "torch_seed": torch_seed,
        "sample_count": estimate.sample_count,
        "slot_count": substrate.slot_count,
        "loss_sum": estimate.loss_sum,
        "finite_fisher": bool(torch.isfinite(torch.as_tensor(estimate.slot_fisher)).all()),
        "positive_fisher_slots": int((estimate.slot_fisher > 0).sum()),
        "approved_snapshot": str(approved_path),
        "approved_snapshot_hash": approved_hash,
        "candidate_snapshot": str(candidate_path),
        "baseline_snapshot": baseline["baseline_snapshot_path"],
        "baseline_snapshot_hash": baseline["baseline_snapshot_hash"],
        "baseline_restored": baseline["baseline_restored"],
        "baseline_source": baseline["baseline_source"],
        "drift_value": under.drift_value,
        "drift_threshold": under.drift_threshold,
        "drift_within_bound": under.drift_within_bound,
        "artificial_over_threshold_rejected": not exceeded.drift_within_bound,
        "gcs_output_uri": os.environ.get("VECL_FISHER_GCS_OUTPUT_URI", ""),
        "outputs_uploaded": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["outputs_uploaded"] = _upload_outputs(
        [approved_path, candidate_path, exceeded_path, summary_path],
        os.environ.get("VECL_FISHER_GCS_OUTPUT_URI", ""),
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _upload_outputs([summary_path], os.environ.get("VECL_FISHER_GCS_OUTPUT_URI", ""))
    print("FISHER_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if _summary_passed(summary) else 1


def _restore_baseline_if_configured(
    substrate: LoRAMemorySubstrate, output_dir: Path
) -> dict[str, Any]:
    baseline_uri = os.environ.get("VECL_FISHER_BASELINE_SNAPSHOT_URI", "").strip()
    baseline_path_raw = os.environ.get("VECL_FISHER_BASELINE_SNAPSHOT_PATH", "").strip()
    expected_hash = os.environ.get("VECL_FISHER_BASELINE_SNAPSHOT_HASH", "").strip()
    if baseline_uri:
        path = output_dir / "baseline-lora.npz"
        _download_gcs_file(baseline_uri, path)
        source = "gcs"
    elif baseline_path_raw:
        path = Path(baseline_path_raw)
        source = "local"
    else:
        return {
            "baseline_snapshot_path": None,
            "baseline_snapshot_hash": None,
            "baseline_restored": False,
            "baseline_source": "none",
        }
    if not path.exists():
        raise ValueError(f"baseline snapshot not found: {path}")
    baseline_hash = snapshot_file_hash(path)
    if expected_hash and baseline_hash != expected_hash:
        raise ValueError("baseline snapshot hash does not match expected hash")
    substrate.restore(path)
    return {
        "baseline_snapshot_path": str(path),
        "baseline_snapshot_hash": baseline_hash,
        "baseline_restored": True,
        "baseline_source": source,
    }


def _supervised_loss(*, model: Any, processor: Any, prompt: str, target: str, torch: Any) -> Any:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    prompt_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    target_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([prompt_inputs["input_ids"], target_ids], dim=-1)
    attention_mask = torch.cat(
        [prompt_inputs["attention_mask"], torch.ones_like(target_ids)], dim=-1
    )
    labels = input_ids.clone()
    labels[:, : prompt_inputs["input_ids"].shape[-1]] = -100
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        labels = labels.to("cuda")
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return output.loss


def _perturb_first_slot(substrate: LoRAMemorySubstrate, torch: Any) -> None:
    slot = substrate._slot_by_id[0]  # noqa: SLF001 - eval deliberately mutates one slot.
    a_row, _b_column = substrate._slot_tensors(slot)  # noqa: SLF001
    with torch.no_grad():
        a_row.add_(torch.full_like(a_row, 1e-3))


def _summary_passed(summary: dict[str, Any]) -> bool:
    upload_required = bool(summary.get("gcs_output_uri"))
    return (
        summary["sample_count"] > 0
        and summary["slot_count"] > 0
        and summary["finite_fisher"]
        and summary["positive_fisher_slots"] > 0
        and summary["drift_within_bound"]
        and summary["artificial_over_threshold_rejected"]
        and bool(summary.get("approved_snapshot_hash"))
        and (not upload_required or bool(summary.get("outputs_uploaded")))
    )


def _download_gcs_file(gcs_uri: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gsutil", "-q", "cp", gcs_uri, str(destination)], check=True)


def _upload_outputs(paths: list[Path], gcs_output_uri: str) -> bool:
    if not paths or not gcs_output_uri:
        return False
    for path in paths:
        subprocess.run(
            ["gsutil", "-q", "cp", str(path), gcs_output_uri.rstrip("/") + "/"], check=True
        )
    return True


if __name__ == "__main__":
    raise SystemExit(main())
