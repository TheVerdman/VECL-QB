#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from vecl.qb._llm_inference import (
    DEFAULT_ROUTING_MODEL_ID,
    RoutingInferenceConfig,
    gemma_route_once,
)
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.stockfish import StockfishSpecialist

try:
    from scripts.routing_eval_gemma_stockfish import ensure_stockfish_18
    from scripts.terraform_eval_gemma import ensure_terraform_cli, run_terraform_eval
except ModuleNotFoundError:
    from routing_eval_gemma_stockfish import ensure_stockfish_18
    from terraform_eval_gemma import ensure_terraform_cli, run_terraform_eval


def main() -> int:
    fixture_path = Path(
        os.environ.get("VECL_TERRAFORM_STOCKFISH_EVAL_FIXTURE", _default_fixture_path())
    )
    entries = json.loads(fixture_path.read_text())
    model_id = os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID)
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is required for the Terraform + Stockfish eval.", flush=True)
        return 2
    max_entries = int(
        os.environ.get("VECL_TERRAFORM_STOCKFISH_EVAL_MAX_ENTRIES", str(len(entries)))
    )
    config = RoutingInferenceConfig(
        model_id=model_id,
        max_new_tokens=int(os.environ.get("VECL_ROUTING_MAX_NEW_TOKENS", "192")),
        dtype=os.environ.get("VECL_ROUTING_DTYPE", "bfloat16"),  # type: ignore[arg-type]
        device_map=os.environ.get("VECL_ROUTING_DEVICE_MAP", "auto"),
        hf_token=os.environ.get("HF_TOKEN"),
    )

    def inference_fn(prompt: str) -> str:
        return gemma_route_once(prompt, config)

    terraform_binary = ensure_terraform_cli()
    stockfish_binary = ensure_stockfish_18()
    artifact_root = Path(
        os.environ.get("VECL_ARTIFACT_STORE", "/tmp/vecl-terraform-stockfish-eval-artifacts")
    )
    stockfish = StockfishSpecialist(
        binary=stockfish_binary,
        working_directory=stockfish_binary.parent,
        artifact_store=ContentAddressedStore(artifact_root / "artifacts"),
    )
    summary = run_terraform_eval(
        entries=entries[:max_entries],
        route_fn=inference_fn,
        answer_fn=inference_fn,
        model_id=model_id,
        artifact_root=artifact_root,
        terraform_binary=terraform_binary,
        stockfish_specialist=stockfish,
        ledger_path=artifact_root / "ledger.sqlite",
    )
    print("TERRAFORM_STOCKFISH_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["plan_chain_success_rate"] < 0.9:
        return 1
    if summary["answer_success_rate"] < 0.9:
        return 1
    if summary["mutation_safe_rate"] < 1.0:
        return 1
    if summary["stockfish_routing_success_rate"] < 0.9:
        return 1
    if summary["out_of_domain_terraform_rate"] > 0.1:
        return 1
    if summary["blocked_specialist_calls"] != 0:
        return 1
    if summary["fallback_events"] != 0:
        return 1
    if summary["persisted_artifact_restore_count"] != summary["artifact_events"]:
        return 1
    return 0


def _default_fixture_path() -> str:
    packaged = Path(__file__).with_name("terraform_stockfish_eval_v0.json")
    if packaged.exists():
        return str(packaged)
    return str(
        Path(__file__).parents[1] / "tests" / "qb" / "fixtures" / "terraform_stockfish_eval_v0.json"
    )


if __name__ == "__main__":
    raise SystemExit(main())
