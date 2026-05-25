#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vecl.evaluation.evaluators import SummaryEvaluator
from vecl.evaluation.release import approve_release_report, save_release_report
from vecl.evaluation.types import (
    EvaluationResult,
    MetricThreshold,
    ReleaseEvaluationReport,
    thresholds_from_payloads,
)
from vecl.provenance.ledger import SqliteProvenanceLedger
from vecl.qb.model_driver import ModelDriver, resolve_model_driver
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.stockfish import StockfishSpecialist
from vecl.specialists.sympy_specialist import SymPySpecialist
from vecl.specialists.timesfm_specialist import LocalTimesFMRunner, TimesFMRunner

try:
    from scripts.ethics_eval_gemma import DEFAULT_CASES, run_ethics_eval
    from scripts.model_driver_eval import run_model_driver_eval
    from scripts.routing_eval_gemma_phase7 import run_phase7_routing_eval
    from scripts.routing_eval_gemma_stockfish import ensure_stockfish_18
    from scripts.terraform_eval_gemma import ensure_terraform_cli, run_terraform_eval
    from scripts.timesfm_demand_eval_gemma import run_demand_eval
    from scripts.tool_call_payload_eval import default_cards, run_tool_call_payload_eval
except ModuleNotFoundError:
    from ethics_eval_gemma import DEFAULT_CASES, run_ethics_eval
    from model_driver_eval import run_model_driver_eval
    from routing_eval_gemma_phase7 import run_phase7_routing_eval
    from routing_eval_gemma_stockfish import ensure_stockfish_18
    from terraform_eval_gemma import ensure_terraform_cli, run_terraform_eval
    from timesfm_demand_eval_gemma import run_demand_eval
    from tool_call_payload_eval import default_cards, run_tool_call_payload_eval


EVAL_ORDER = (
    "gemma_model_driver_eval",
    "gemma_tool_call_payload_eval",
    "phase7_sympy_blast_routing_eval",
    "phase6b_ethics_eval",
    "phase8b2_terraform_stockfish_persistence_eval",
    "phase7b_timesfm_demand_eval",
)

SUMMARY_PREFIXES = {
    "gemma_model_driver_eval": "MODEL_DRIVER_EVAL_SUMMARY ",
    "gemma_tool_call_payload_eval": "TOOL_CALL_PAYLOAD_EVAL_SUMMARY ",
    "phase7_sympy_blast_routing_eval": "PHASE7_ROUTING_EVAL_SUMMARY ",
    "phase6b_ethics_eval": "ETHICS_EVAL_SUMMARY ",
    "phase8b2_terraform_stockfish_persistence_eval": "TERRAFORM_STOCKFISH_EVAL_SUMMARY ",
    "phase7b_timesfm_demand_eval": "TIMESFM_DEMAND_EVAL_SUMMARY ",
}


@dataclass(frozen=True)
class AggregateEvalSpec:
    name: str
    category: str
    thresholds: tuple[MetricThreshold, ...]


def main(
    argv: list[str] | None = None,
    *,
    base_dir: Path | None = None,
    artifact_root: Path | None = None,
) -> int:
    del argv
    base = base_dir or Path(__file__).resolve().parent
    artifact_root = artifact_root or Path(
        os.environ.get("VECL_RELEASE_EVAL_ARTIFACT_ROOT", "/tmp/vecl-release-eval-artifacts")
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("VECL_MODEL_DRIVER", "gemma")
    candidate_id = os.environ.get("VECL_RELEASE_CANDIDATE_ID", "vertex-gemma-release-eval")
    manifest_path = base / "phase9a-vertex-gemma.json"
    report_path = artifact_root / "release-report.json"
    ledger_path = artifact_root / "release-ledger.sqlite"

    report = run_aggregate_release_eval(
        candidate_id=candidate_id,
        base_dir=base,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
    )
    save_release_report(report, report_path)
    print(json.dumps(report.to_payload(), sort_keys=True), flush=True)
    if not report.passed:
        return 1
    with SqliteProvenanceLedger(ledger_path) as ledger:
        event = approve_release_report(report, ledger, "vertex-release-eval")
    print(json.dumps(event.to_dict(), sort_keys=True), flush=True)
    return 0


def run_aggregate_release_eval(
    *,
    candidate_id: str,
    base_dir: Path,
    artifact_root: Path,
    manifest_path: Path,
    driver: ModelDriver | None = None,
    inference_fn: Callable[[str], str] | None = None,
    timesfm_runner: TimesFMRunner | None = None,
) -> ReleaseEvaluationReport:
    specs = load_aggregate_specs(manifest_path)
    shared_driver = driver or resolve_model_driver("gemma")

    def infer(prompt: str) -> str:
        if inference_fn is not None:
            return inference_fn(prompt)
        return shared_driver.synthesize(prompt).raw_text

    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "gemma_model_driver_eval": lambda: _run_model_driver_eval(base_dir, shared_driver),
        "gemma_tool_call_payload_eval": lambda: _run_tool_call_payload_eval(
            base_dir, artifact_root, shared_driver
        ),
        "phase7_sympy_blast_routing_eval": lambda: _run_phase7_eval(
            base_dir, artifact_root, infer, shared_driver.model_id
        ),
        "phase6b_ethics_eval": lambda: _run_ethics_eval(infer, shared_driver.model_id),
        "phase8b2_terraform_stockfish_persistence_eval": lambda: _run_terraform_stockfish_eval(
            base_dir, artifact_root, infer, shared_driver.model_id
        ),
        "phase7b_timesfm_demand_eval": lambda: _run_timesfm_eval(
            base_dir,
            artifact_root,
            infer,
            shared_driver.model_id,
            timesfm_runner=timesfm_runner,
        ),
    }

    results: list[EvaluationResult] = []
    for name in EVAL_ORDER:
        spec = specs[name]
        try:
            summary = runners[name]()
        except Exception as exc:
            result = EvaluationResult(
                name=spec.name,
                category=spec.category,
                passed=False,
                summary={"execution_mode": "in_process_shared_model_driver", "error": repr(exc)},
                reasons=(f"{name} raised {type(exc).__name__}: {exc}",),
                thresholds=spec.thresholds,
            )
        else:
            summary = {"execution_mode": "in_process_shared_model_driver", **summary}
            print(SUMMARY_PREFIXES[name] + json.dumps(summary, sort_keys=True), flush=True)
            result = SummaryEvaluator(
                name=spec.name,
                category=spec.category,
                summary=summary,
                thresholds=spec.thresholds,
            ).evaluate()
        results.append(result)
    return ReleaseEvaluationReport(candidate_id=candidate_id, results=tuple(results))


def load_aggregate_specs(manifest_path: Path) -> dict[str, AggregateEvalSpec]:
    payload = json.loads(manifest_path.read_text())
    specs: dict[str, AggregateEvalSpec] = {}
    for config in payload.get("evaluators", []):
        name = str(config["name"])
        specs[name] = AggregateEvalSpec(
            name=name,
            category=str(config.get("category") or "regression"),
            thresholds=thresholds_from_payloads(config.get("thresholds", [])),
        )
    missing = [name for name in EVAL_ORDER if name not in specs]
    if missing:
        raise ValueError(f"aggregate manifest missing evaluator specs: {missing}")
    return specs


def build_report_from_summaries(
    *,
    candidate_id: str,
    specs: Mapping[str, AggregateEvalSpec],
    summaries: Mapping[str, dict[str, Any]],
) -> ReleaseEvaluationReport:
    results = []
    for name in EVAL_ORDER:
        spec = specs[name]
        results.append(
            SummaryEvaluator(
                name=spec.name,
                category=spec.category,
                summary=dict(summaries[name]),
                thresholds=spec.thresholds,
            ).evaluate()
        )
    return ReleaseEvaluationReport(candidate_id=candidate_id, results=tuple(results))


def _run_model_driver_eval(base_dir: Path, driver: ModelDriver) -> dict[str, Any]:
    entries = json.loads((base_dir / "model_driver_eval_v0.json").read_text())
    return run_model_driver_eval(entries=entries, driver=driver)


def _run_tool_call_payload_eval(
    base_dir: Path, artifact_root: Path, driver: ModelDriver
) -> dict[str, Any]:
    entries = json.loads((base_dir / "tool_call_payload_eval_v0.json").read_text())
    store = ContentAddressedStore(artifact_root / "tool-call-artifacts")
    specialists = {
        "stockfish": _stockfish_specialist(store),
        "sympy": SymPySpecialist(artifact_store=store),
    }
    cards = {card.specialist_id: card for card in default_cards()}
    return run_tool_call_payload_eval(
        entries=entries,
        driver=driver,
        cards=cards,
        specialists=specialists,
    )


def _run_phase7_eval(
    base_dir: Path, artifact_root: Path, inference_fn: Callable[[str], str], model_id: str
) -> dict[str, Any]:
    entries = json.loads((base_dir / "phase7_routing_eval_v0.json").read_text())
    return run_phase7_routing_eval(
        entries=entries,
        inference_fn=inference_fn,
        model_id=model_id,
        artifact_root=artifact_root / "phase7",
    )


def _run_ethics_eval(inference_fn: Callable[[str], str], model_id: str) -> dict[str, Any]:
    max_cases = int(os.environ.get("VECL_ETHICS_EVAL_MAX_CASES", str(len(DEFAULT_CASES))))
    return run_ethics_eval(
        cases=DEFAULT_CASES[:max_cases],
        inference_fn=inference_fn,
        model_id=model_id,
    )


def _run_terraform_stockfish_eval(
    base_dir: Path, artifact_root: Path, inference_fn: Callable[[str], str], model_id: str
) -> dict[str, Any]:
    entries = json.loads((base_dir / "terraform_stockfish_eval_v0.json").read_text())
    max_entries = int(
        os.environ.get("VECL_TERRAFORM_STOCKFISH_EVAL_MAX_ENTRIES", str(len(entries)))
    )
    terraform_binary = ensure_terraform_cli()
    stockfish_binary = ensure_stockfish_18()
    eval_root = artifact_root / "terraform-stockfish"
    stockfish = StockfishSpecialist(
        binary=stockfish_binary,
        working_directory=Path(stockfish_binary).parent,
        artifact_store=ContentAddressedStore(eval_root / "stockfish-artifacts"),
    )
    return run_terraform_eval(
        entries=entries[:max_entries],
        route_fn=inference_fn,
        answer_fn=inference_fn,
        model_id=model_id,
        artifact_root=eval_root,
        terraform_binary=terraform_binary,
        stockfish_specialist=stockfish,
        ledger_path=eval_root / "ledger.sqlite",
    )


def _run_timesfm_eval(
    base_dir: Path,
    artifact_root: Path,
    inference_fn: Callable[[str], str],
    model_id: str,
    *,
    timesfm_runner: TimesFMRunner | None,
) -> dict[str, Any]:
    entries = json.loads((base_dir / "timesfm_demand_eval_v0.json").read_text())
    max_entries = int(os.environ.get("VECL_TIMESFM_DEMAND_EVAL_MAX_ENTRIES", "6"))
    runner = timesfm_runner or LocalTimesFMRunner(
        model_id=os.environ.get("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
    )
    return run_demand_eval(
        entries=entries[:max_entries],
        runner=runner,
        route_fn=inference_fn,
        answer_fn=inference_fn,
        model_id=model_id,
        artifact_root=artifact_root / "timesfm-demand",
    )


def _stockfish_specialist(store: ContentAddressedStore) -> StockfishSpecialist:
    try:
        return StockfishSpecialist(artifact_store=store)
    except FileNotFoundError:
        stockfish_binary = ensure_stockfish_18()
        return StockfishSpecialist(
            binary=stockfish_binary,
            working_directory=Path(stockfish_binary).parent,
            artifact_store=store,
        )


if __name__ == "__main__":
    raise SystemExit(main())
