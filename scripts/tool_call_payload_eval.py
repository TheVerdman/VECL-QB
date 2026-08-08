#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl._paths import environment_directory  # noqa: E402
from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash  # noqa: E402
from vecl.provenance.ledger import ProvenanceLedger  # noqa: E402
from vecl.qb.model_driver import ModelDriver, resolve_model_driver  # noqa: E402
from vecl.qb.router import SpecialistCard  # noqa: E402
from vecl.qb.specialist import Specialist, SpecialistRequest  # noqa: E402
from vecl.qb.tool_call import (  # noqa: E402
    ToolCallValidationConfig,
    execute_tool_call,
    propose_tool_call,
)
from vecl.specialists.artifacts import ContentAddressedStore  # noqa: E402

SYNTHESIS_PROMPT_TEMPLATE = """You are VECL-QB's final answer writer.
Use only the verified specialist responses and artifact ids below.
Return a concise final answer. Do not invent tool output.

User request:
{query}

Selected specialist:
{specialist_id}

Specialist responses:
{responses}

Artifact ids:
{artifact_ids}
"""


def main() -> int:
    driver = resolve_model_driver()
    store = ContentAddressedStore(
        environment_directory("VECL_ARTIFACT_STORE", prefix="vecl-tool-call-eval-artifacts-")
    )
    from vecl.specialists.sympy_specialist import SymPySpecialist

    specialists: dict[str, Specialist] = {
        "stockfish": _stockfish_specialist(store),
        "sympy": SymPySpecialist(artifact_store=store),
    }
    cards = {card.specialist_id: card for card in default_cards()}
    summary = run_tool_call_payload_eval(
        entries=json.loads(Path(_default_fixture_path()).read_text()),
        driver=driver,
        cards=cards,
        specialists=specialists,
    )
    print("TOOL_CALL_PAYLOAD_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["proposal_success_rate"] < 1.0:
        return 1
    if summary["validation_success_rate"] < 1.0:
        return 1
    if summary["execution_success_rate"] < 1.0:
        return 1
    if summary["synthesis_success_rate"] < 1.0:
        return 1
    return 0


def run_tool_call_payload_eval(
    *,
    entries: Sequence[dict[str, Any]],
    driver: ModelDriver,
    cards: Mapping[str, SpecialistCard],
    specialists: Mapping[str, Specialist],
    terraform_config_dir: str | Path | None = None,
) -> dict[str, Any]:
    ledger = ProvenanceLedger()
    rows: list[dict[str, Any]] = []
    proposal_success = 0
    validation_success = 0
    execution_success = 0
    synthesis_success = 0
    for index, entry in enumerate(entries, start=1):
        request_event = ledger.append(
            ProvenanceEvent(
                EventType.EVIDENCE_INGESTED,
                "tool-call-eval",
                "tool-call-eval",
                {"case_id": entry["id"], "prompt_hash": stable_hash(entry["prompt"])},
            )
        )
        request = SpecialistRequest(
            f"tool-call-eval-{index}",
            "tool-call-eval",
            str(entry.get("task_type_hint") or "unknown"),
            {"query": entry["prompt"]},
            {},
            {"parent_event_id": request_event.event_id},
        )
        try:
            proposal, model_response = propose_tool_call(driver, request, list(cards.values()))
            proposal_ok = (
                proposal.specialist_id == entry["expected_specialist_id"]
                and proposal.task_type == entry["expected_task_type"]
            )
        except Exception as exc:
            rows.append(
                {
                    "id": entry["id"],
                    "proposal_status": "FAILED",
                    "reason": str(exc),
                }
            )
            continue
        proposal_success += int(proposal_ok)
        run = execute_tool_call(
            proposal=proposal,
            model_response=model_response,
            cards=cards,
            specialists=specialists,
            ledger=ledger,
            config=ToolCallValidationConfig(
                tenant_id="tool-call-eval",
                request_id=request.request_id,
                provenance_context={"parent_event_id": request_event.event_id},
                terraform_config_dir=terraform_config_dir,
            ),
        )
        validation_ok = run.validated is not None
        validation_success += int(validation_ok)
        execution_ok = run.execution is not None and not run.execution.aborted
        execution_success += int(execution_ok)
        synthesis_response = driver.synthesize(
            build_synthesis_prompt(entry["prompt"], run.validated, run.execution)
        )
        synthesis_ok = bool(synthesis_response.raw_text.strip()) and execution_ok
        synthesis_success += int(synthesis_ok)
        rows.append(
            {
                "id": entry["id"],
                "proposal_status": "PASSED" if proposal_ok else "FAILED",
                "validation_status": "PASSED" if validation_ok else "FAILED",
                "execution_status": "PASSED" if execution_ok else "FAILED",
                "synthesis_status": "PASSED" if synthesis_ok else "FAILED",
                "specialist_id": proposal.specialist_id,
                "task_type": proposal.task_type,
                "input_payload": proposal.input_payload,
                "warnings": list(run.warnings),
                "final_answer": synthesis_response.raw_text,
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    return {
        "provider": driver.provider,
        "model_id": driver.model_id,
        "total_entries": len(entries),
        "proposal_success": proposal_success,
        "validation_success": validation_success,
        "execution_success": execution_success,
        "synthesis_success": synthesis_success,
        "proposal_success_rate": _rate(proposal_success, len(entries)),
        "validation_success_rate": _rate(validation_success, len(entries)),
        "execution_success_rate": _rate(execution_success, len(entries)),
        "synthesis_success_rate": _rate(synthesis_success, len(entries)),
        "tool_call_events": len(ledger.find_by_type(EventType.LLM_TOOL_CALL_PROPOSED)),
        "artifact_events": len(ledger.find_by_type(EventType.ARTIFACT_PRODUCED)),
        "rows": rows,
    }


def build_synthesis_prompt(query: str, validated: object | None, execution: object | None) -> str:
    specialist_id = getattr(validated, "specialist_id", None)
    responses = []
    artifact_ids: list[str] = []
    if execution is not None:
        responses = [
            {
                "specialist_id": response.specialist_id,
                "claims": [claim.claim_text for claim in response.claims],
            }
            for response in getattr(execution, "responses", [])
        ]
        for response in getattr(execution, "responses", []):
            for claim in response.claims:
                artifact_ids.extend(claim.artifact_ids)
    return SYNTHESIS_PROMPT_TEMPLATE.format(
        query=query,
        specialist_id=specialist_id,
        responses=json.dumps(responses, sort_keys=True),
        artifact_ids=json.dumps(sorted(set(artifact_ids))),
    )


def default_cards() -> list[SpecialistCard]:
    return [
        SpecialistCard(
            "stockfish",
            {"chess_eval"},
            {"description": "Analyze chess positions from FEN and depth."},
            cost_hint=1.0,
            latency_hint=1.0,
            version="stockfish-18",
            effective_trust=0.9,
            description="Analyze chess positions with Stockfish 18.",
        ),
        SpecialistCard(
            "sympy",
            {"symbolic_math", "sequence_math"},
            {"description": "Solve, simplify, factor, integrate, differentiate, and plot math."},
            cost_hint=0.2,
            latency_hint=0.3,
            version="sympy-1.14",
            effective_trust=0.95,
            description="Symbolic mathematics with SymPy.",
        ),
    ]


def _stockfish_specialist(store: ContentAddressedStore) -> Specialist:
    from vecl.specialists.stockfish import StockfishSpecialist

    try:
        return StockfishSpecialist(artifact_store=store)
    except FileNotFoundError:
        try:
            from scripts.routing_eval_gemma_stockfish import ensure_stockfish_18
        except ModuleNotFoundError:
            from routing_eval_gemma_stockfish import ensure_stockfish_18

        stockfish_binary = ensure_stockfish_18()
        return StockfishSpecialist(
            binary=stockfish_binary,
            working_directory=stockfish_binary.parent,
            artifact_store=store,
        )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _default_fixture_path() -> str:
    env_path = os.environ.get("VECL_TOOL_CALL_PAYLOAD_EVAL_FIXTURE")
    if env_path:
        return env_path
    packaged = Path(__file__).with_name("tool_call_payload_eval_v0.json")
    if packaged.exists():
        return str(packaged)
    return str(
        Path(__file__).parents[1] / "tests" / "qb" / "fixtures" / "tool_call_payload_eval_v0.json"
    )


if __name__ == "__main__":
    raise SystemExit(main())
