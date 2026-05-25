#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scripts.routing_eval_gemma_stockfish import ensure_stockfish_18
from vecl.provenance.events import EventType, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb._llm_inference import DEFAULT_ROUTING_MODEL_ID
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.planner import ChainPlan, ChainPlanner, ChainStep
from vecl.qb.prompted_router import PromptedLLMRouter
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.stockfish import StockfishSpecialist


class StockfishFormatterSpecialist(Specialist):
    specialist_id = "stockfish-formatter"

    def __init__(self, store: ContentAddressedStore) -> None:
        self.store = store

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "chess_eval"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        upstream = request.input_payload["upstream"]["analyze"]
        claim_text = upstream["claims"][0]["claim_text"]
        report = {
            "source_step": "analyze",
            "source_claim_id": upstream["claim_ids"][0],
            "analysis": claim_text,
            "artifact_ids": upstream["artifact_ids"],
        }
        record = self.store.write_text(
            json.dumps(report, sort_keys=True),
            producer_specialist_id=self.specialist_id,
            producer_version="v1",
            input_hash=stable_hash(report),
            output_format="json",
            parent_event_id=str(request.provenance_context["parent_event_id"]),
        )
        claim = SpecialistClaim(
            "claim-stockfish-chain-report-" + stable_hash(report)[:12],
            self.specialist_id,
            f"stockfish_chain_report={json.dumps(report, sort_keys=True)}",
            "chess_report",
            0.85,
            evidence_ids=[stable_hash({"upstream_claim_id": upstream["claim_ids"][0]})],
            artifact_ids=[record.artifact_id],
        )
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [claim],
            request.tenant_id,
            cost_metadata={"artifact_records": [record.to_payload()]},
        )


def main() -> int:
    fixture_path = Path(
        os.environ.get("VECL_CHAIN_EVAL_FIXTURE", Path(__file__).with_name("routing_eval_v0.json"))
    )
    entries = json.loads(fixture_path.read_text())
    model_id = os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID)
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is required for the Gemma chain eval.", flush=True)
        return 2

    max_entries = int(os.environ.get("VECL_CHAIN_EVAL_MAX_ENTRIES", "30"))
    entries = entries[:max_entries]
    stockfish_binary = ensure_stockfish_18()
    artifact_root = Path(os.environ.get("VECL_ARTIFACT_STORE", "/tmp/vecl-chain-artifacts"))
    store = ContentAddressedStore(artifact_root)
    ledger = ProvenanceLedger()
    router = PromptedLLMRouter(ledger=ledger, model_id=model_id)
    stockfish = StockfishSpecialist(
        binary=stockfish_binary,
        working_directory=stockfish_binary.parent,
        artifact_store=store,
    )
    formatter = StockfishFormatterSpecialist(store)
    router.register_specialist(stockfish_card(), stockfish)
    router.register_specialist(formatter_card(), formatter)
    planner = ChainPlanner({"chess_eval": stockfish_chain_plan()})
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"chess_eval", "chess_report"}),
    )

    chess_total = 0
    chains_started = 0
    chains_passed = 0
    out_total = 0
    out_routed_to_stockfish = 0
    for index, entry in enumerate(entries, start=1):
        route_request = SpecialistRequest(
            f"route-{index}",
            "chain-eval",
            entry["task_type"],
            entry["input_payload"],
            {},
            {"parent_event_id": f"eval-{index}"},
        )
        routed = router.route(route_request, max_specialists=1)
        routed_ids = [specialist.specialist_id for specialist in routed]
        if entry["expected_specialist_id"] == "stockfish":
            chess_total += 1
            result = None
            if routed_ids == ["stockfish"]:
                plan = planner.plan(route_request, router.specialists)
                result = orchestrator.run_task(
                    tenant_id="chain-eval",
                    task_type=entry["task_type"],
                    input_payload=entry["input_payload"],
                    chain_plan=plan,
                )
                chains_started += 1
                chains_passed += int(result.verification_status == "PASSED")
            row = {
                "id": entry["id"],
                "index": index,
                "expected": "stockfish",
                "routed_ids": routed_ids,
                "chain_status": result.verification_status if result else "NOT_RUN",
            }
        else:
            out_total += 1
            routed_to_stockfish = "stockfish" in routed_ids
            out_routed_to_stockfish += int(routed_to_stockfish)
            row = {
                "id": entry["id"],
                "index": index,
                "expected": None,
                "routed_ids": routed_ids,
                "chain_status": "NOT_RUN",
            }
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = chain_eval_summary(
        ledger=ledger,
        model_id=model_id,
        stockfish_binary=stockfish_binary,
        chess_total=chess_total,
        chains_started=chains_started,
        chains_passed=chains_passed,
        out_total=out_total,
        out_routed_to_stockfish=out_routed_to_stockfish,
    )
    print("CHAIN_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["chess_chain_success_rate"] < 0.9:
        return 1
    if summary["out_of_domain_stockfish_rate"] > 0.1:
        return 1
    if summary["chain_step_completed_events"] != summary["chain_started_events"] * 2:
        return 1
    if summary["artifact_events"] != summary["chain_started_events"] * 2:
        return 1
    if summary["chain_aborted_events"] != 0 or summary["fallback_events"] != 0:
        return 1
    return 0


def stockfish_chain_plan() -> ChainPlan:
    return ChainPlan(
        "stockfish-format-chain",
        "chess_eval",
        (
            ChainStep(
                "analyze", "stockfish", parameters={"depth": 4}, expected_artifact_type="txt"
            ),
            ChainStep(
                "format",
                "stockfish-formatter",
                inputs_from=("analyze",),
                expected_artifact_type="json",
            ),
        ),
    )


def stockfish_card() -> SpecialistCard:
    return SpecialistCard(
        "stockfish",
        {"chess_eval"},
        {"description": "Analyzes chess positions from FEN and returns best moves."},
        cost_hint=1.0,
        latency_hint=1.0,
        version="stockfish-18",
        effective_trust=0.9,
        description="Analyze chess positions with Stockfish.",
    )


def formatter_card() -> SpecialistCard:
    return SpecialistCard(
        "stockfish-formatter",
        {"chess_eval"},
        {"description": "Formats Stockfish analysis into a JSON chain report."},
        cost_hint=0.1,
        latency_hint=0.1,
        version="v1",
        effective_trust=0.8,
        description="Format Stockfish chain output as JSON.",
    )


def chain_eval_summary(
    *,
    ledger: ProvenanceLedger,
    model_id: str,
    stockfish_binary: Path,
    chess_total: int,
    chains_started: int,
    chains_passed: int,
    out_total: int,
    out_routed_to_stockfish: int,
) -> dict[str, Any]:
    decision_events = ledger.find_by_type(EventType.LLM_ROUTING_DECIDED)
    fallback_events = ledger.find_by_type(EventType.ROUTING_FALLBACK)
    chain_started_events = ledger.find_by_type(EventType.CHAIN_STARTED)
    chain_completed_events = ledger.find_by_type(EventType.CHAIN_STEP_COMPLETED)
    chain_aborted_events = ledger.find_by_type(EventType.CHAIN_ABORTED)
    artifact_events = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    return {
        "model_id": model_id,
        "stockfish_binary": str(stockfish_binary),
        "chess_total": chess_total,
        "chain_started_events": len(chain_started_events),
        "chain_step_completed_events": len(chain_completed_events),
        "chain_aborted_events": len(chain_aborted_events),
        "artifact_events": len(artifact_events),
        "chess_chain_success_rate": chains_passed / chess_total if chess_total else 0.0,
        "out_of_domain_stockfish_rate": out_routed_to_stockfish / out_total if out_total else 0.0,
        "decision_events": len(decision_events),
        "fallback_events": len(fallback_events),
        "chains_started": chains_started,
        "chains_passed": chains_passed,
        "total_entries": chess_total + out_total,
    }


if __name__ == "__main__":
    raise SystemExit(main())
