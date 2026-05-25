from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb._llm_inference import DEFAULT_ROUTING_MODEL_ID
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.prompted_router import (
    PromptedLLMRouter,
    parse_routing_response,
    serialize_specialist_card,
)
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import MockSpecialist, SpecialistClaim, SpecialistRequest
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.stockfish import StockfishSpecialist

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "routing_eval_v0.json"
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _request(task_type: str = "chess_eval") -> SpecialistRequest:
    return SpecialistRequest(
        "req-router",
        "tenant",
        task_type,
        {"query": "Which chess engine move is best?", "fen": STARTING_FEN},
        {},
        {"parent_event_id": "evt-request"},
    )


def _stockfish_card() -> SpecialistCard:
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


def test_parse_top_level_json_object() -> None:
    decision = parse_routing_response(
        '{"tool":"stockfish","confidence":0.93,"reasoning":"This is a chess position."}'
    )

    assert decision.specialist_id == "stockfish"
    assert decision.confidence == 0.93
    assert "chess" in decision.reasoning


def test_parse_function_call_style_response() -> None:
    decision = parse_routing_response(
        'select_specialist(tool="stockfish", confidence=0.91, reasoning="best chess tool")'
    )

    assert decision.specialist_id == "stockfish"
    assert decision.confidence == 0.91
    assert decision.reasoning == "best chess tool"


def test_parse_fenced_json_block() -> None:
    decision = parse_routing_response(
        'Here is the route:\n```json\n{"tool": null, "confidence": 0.2, "reasoning": "No matching tool."}\n```'
    )

    assert decision.specialist_id is None
    assert decision.confidence == 0.2


def test_serializer_outputs_structured_card_json() -> None:
    payload = json.loads(serialize_specialist_card(_stockfish_card()))

    assert payload["id"] == "stockfish"
    assert payload["supported_task_types"] == ["chess_eval"]
    assert payload["description"] == "Analyze chess positions with Stockfish."
    assert payload["cost_hint"] == 1.0
    assert payload["latency_hint"] == 1.0


def test_prompted_router_records_decision_before_specialist_invocation() -> None:
    ledger = ProvenanceLedger()
    specialist = MockSpecialist(
        "stockfish",
        {"chess_eval"},
        [
            SpecialistClaim(
                "c-stockfish",
                "stockfish",
                "bestmove=d2d4",
                "chess_eval",
                0.9,
                evidence_ids=["fen"],
                source_ids=["stockfish-v18"],
            )
        ],
    )
    router = PromptedLLMRouter(
        ledger=ledger,
        inference_fn=lambda _prompt: (
            '{"tool":"stockfish","confidence":0.94,"reasoning":"Chess query."}'
        ),
        model_id="mock-gemma",
    )
    router.register_specialist(_stockfish_card(), specialist)
    orchestrator = QBOrchestrator(
        router, ledger, policy=VerificationPolicy(required_claim_types={"chess_eval"})
    )

    result = orchestrator.run_task(
        tenant_id="tenant",
        task_type="chess_eval",
        input_payload={"query": "Analyze this chess position.", "fen": STARTING_FEN},
    )

    event_types = [event.event_type for event in ledger.events()]
    assert result.verification_status == "PASSED"
    assert event_types[:3] == [
        EventType.EVIDENCE_INGESTED,
        EventType.LLM_ROUTING_DECIDED,
        EventType.REPLAY_BATCH_PREPARED,
    ]
    decision = ledger.find_by_type(EventType.LLM_ROUTING_DECIDED)[0]
    assert decision.parent_event_ids == [result.provenance_event_ids[0]]
    assert decision.payload["chosen_specialist_id"] == "stockfish"
    assert decision.payload["model_id"] == "mock-gemma"


def test_parse_failure_falls_back_to_rule_router_and_records_event() -> None:
    ledger = ProvenanceLedger()
    specialist = MockSpecialist("stockfish", {"chess_eval"})
    router = PromptedLLMRouter(
        ledger=ledger,
        inference_fn=lambda _prompt: "not parseable",
        model_id="mock-gemma",
    )
    router.register_specialist(_stockfish_card(), specialist)

    routed = router.route(_request())

    fallback = ledger.find_by_type(EventType.ROUTING_FALLBACK)[0]
    assert routed == [specialist]
    assert fallback.payload["fallback_specialist_id"] == "stockfish"
    assert "could not parse" in str(fallback.payload["parse_failure_reason"])
    assert fallback.parent_event_ids == ["evt-request"]


def test_inference_failure_is_not_silent_fallback() -> None:
    ledger = ProvenanceLedger()
    specialist = MockSpecialist("stockfish", {"chess_eval"})

    def fail(_prompt: str) -> str:
        raise RuntimeError("model load failed")

    router = PromptedLLMRouter(ledger=ledger, inference_fn=fail, model_id="mock-gemma")
    router.register_specialist(_stockfish_card(), specialist)

    with pytest.raises(RuntimeError, match="model load failed"):
        router.route(_request())

    assert ledger.find_by_type(EventType.ROUTING_FALLBACK) == []


@pytest.mark.skipif(
    os.environ.get("VECL_RUN_GEMMA_TESTS") != "1"
    or os.environ.get("VECL_RUN_STOCKFISH_TESTS") != "1",
    reason="set VECL_RUN_GEMMA_TESTS=1 and VECL_RUN_STOCKFISH_TESTS=1 for real routing eval",
)
def test_opt_in_gemma_stockfish_routing_eval(tmp_path: Path) -> None:
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN is required for Gemma routing eval")
    entries = json.loads(FIXTURE_PATH.read_text())
    ledger = ProvenanceLedger()
    model_id = os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID)
    router = PromptedLLMRouter(ledger=ledger, model_id=model_id)
    specialist = StockfishSpecialist(artifact_store=ContentAddressedStore(tmp_path))
    router.register_specialist(_stockfish_card(), specialist)
    orchestrator = QBOrchestrator(
        router, ledger, policy=VerificationPolicy(required_claim_types={"chess_eval"})
    )

    chess_total = 0
    chess_stockfish = 0
    out_total = 0
    out_stockfish = 0
    for entry in entries:
        result = orchestrator.run_task(
            tenant_id="routing-eval",
            task_type=entry["task_type"],
            input_payload=entry["input_payload"],
        )
        routed_to_stockfish = "stockfish" in result.specialist_ids
        if entry["expected_specialist_id"] == "stockfish":
            chess_total += 1
            chess_stockfish += int(routed_to_stockfish)
        else:
            out_total += 1
            out_stockfish += int(routed_to_stockfish)

    decision_events = ledger.find_by_type(EventType.LLM_ROUTING_DECIDED)
    fallback_events = ledger.find_by_type(EventType.ROUTING_FALLBACK)
    assert chess_stockfish / chess_total >= 0.9
    assert out_stockfish / out_total <= 0.1
    assert len(decision_events) == len(entries)
    assert fallback_events == []
    for event in decision_events:
        assert set(event.payload) >= {
            "query_hash",
            "chosen_specialist_id",
            "model_id",
            "reasoning",
            "confidence",
        }
    for event in fallback_events:
        assert event.payload["parse_failure_reason"]
