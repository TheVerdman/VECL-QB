from __future__ import annotations

import json
from pathlib import Path

from scripts.chain_eval_gemma_stockfish import (
    StockfishFormatterSpecialist,
    chain_eval_summary,
    formatter_card,
    stockfish_card,
    stockfish_chain_plan,
)
from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.specialist import SpecialistRequest
from vecl.specialists.artifacts import ContentAddressedStore


def test_stockfish_chain_plan_shape() -> None:
    plan = stockfish_chain_plan()

    assert plan.plan_id == "stockfish-format-chain"
    assert [step.step_id for step in plan.steps] == ["analyze", "format"]
    assert plan.steps[1].inputs_from == ("analyze",)
    assert stockfish_card().specialist_id == "stockfish"
    assert formatter_card().specialist_id == "stockfish-formatter"


def test_formatter_writes_chain_report_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    formatter = StockfishFormatterSpecialist(ContentAddressedStore(tmp_path))
    request = SpecialistRequest(
        "req:format",
        "tenant",
        "chess_eval",
        {
            "upstream": {
                "analyze": {
                    "claim_ids": ["claim-stockfish"],
                    "claims": [
                        {
                            "claim_id": "claim-stockfish",
                            "claim_type": "chess_eval",
                            "claim_text": "bestmove=e2e4",
                            "confidence": 0.9,
                            "artifact_ids": ["artifact-stockfish"],
                        }
                    ],
                    "artifact_ids": ["artifact-stockfish"],
                }
            }
        },
        {},
        {"parent_event_id": "evt-step-started"},
    )

    response = formatter.run(request)

    assert response.refusal_or_error is None
    assert response.claims[0].claim_type == "chess_report"
    record = response.cost_metadata["artifact_records"][0]  # type: ignore[index]
    assert record["producer_specialist_id"] == "stockfish-formatter"
    payload = json.loads(Path(str(record["output_path"])).read_text())
    assert payload["source_claim_id"] == "claim-stockfish"


def test_chain_eval_summary_counts_phase4_events() -> None:
    ledger = ProvenanceLedger()
    parent = ledger.append(ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant", "tester", {}))
    chain = ledger.append(
        ProvenanceEvent(
            EventType.CHAIN_STARTED,
            "tenant",
            "chain-executor",
            {"request_id": "req", "plan_id": "p"},
            parent_event_ids=[parent.event_id],
        )
    )
    for step_id in ["analyze", "format"]:
        completed = ledger.append(
            ProvenanceEvent(
                EventType.CHAIN_STEP_COMPLETED,
                "tenant",
                "chain-executor",
                {"step_id": step_id},
                parent_event_ids=[chain.event_id],
            )
        )
        ledger.append(
            ProvenanceEvent(
                EventType.ARTIFACT_PRODUCED,
                "tenant",
                "tester",
                {"artifact_id": f"artifact-{step_id}"},
                parent_event_ids=[completed.event_id],
            )
        )
    ledger.append(
        ProvenanceEvent(
            EventType.LLM_ROUTING_DECIDED,
            "tenant",
            "router",
            {"chosen_specialist_id": "stockfish"},
            parent_event_ids=[parent.event_id],
        )
    )

    summary = chain_eval_summary(
        ledger=ledger,
        model_id="mock",
        stockfish_binary=Path("/tmp/stockfish"),
        chess_total=1,
        chains_started=1,
        chains_passed=1,
        out_total=1,
        out_routed_to_stockfish=0,
    )

    assert summary["stockfish_binary"] == "/tmp/stockfish"
    assert summary["chain_started_events"] == 1
    assert summary["chain_step_completed_events"] == 2
    assert summary["artifact_events"] == 2
    assert summary["chess_chain_success_rate"] == 1.0
    assert summary["out_of_domain_stockfish_rate"] == 0.0
    assert summary["decision_events"] == 1
