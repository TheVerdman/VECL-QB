from __future__ import annotations

from pathlib import Path

from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.claim_graph import ClaimGraph
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.specialist import SpecialistRequest
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.stockfish import (
    STOCKFISH_SOURCE_ID,
    StockfishSpecialist,
    create_stockfish_trust_anchor,
    resolve_stockfish_binary,
)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _request(tmp_path: Path) -> tuple[StockfishSpecialist, SpecialistRequest]:
    specialist = StockfishSpecialist(artifact_store=ContentAddressedStore(tmp_path))
    request = SpecialistRequest(
        request_id="req-stockfish",
        tenant_id="tenant-chess",
        task_type="chess_eval",
        input_payload={"fen": STARTING_FEN, "depth": 4},
        required_output_schema={},
        provenance_context={"parent_event_id": "evt-parent"},
    )
    return specialist, request


def test_real_stockfish_18_runs_and_writes_artifact(tmp_path: Path) -> None:
    specialist, request = _request(tmp_path)

    response = specialist.run(request)

    assert response.refusal_or_error is None
    assert response.cost_metadata is not None
    assert response.cost_metadata["engine_name"] == "Stockfish 18"
    claim = response.claims[0]
    assert claim.claim_type == "chess_eval"
    assert claim.source_ids == [STOCKFISH_SOURCE_ID]
    assert "bestmove=" in claim.claim_text
    assert "eval_cp=" in claim.claim_text or "mate_in=" in claim.claim_text
    assert "time_ms=" in claim.claim_text
    assert "depth_reached=4" in claim.claim_text

    record = response.cost_metadata["artifact_records"][0]
    assert claim.artifact_ids == [record["artifact_id"]]
    transcript = Path(str(record["output_path"])).read_text()
    assert "id name Stockfish 18" in transcript
    assert f"position fen {STARTING_FEN}" in transcript
    assert "bestmove" in transcript


def test_real_stockfish_claim_graph_hash_is_stable_across_runs(tmp_path: Path) -> None:
    specialist, request = _request(tmp_path)

    first = specialist.run(request)
    second = specialist.run(request)

    assert first.refusal_or_error is None
    assert second.refusal_or_error is None
    first_graph = ClaimGraph()
    second_graph = ClaimGraph()
    first_graph.add_claim(first.claims[0])
    second_graph.add_claim(second.claims[0])
    assert first_graph.stable_hash() == second_graph.stable_hash()


def test_stockfish_orchestrator_records_artifact_event(tmp_path: Path) -> None:
    specialist = StockfishSpecialist(artifact_store=ContentAddressedStore(tmp_path))
    anchor = create_stockfish_trust_anchor(specialist.binary, "Stockfish 18")
    router = QBRouter(max_specialists=1)
    router.register_specialist(
        SpecialistCard(
            specialist.specialist_id,
            {"chess_eval"},
            {"trust_anchor_id": anchor.anchor_id},
            cost_hint=1.0,
            latency_hint=1.0,
            version=specialist.version,
            effective_trust=anchor.trust_value,
        ),
        specialist,
    )
    ledger = ProvenanceLedger()
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"chess_eval"}),
    )

    result = orchestrator.run_task(
        tenant_id="tenant-chess",
        task_type="chess_eval",
        input_payload={"fen": STARTING_FEN, "depth": 4},
    )

    artifact_events = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    assert result.verification_status == "PASSED"
    assert len(artifact_events) == 1
    assert artifact_events[0].payload["producer_specialist_id"] == "stockfish"
    assert artifact_events[0].parent_event_ids == [result.provenance_event_ids[0]]
    assert artifact_events[0].event_id in result.provenance_event_ids


def test_stockfish_trust_anchor_binds_binary_and_version() -> None:
    binary = resolve_stockfish_binary()
    anchor = create_stockfish_trust_anchor(binary, "Stockfish 18")

    assert anchor.source_id == STOCKFISH_SOURCE_ID
    assert anchor.root_kind.value == "VERIFIED_OPERATIONAL_RECORD"
    assert anchor.trust_value == 0.9
    assert anchor.credential_hash
