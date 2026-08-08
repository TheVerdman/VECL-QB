# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl._paths import environment_directory
from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.stockfish import StockfishSpecialist, create_stockfish_trust_anchor

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def build_orchestrator(artifact_dir: Path | None = None) -> QBOrchestrator:
    artifact_root = artifact_dir or environment_directory(
        "VECL_ARTIFACT_STORE", prefix="vecl-stockfish-demo-artifacts-"
    )
    specialist = StockfishSpecialist(artifact_store=ContentAddressedStore(artifact_root))
    trust_anchor = create_stockfish_trust_anchor(specialist.binary, "Stockfish 18")
    router = QBRouter(max_specialists=1)
    router.register_specialist(
        SpecialistCard(
            specialist.specialist_id,
            {"chess_eval"},
            {"trust_anchor_id": trust_anchor.anchor_id},
            cost_hint=1.0,
            latency_hint=1.0,
            version=specialist.version,
            effective_trust=trust_anchor.trust_value,
        ),
        specialist,
    )
    return QBOrchestrator(
        router,
        ProvenanceLedger(),
        policy=VerificationPolicy(required_claim_types={"chess_eval"}),
    )


def main() -> None:
    orchestrator = build_orchestrator()
    result = orchestrator.run_task(
        tenant_id="chess-demo",
        task_type="chess_eval",
        input_payload={"fen": STARTING_FEN, "depth": 4},
    )
    print("Final answer:")
    print(result.answer_text)
    print("\nVerification status:")
    print(result.verification_status)
    print("\nClaim graph hash:")
    print(result.claim_graph_hash)
    print("\nProvenance event ids:")
    print(", ".join(result.provenance_event_ids))
    artifact_events = orchestrator.ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    print("\nArtifact ids:")
    print(", ".join(str(event.payload["artifact_id"]) for event in artifact_events))


if __name__ == "__main__":
    main()
