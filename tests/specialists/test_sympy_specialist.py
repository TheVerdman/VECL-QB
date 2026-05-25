from __future__ import annotations

from pathlib import Path

from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.specialist import SpecialistRequest
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.sympy_specialist import (
    SYMPY_SOURCE_ID,
    SymPySpecialist,
    create_sympy_trust_anchor,
)


def _request(payload: dict[str, object]) -> SpecialistRequest:
    return SpecialistRequest(
        "req-sympy",
        "tenant-math",
        "symbolic_math",
        payload,
        {},
        {"parent_event_id": "evt-parent"},
    )


def test_sympy_simplifies_expression_and_writes_tex_artifact(tmp_path: Path) -> None:
    specialist = SymPySpecialist(artifact_store=ContentAddressedStore(tmp_path))

    response = specialist.run(
        _request({"operation": "simplify", "expression": "sin(x)**2 + cos(x)**2"})
    )

    assert response.refusal_or_error is None
    claim = response.claims[0]
    assert claim.claim_type == "symbolic_math"
    assert claim.source_ids == [SYMPY_SOURCE_ID]
    assert "result=1" in claim.claim_text
    records = response.cost_metadata["artifact_records"]  # type: ignore[index]
    assert [record["output_format"] for record in records] == ["tex"]
    tex = Path(str(records[0]["output_path"])).read_text()
    assert "\\documentclass{article}" in tex
    assert "\\sin" in tex


def test_sympy_solves_polynomial(tmp_path: Path) -> None:
    specialist = SymPySpecialist(artifact_store=ContentAddressedStore(tmp_path))

    response = specialist.run(
        _request({"operation": "solve", "expression": "x**2 - 4", "variable": "x"})
    )

    assert response.refusal_or_error is None
    assert "result=[-2, 2]" in response.claims[0].claim_text


def test_sympy_orchestrator_records_artifact_event(tmp_path: Path) -> None:
    specialist = SymPySpecialist(artifact_store=ContentAddressedStore(tmp_path))
    anchor = create_sympy_trust_anchor()
    router = QBRouter(max_specialists=1)
    router.register_specialist(
        SpecialistCard(
            specialist.specialist_id,
            {"symbolic_math"},
            {"trust_anchor_id": anchor.anchor_id},
            cost_hint=0.2,
            latency_hint=0.2,
            version=specialist.version,
            effective_trust=anchor.trust_value,
        ),
        specialist,
    )
    ledger = ProvenanceLedger()
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"symbolic_math"}),
    )

    result = orchestrator.run_task(
        tenant_id="tenant-math",
        task_type="symbolic_math",
        input_payload={"operation": "factor", "expression": "x**2 - 1"},
    )

    artifacts = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    assert result.verification_status == "PASSED"
    assert result.specialist_ids == ["sympy"]
    assert len(artifacts) == 1
    assert artifacts[0].payload["producer_specialist_id"] == "sympy"


def test_sympy_trust_anchor() -> None:
    anchor = create_sympy_trust_anchor()

    assert anchor.source_id == SYMPY_SOURCE_ID
    assert anchor.root_kind.value == "VERIFIED_OPERATIONAL_RECORD"
    assert anchor.trust_value == 0.95
    assert anchor.credential_hash
