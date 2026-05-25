from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.claim_graph import ClaimGraph
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.specialist import MockSpecialist, SpecialistClaim
from vecl.qb.synthesis import synthesize_claim_graph
from vecl.qb.verifier import VerificationPolicy, VerificationStatus, Verifier


def _claim(claim_id: str, text: str, claim_type: str = "decision") -> SpecialistClaim:
    return SpecialistClaim(
        claim_id,
        "s",
        text,
        claim_type,
        0.8,
        evidence_ids=[f"e-{claim_id}"],
        source_ids=["source"],
    )


def test_simple_synthesis() -> None:
    graph = ClaimGraph()
    graph.add_claim(_claim("c1", "Carrier is qualified."))
    result = synthesize_claim_graph(graph)
    assert result.verification_status == "PASSED"
    assert result.supporting_claim_ids == ["c1"]


def test_conflicting_claims_mark_review() -> None:
    graph = ClaimGraph()
    graph.add_claim(_claim("c1", "Accept carrier."))
    graph.add_claim(_claim("c2", "Reject carrier."))
    graph.add_edge("c1", "c2", "contradicts")
    result = synthesize_claim_graph(graph)
    assert result.verification_status == "NEEDS_REVIEW"


def test_conclusion_with_missing_support_rejected() -> None:
    graph = ClaimGraph()
    graph.graph.add_node("c1", kind="claim", claim=_claim("c1", "unsupported"))
    result = synthesize_claim_graph(graph)
    assert result.verification_status == "FAILED"


def test_provenance_hash_stable_across_repeated_runs() -> None:
    a = ClaimGraph()
    b = ClaimGraph()
    a.add_claim(_claim("c1", "Stable."))
    b.add_claim(_claim("c1", "Stable."))
    assert a.stable_hash() == b.stable_hash()


def test_verifier_cases() -> None:
    verifier = Verifier()
    graph = ClaimGraph()
    graph.graph.add_node("c1", kind="claim", claim=_claim("c1", "unsupported"))
    failed = verifier.verify_claim_graph(graph, VerificationPolicy())
    assert failed.status == VerificationStatus.FAILED
    assert failed.unsupported_claim_ids == ["c1"]

    rooted = ClaimGraph()
    rooted.add_claim(_claim("c2", "rooted"))
    rooted.graph.nodes["c2"]["independent_roots"] = ["r1"]
    assert (
        verifier.verify_claim_graph(rooted, VerificationPolicy(required_independent_roots=2)).status
        == VerificationStatus.FAILED
    )

    cyclic = ClaimGraph()
    cyclic.add_claim(_claim("c3", "a"))
    cyclic.add_claim(_claim("c4", "b"))
    cyclic.add_edge("c3", "c4", "supports")
    cyclic.add_edge("c4", "c3", "supports")
    assert verifier.verify_claim_graph(cyclic, VerificationPolicy()).circular_support_detected

    conflict = ClaimGraph()
    conflict.add_claim(_claim("c5", "a"))
    conflict.add_claim(_claim("c6", "b"))
    conflict.add_edge("c5", "c6", "contradicts")
    assert (
        verifier.verify_claim_graph(conflict, VerificationPolicy()).status
        == VerificationStatus.NEEDS_REVIEW
    )

    passed = ClaimGraph()
    passed.add_claim(_claim("c7", "ok"))
    assert (
        verifier.verify_claim_graph(passed, VerificationPolicy()).status
        == VerificationStatus.PASSED
    )

    high_risk = ClaimGraph()
    claim = SpecialistClaim("c8", "s", "risky", "decision", 0.9, evidence_ids=["e"], high_risk=True)
    high_risk.add_claim(claim)
    assert (
        verifier.verify_claim_graph(
            high_risk, VerificationPolicy(high_risk_requires_human_review=True)
        ).status
        == VerificationStatus.NEEDS_REVIEW
    )


def test_orchestrator_pass_conflict_and_unsupported() -> None:
    router = QBRouter()
    rate = MockSpecialist("rate", {"freight"}, [_claim("rate-c", "Accept carrier.", "rate")])
    compliance = MockSpecialist(
        "compliance", {"freight"}, [_claim("comp-c", "Carrier is qualified.", "compliance")]
    )
    for specialist in [rate, compliance]:
        router.register_specialist(
            SpecialistCard(
                specialist.specialist_id,
                {"freight"},
                {},
                1.0,
                1.0,
                "v1",
                effective_trust=0.8,
            ),
            specialist,
        )
    orchestrator = QBOrchestrator(router, ProvenanceLedger())
    passed = orchestrator.run_task(tenant_id="t", task_type="freight", input_payload={})
    assert passed.verification_status == "PASSED"
    assert set(passed.specialist_ids) == {"rate", "compliance"}
    assert passed.provenance_event_ids

    conflict = orchestrator.run_task(
        tenant_id="t",
        task_type="freight",
        input_payload={"contradictions": [("rate-c", "comp-c")]},
    )
    assert conflict.verification_status == "NEEDS_REVIEW"

    unsupported = QBOrchestrator(QBRouter(), ProvenanceLedger()).run_task(
        tenant_id="t", task_type="missing", input_payload={}
    )
    assert unsupported.verification_status == "FAILED"
