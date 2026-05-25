from datetime import UTC, datetime, timedelta

from vecl.provenance.graph import ProvenanceGraph
from vecl.trust.anchors import TrustAnchor, TrustRootKind


def _anchor(anchor_id: str, source_id: str) -> TrustAnchor:
    now = datetime.now(UTC)
    return TrustAnchor(
        anchor_id=anchor_id,
        source_id=source_id,
        root_kind=TrustRootKind.TEST_FIXTURE,
        trust_value=0.8,
        issued_by="tester",
        issued_at=now,
        expires_at=now + timedelta(days=1),
        credential_hash=f"hash-{anchor_id}",
    )


def test_one_source_many_evidence_counts_as_one_source() -> None:
    graph = ProvenanceGraph()
    graph.add_evidence("e1", "s1", "t")
    graph.add_evidence("e2", "s1", "t")
    graph.add_claim("c1")
    graph.link_claim_to_evidence("c1", "e1")
    graph.link_claim_to_evidence("c1", "e2")
    assert graph.support_summary("c1")["unique_source_count"] == 1


def test_two_sources_sharing_same_root_are_not_independent_roots() -> None:
    graph = ProvenanceGraph()
    anchor = _anchor("root", "s1")
    graph.add_evidence("e1", "s1", "t")
    graph.add_evidence("e2", "s2", "t")
    graph.add_claim("c1")
    graph.link_claim_to_evidence("c1", "e1")
    graph.link_claim_to_evidence("c1", "e2")
    graph.link_source_to_anchor("s1", anchor)
    graph.link_source_to_anchor("s2", anchor)
    assert len(graph.independent_root_support("c1")) == 1


def test_circular_support_detection() -> None:
    graph = ProvenanceGraph()
    graph.add_claim("c1")
    graph.add_claim("c2")
    graph.link_claims("c1", "c2")
    graph.link_claims("c2", "c1")
    assert graph.has_circular_support("c1")


def test_root_supported_claim() -> None:
    graph = ProvenanceGraph()
    graph.add_evidence("e1", "s1", "t")
    graph.add_claim("c1")
    graph.link_claim_to_evidence("c1", "e1")
    graph.link_source_to_anchor("s1", _anchor("root", "s1"))
    assert graph.support_summary("c1")["independent_root_count"] == 1
