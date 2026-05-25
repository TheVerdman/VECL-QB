from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from vecl.trust.anchors import TrustAnchor


@dataclass
class ProvenanceGraph:
    graph: nx.DiGraph

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def add_evidence(
        self,
        evidence_id: str,
        source_id: str,
        tenant_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.graph.add_node(source_id, kind="source", source_id=source_id, tenant_id=tenant_id)
        self.graph.add_node(
            evidence_id,
            kind="evidence",
            evidence_id=evidence_id,
            source_id=source_id,
            tenant_id=tenant_id,
            payload=payload or {},
        )
        self.graph.add_edge(source_id, evidence_id, relation="provided")

    def add_claim(
        self,
        claim_id: str,
        source_ids: list[str] | None = None,
        tenant_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.graph.add_node(
            claim_id,
            kind="claim",
            claim_id=claim_id,
            source_ids=source_ids or [],
            tenant_id=tenant_id,
            payload=payload or {},
        )

    def add_specialist(self, specialist_id: str, tenant_id: str) -> None:
        self.graph.add_node(
            specialist_id, kind="specialist", specialist_id=specialist_id, tenant_id=tenant_id
        )

    def add_learning_event(self, event_id: str, tenant_id: str) -> None:
        self.graph.add_node(event_id, kind="learning_event", event_id=event_id, tenant_id=tenant_id)

    def link_claim_to_evidence(self, claim_id: str, evidence_id: str) -> None:
        self.graph.add_edge(evidence_id, claim_id, relation="supports")

    def link_claims(
        self, source_claim_id: str, target_claim_id: str, relation: str = "supports"
    ) -> None:
        self.graph.add_edge(source_claim_id, target_claim_id, relation=relation)

    def link_source_to_anchor(self, source_id: str, anchor: TrustAnchor) -> None:
        self.graph.add_node(
            anchor.anchor_id,
            kind="trust_anchor",
            anchor=anchor,
            trust_value=anchor.trust_value,
            revoked=anchor.revocation_status == "revoked",
        )
        self.graph.add_node(source_id, kind="source", source_id=source_id)
        self.graph.add_edge(anchor.anchor_id, source_id, relation="anchors")

    def independent_root_support(self, claim_id: str) -> set[TrustAnchor]:
        if claim_id not in self.graph:
            return set()
        roots: set[TrustAnchor] = set()
        for node in nx.ancestors(self.graph, claim_id):
            data = self.graph.nodes[node]
            if data.get("kind") == "trust_anchor":
                anchor = data["anchor"]
                if anchor.revocation_status != "revoked":
                    roots.add(anchor)
        return roots

    def has_circular_support(self, claim_id: str) -> bool:
        if claim_id not in self.graph:
            return False
        reachable = nx.ancestors(self.graph, claim_id) | {claim_id}
        return not nx.is_directed_acyclic_graph(self.graph.subgraph(reachable))

    def support_summary(self, claim_id: str) -> dict[str, Any]:
        evidence_ids = [
            node
            for node in self.graph.predecessors(claim_id)
            if self.graph.nodes[node].get("kind") == "evidence"
        ]
        source_ids = sorted(
            {
                self.graph.nodes[evidence_id].get("source_id")
                for evidence_id in evidence_ids
                if self.graph.nodes[evidence_id].get("source_id")
            }
        )
        roots = self.independent_root_support(claim_id)
        return {
            "claim_id": claim_id,
            "evidence_count": len(evidence_ids),
            "unique_source_count": len(source_ids),
            "source_ids": source_ids,
            "independent_root_count": len(roots),
            "root_anchor_ids": sorted(anchor.anchor_id for anchor in roots),
            "circular_support": self.has_circular_support(claim_id),
        }
