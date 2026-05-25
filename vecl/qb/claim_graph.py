from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from vecl.provenance.events import stable_hash
from vecl.qb.specialist import SpecialistClaim


@dataclass
class ClaimGraph:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def add_claim(self, claim: SpecialistClaim) -> None:
        self.graph.add_node(
            claim.claim_id,
            kind="claim",
            claim=claim,
            claim_type=claim.claim_type,
            confidence=claim.confidence,
            high_risk=claim.high_risk,
        )
        for evidence_id in claim.evidence_ids:
            self.add_evidence(evidence_id, source_ids=claim.source_ids)
            self.graph.add_edge(evidence_id, claim.claim_id, relation="supports")
        for assumption in claim.assumptions:
            assumption_id = f"assumption:{claim.claim_id}:{stable_hash(assumption)[:8]}"
            self.graph.add_node(assumption_id, kind="assumption", text=assumption)
            self.graph.add_edge(assumption_id, claim.claim_id, relation="supports")

    def add_evidence(self, evidence_id: str, source_ids: list[str] | None = None) -> None:
        self.graph.add_node(
            evidence_id, kind="evidence", evidence_id=evidence_id, source_ids=source_ids or []
        )

    def add_conclusion(
        self, conclusion_id: str, text: str, supporting_claim_ids: list[str]
    ) -> None:
        self.graph.add_node(conclusion_id, kind="conclusion", text=text)
        for claim_id in supporting_claim_ids:
            self.graph.add_edge(claim_id, conclusion_id, relation="supports")

    def add_edge(self, source: str, target: str, relation: str) -> None:
        self.graph.add_edge(source, target, relation=relation)

    def claims(self) -> list[str]:
        return sorted(
            node for node, data in self.graph.nodes(data=True) if data.get("kind") == "claim"
        )

    def conclusions(self) -> list[str]:
        return sorted(
            node for node, data in self.graph.nodes(data=True) if data.get("kind") == "conclusion"
        )

    def unsupported_claims(self) -> list[str]:
        unsupported = []
        for claim_id in self.claims():
            predecessors = list(self.graph.predecessors(claim_id))
            if not any(
                self.graph.edges[pred, claim_id].get("relation") == "supports"
                for pred in predecessors
            ):
                unsupported.append(claim_id)
        return unsupported

    def conflicting_claims(self) -> list[str]:
        conflicts: set[str] = set()
        for source, target, data in self.graph.edges(data=True):
            if data.get("relation") == "contradicts":
                conflicts.update([source, target])
        return sorted(conflicts)

    def stable_hash(self) -> str:
        nodes: list[dict[str, Any]] = []
        for node, data in sorted(self.graph.nodes(data=True), key=lambda item: str(item[0])):
            clean = {
                key: value
                for key, value in data.items()
                if key != "claim"
                and isinstance(value, str | int | float | bool | list | dict | type(None))
            }
            if "claim" in data:
                claim = data["claim"]
                clean["claim"] = {
                    "claim_id": claim.claim_id,
                    "specialist_id": claim.specialist_id,
                    "claim_text": claim.claim_text,
                    "claim_type": claim.claim_type,
                    "confidence": claim.confidence,
                    "evidence_ids": sorted(claim.evidence_ids),
                    "source_ids": sorted(claim.source_ids),
                    "assumptions": sorted(claim.assumptions),
                    "limitations": sorted(claim.limitations),
                    "high_risk": claim.high_risk,
                }
            nodes.append({"id": node, "data": clean})
        edges = sorted(
            [
                {"source": source, "target": target, "relation": data.get("relation")}
                for source, target, data in self.graph.edges(data=True)
            ],
            key=lambda item: (str(item["source"]), str(item["target"]), str(item["relation"])),
        )
        return stable_hash({"nodes": nodes, "edges": edges})

    def summary(self) -> str:
        return json.dumps(
            {
                "claims": self.claims(),
                "conclusions": self.conclusions(),
                "unsupported_claims": self.unsupported_claims(),
                "conflicting_claims": self.conflicting_claims(),
                "hash": self.stable_hash(),
            },
            indent=2,
            sort_keys=True,
        )
