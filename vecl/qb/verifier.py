from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from vecl._compat import StrEnum
from vecl.qb.claim_graph import ClaimGraph


class VerificationStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass(frozen=True)
class VerificationPolicy:
    required_independent_roots: int = 0
    allow_assumptions: bool = True
    max_conflict_score: float = 0.0
    required_claim_types: set[str] = field(default_factory=set)
    high_risk_requires_human_review: bool = False


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    reasons: list[str]
    unsupported_claim_ids: list[str]
    conflicting_claim_ids: list[str]
    missing_root_support: list[str]
    circular_support_detected: bool


class Verifier:
    def verify_claim_graph(
        self, claim_graph: ClaimGraph, policy: VerificationPolicy
    ) -> VerificationResult:
        unsupported = claim_graph.unsupported_claims()
        conflicts = claim_graph.conflicting_claims()
        missing_types: list[str] = []
        present_types = {
            data.get("claim_type")
            for _, data in claim_graph.graph.nodes(data=True)
            if data.get("kind") == "claim"
        }
        for claim_type in policy.required_claim_types:
            if claim_type not in present_types:
                missing_types.append(claim_type)

        missing_roots: list[str] = []
        if policy.required_independent_roots > 0:
            for claim_id in claim_graph.claims():
                roots = claim_graph.graph.nodes[claim_id].get("independent_roots", [])
                if len(set(roots)) < policy.required_independent_roots:
                    missing_roots.append(claim_id)

        has_cycle = not nx.is_directed_acyclic_graph(claim_graph.graph)
        high_risk_claims = [
            claim_id
            for claim_id in claim_graph.claims()
            if claim_graph.graph.nodes[claim_id].get("high_risk")
        ]

        reasons: list[str] = []
        if unsupported:
            reasons.append("unsupported claims present")
        if conflicts:
            reasons.append("conflicting claims present")
        if missing_roots:
            reasons.append("insufficient independent root support")
        if has_cycle:
            reasons.append("circular support detected")
        if missing_types:
            reasons.append(f"missing required claim types: {sorted(missing_types)}")
        if not policy.allow_assumptions:
            assumed = [
                claim_id
                for claim_id in claim_graph.claims()
                if claim_graph.graph.nodes[claim_id]["claim"].assumptions
            ]
            if assumed:
                unsupported.extend(assumed)
                reasons.append("assumptions are not allowed")

        if policy.high_risk_requires_human_review and high_risk_claims and not reasons:
            return VerificationResult(
                status=VerificationStatus.NEEDS_REVIEW,
                reasons=["high-risk claim requires human review"],
                unsupported_claim_ids=[],
                conflicting_claim_ids=[],
                missing_root_support=[],
                circular_support_detected=False,
            )
        status = VerificationStatus.PASSED if not reasons else VerificationStatus.FAILED
        if conflicts and len(conflicts) > policy.max_conflict_score:
            status = VerificationStatus.NEEDS_REVIEW
        return VerificationResult(
            status=status,
            reasons=reasons,
            unsupported_claim_ids=sorted(set(unsupported)),
            conflicting_claim_ids=conflicts,
            missing_root_support=sorted(set(missing_roots)),
            circular_support_detected=has_cycle,
        )
