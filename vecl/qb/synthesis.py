from __future__ import annotations

from dataclasses import dataclass

from vecl.qb.claim_graph import ClaimGraph


@dataclass(frozen=True)
class SynthesisResult:
    conclusion_id: str
    answer_text: str
    supporting_claim_ids: list[str]
    rejected_claim_ids: list[str]
    assumptions: list[str]
    verification_status: str
    provenance_graph_hash: str
    synthesis_version: str = "v0"


def synthesize_claim_graph(
    claim_graph: ClaimGraph, verification_status: str = "PASSED"
) -> SynthesisResult:
    conflicts = claim_graph.conflicting_claims()
    unsupported = claim_graph.unsupported_claims()
    if conflicts:
        status = "NEEDS_REVIEW"
        answer = "Claims conflict; synthesis requires review."
    elif unsupported:
        status = "FAILED"
        answer = "Conclusion rejected because at least one claim is unsupported."
    else:
        status = verification_status
        texts = [
            claim_graph.graph.nodes[claim_id]["claim"].claim_text
            for claim_id in claim_graph.claims()
        ]
        answer = " ".join(texts) if texts else "No supported answer."
    supporting = [] if conflicts or unsupported else claim_graph.claims()
    assumptions = []
    for claim_id in claim_graph.claims():
        assumptions.extend(claim_graph.graph.nodes[claim_id]["claim"].assumptions)
    conclusion_id = f"conclusion:{claim_graph.stable_hash()[:12]}"
    if status == "PASSED" and supporting:
        claim_graph.add_conclusion(conclusion_id, answer, supporting)
    return SynthesisResult(
        conclusion_id=conclusion_id,
        answer_text=answer,
        supporting_claim_ids=supporting,
        rejected_claim_ids=sorted(set(conflicts + unsupported)),
        assumptions=assumptions,
        verification_status=status,
        provenance_graph_hash=claim_graph.stable_hash(),
    )
