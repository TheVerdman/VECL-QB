from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from vecl.episodic.store import EpisodicStore
from vecl.ethics.kernel import EthicsKernel
from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.chain_executor import ChainExecutionResult, ChainExecutor
from vecl.qb.claim_graph import ClaimGraph
from vecl.qb.planner import ChainPlan
from vecl.qb.router import QBRouter
from vecl.qb.specialist import Specialist, SpecialistRequest, SpecialistResponse
from vecl.qb.synthesis import SynthesisResult, synthesize_claim_graph
from vecl.qb.verifier import VerificationPolicy, VerificationResult, VerificationStatus, Verifier


@dataclass(frozen=True)
class QBOrchestrationResult:
    answer_text: str
    claim_graph_hash: str
    verification_status: str
    specialist_ids: list[str]
    provenance_event_ids: list[str]
    synthesis: SynthesisResult | None
    verification: VerificationResult


class QBOrchestrator:
    def __init__(
        self,
        router: QBRouter,
        ledger: ProvenanceLedger,
        verifier: Verifier | None = None,
        policy: VerificationPolicy | None = None,
        episodic_store: EpisodicStore | None = None,
        ethics_kernel: EthicsKernel | None = None,
    ) -> None:
        self.router = router
        self.ledger = ledger
        self.verifier = verifier or Verifier()
        self.policy = policy or VerificationPolicy()
        self.episodic_store = episodic_store
        self.ethics_kernel = ethics_kernel

    def run_task(
        self,
        *,
        tenant_id: str,
        task_type: str,
        input_payload: dict[str, Any],
        required_output_schema: dict[str, Any] | None = None,
        provenance_context: dict[str, Any] | None = None,
        chain_plan: ChainPlan | None = None,
    ) -> QBOrchestrationResult:
        request_id = f"req-{uuid4()}"
        request_event = self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.EVIDENCE_INGESTED,
                tenant_id=tenant_id,
                actor="qb-orchestrator",
                payload={"request_id": request_id, "task_type": task_type},
            )
        )
        event_ids: list[str] = [request_event.event_id]
        runtime_context = {
            **(provenance_context or {}),
            "parent_event_id": request_event.event_id,
        }
        request = SpecialistRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            task_type=task_type,
            input_payload=input_payload,
            required_output_schema=required_output_schema or {},
            provenance_context=runtime_context,
        )
        if chain_plan is not None:
            return self._run_chain_task(
                request=request,
                input_payload=input_payload,
                event_ids=event_ids,
                chain_plan=chain_plan,
            )

        specialists = self.router.route(request)
        if not specialists:
            graph = ClaimGraph()
            verification = VerificationResult(
                status=VerificationStatus.FAILED,
                reasons=["no specialist could handle task"],
                unsupported_claim_ids=[],
                conflicting_claim_ids=[],
                missing_root_support=[],
                circular_support_detected=False,
            )
            return QBOrchestrationResult(
                answer_text="No registered specialist can handle this task.",
                claim_graph_hash=graph.stable_hash(),
                verification_status=verification.status.value,
                specialist_ids=[],
                provenance_event_ids=event_ids,
                synthesis=None,
                verification=verification,
            )

        responses = [specialist.run(request) for specialist in specialists]
        event_ids.extend(self._record_artifacts(responses, tenant_id, request_event.event_id))
        graph = self._build_claim_graph(responses, input_payload)
        verification = self.verifier.verify_claim_graph(graph, self.policy)
        synthesis = synthesize_claim_graph(graph, verification.status.value)
        response_event = self._record_final_response(
            request=request,
            responses=responses,
            graph=graph,
            verification=verification,
            parent_event_id=request_event.event_id,
        )
        event_ids.append(response_event.event_id)
        return QBOrchestrationResult(
            answer_text=synthesis.answer_text,
            claim_graph_hash=synthesis.provenance_graph_hash,
            verification_status=synthesis.verification_status,
            specialist_ids=[response.specialist_id for response in responses],
            provenance_event_ids=event_ids,
            synthesis=synthesis,
            verification=verification,
        )

    def _run_chain_task(
        self,
        *,
        request: SpecialistRequest,
        input_payload: dict[str, Any],
        event_ids: list[str],
        chain_plan: ChainPlan,
    ) -> QBOrchestrationResult:
        execution = ChainExecutor(
            ledger=self.ledger,
            specialists=self._registered_specialists(),
            episodic_store=self.episodic_store,
            ethics_kernel=self.ethics_kernel,
        ).execute(chain_plan, request)
        event_ids.extend(execution.event_ids)
        graph = self._build_claim_graph(execution.responses, input_payload)
        if execution.aborted:
            verification = VerificationResult(
                status=VerificationStatus.FAILED,
                reasons=[execution.failure_reason or "chain aborted"],
                unsupported_claim_ids=[],
                conflicting_claim_ids=[],
                missing_root_support=[],
                circular_support_detected=False,
            )
            return QBOrchestrationResult(
                answer_text=(
                    f"Chain aborted at {execution.failed_step_id}: {execution.failure_reason}"
                ),
                claim_graph_hash=graph.stable_hash(),
                verification_status=verification.status.value,
                specialist_ids=[response.specialist_id for response in execution.responses],
                provenance_event_ids=event_ids,
                synthesis=None,
                verification=verification,
            )
        verification = self.verifier.verify_claim_graph(graph, self.policy)
        synthesis = synthesize_claim_graph(graph, verification.status.value)
        response_event = self._record_final_response(
            request=request,
            responses=execution.responses,
            graph=graph,
            verification=verification,
            parent_event_id=execution.chain_event_id,
            chain_execution=execution,
        )
        event_ids.append(response_event.event_id)
        return QBOrchestrationResult(
            answer_text=synthesis.answer_text,
            claim_graph_hash=synthesis.provenance_graph_hash,
            verification_status=synthesis.verification_status,
            specialist_ids=[response.specialist_id for response in execution.responses],
            provenance_event_ids=event_ids,
            synthesis=synthesis,
            verification=verification,
        )

    def _record_final_response(
        self,
        *,
        request: SpecialistRequest,
        responses: list[SpecialistResponse],
        graph: ClaimGraph,
        verification: VerificationResult,
        parent_event_id: str,
        chain_execution: ChainExecutionResult | None = None,
    ) -> ProvenanceEvent:
        payload: dict[str, Any] = {
            "request_id": request.request_id,
            "specialist_ids": [response.specialist_id for response in responses],
            "claim_graph_hash": graph.stable_hash(),
            "verification_status": verification.status.value,
        }
        if chain_execution is not None:
            payload["chain_event_id"] = chain_execution.chain_event_id
            payload["artifact_event_ids"] = list(chain_execution.artifact_event_ids)
            payload["episodic_event_ids"] = list(chain_execution.episodic_event_ids)
        return self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.FINAL_RESPONSE_RECORDED,
                tenant_id=request.tenant_id,
                actor="qb-orchestrator",
                parent_event_ids=[parent_event_id],
                payload=payload,
            )
        )

    def _registered_specialists(self) -> dict[str, Specialist]:
        specialists = getattr(self.router, "specialists", None)
        if specialists is None:
            return {}
        return dict(specialists)

    def _record_artifacts(
        self, responses: list[SpecialistResponse], tenant_id: str, parent_event_id: str
    ) -> list[str]:
        event_ids: list[str] = []
        for response in responses:
            for record in (response.cost_metadata or {}).get("artifact_records", []):
                artifact_event = self.ledger.append(
                    ProvenanceEvent(
                        event_type=EventType.ARTIFACT_PRODUCED,
                        tenant_id=tenant_id,
                        actor=response.specialist_id,
                        payload=dict(record),
                        parent_event_ids=[parent_event_id],
                    )
                )
                event_ids.append(artifact_event.event_id)
        return event_ids

    def _build_claim_graph(
        self, responses: list[SpecialistResponse], input_payload: dict[str, Any]
    ) -> ClaimGraph:
        graph = ClaimGraph()
        for response in responses:
            for claim in response.claims:
                graph.add_claim(claim)
                if roots := input_payload.get("independent_roots", {}).get(claim.claim_id):
                    graph.graph.nodes[claim.claim_id]["independent_roots"] = roots
        for source, target in input_payload.get("contradictions", []):
            if source in graph.graph and target in graph.graph:
                graph.add_edge(source, target, "contradicts")
        self._add_obvious_text_conflicts(graph)
        return graph

    def _add_obvious_text_conflicts(self, graph: ClaimGraph) -> None:
        claims = graph.claims()
        for index, left in enumerate(claims):
            left_claim = graph.graph.nodes[left]["claim"]
            for right in claims[index + 1 :]:
                right_claim = graph.graph.nodes[right]["claim"]
                if left_claim.claim_type != right_claim.claim_type:
                    continue
                left_text = left_claim.claim_text.lower()
                right_text = right_claim.claim_text.lower()
                left_negative = any(
                    word in left_text for word in ["reject", "unsafe", "unqualified"]
                )
                right_negative = any(
                    word in right_text for word in ["reject", "unsafe", "unqualified"]
                )
                left_positive = any(word in left_text for word in ["accept", "safe", "qualified"])
                right_positive = any(word in right_text for word in ["accept", "safe", "qualified"])
                if (left_negative and right_positive) or (right_negative and left_positive):
                    graph.add_edge(left, right, "contradicts")
