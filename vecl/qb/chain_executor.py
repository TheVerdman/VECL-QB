from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vecl.episodic.store import EpisodicStore
from vecl.ethics.kernel import EthicsContext, EthicsDecisionType, EthicsEvaluation, EthicsKernel
from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.planner import ChainPlan, ChainStep, topological_steps
from vecl.qb.specialist import Specialist, SpecialistRequest, SpecialistResponse
from vecl.runtime.tokens import LearningEventToken


@dataclass(frozen=True)
class ChainExecutionResult:
    chain_event_id: str
    responses: list[SpecialistResponse]
    event_ids: list[str]
    artifact_event_ids: list[str]
    episodic_event_ids: list[str]
    aborted: bool = False
    failure_reason: str | None = None
    failed_step_id: str | None = None


class ChainExecutor:
    def __init__(
        self,
        *,
        ledger: ProvenanceLedger,
        specialists: dict[str, Specialist],
        episodic_store: EpisodicStore | None = None,
        ethics_kernel: EthicsKernel | None = None,
    ) -> None:
        self.ledger = ledger
        self.specialists = dict(specialists)
        self.episodic_store = episodic_store
        self.ethics_kernel = ethics_kernel

    def execute(self, plan: ChainPlan, request: SpecialistRequest) -> ChainExecutionResult:
        if plan.task_type != request.task_type:
            raise ValueError("chain plan task_type must match request task_type")
        chain_event = self.ledger.append(
            ProvenanceEvent(
                EventType.CHAIN_STARTED,
                request.tenant_id,
                "chain-executor",
                {
                    "request_id": request.request_id,
                    "plan_id": plan.plan_id,
                    "task_type": plan.task_type,
                    "step_ids": [step.step_id for step in plan.steps],
                },
                parent_event_ids=_request_parent_ids(request),
            )
        )
        event_ids = [chain_event.event_id]
        artifact_event_ids: list[str] = []
        episodic_event_ids: list[str] = []
        responses_by_step: dict[str, SpecialistResponse] = {}
        responses: list[SpecialistResponse] = []

        for step in topological_steps(plan):
            started = self.ledger.append(
                ProvenanceEvent(
                    EventType.CHAIN_STEP_STARTED,
                    request.tenant_id,
                    "chain-executor",
                    {
                        "request_id": request.request_id,
                        "plan_id": plan.plan_id,
                        "step_id": step.step_id,
                        "specialist_id": step.specialist_id,
                        "inputs_from": list(step.inputs_from),
                        "expected_artifact_type": step.expected_artifact_type,
                    },
                    parent_event_ids=[chain_event.event_id],
                )
            )
            event_ids.append(started.event_id)
            specialist = self.specialists.get(step.specialist_id)
            step_request = _bind_step_request(
                request=request,
                plan=plan,
                step=step,
                chain_event_id=chain_event.event_id,
                step_started_event_id=started.event_id,
                responses_by_step=responses_by_step,
            )
            if self.ethics_kernel is not None:
                ethics_context = _ethics_context_for_step(
                    request=step_request,
                    step=step,
                    chain_event_id=chain_event.event_id,
                    step_started_event_id=started.event_id,
                    specialist_registered=specialist is not None,
                )
                ethics_evaluation = self.ethics_kernel.evaluate(ethics_context)
                if ethics_evaluation.final_decision.decision != EthicsDecisionType.ALLOW:
                    return self._halt_for_ethics(
                        request=request,
                        plan=plan,
                        step=step,
                        step_started_event_id=started.event_id,
                        chain_event_id=chain_event.event_id,
                        event_ids=event_ids,
                        responses=responses,
                        artifact_event_ids=artifact_event_ids,
                        episodic_event_ids=episodic_event_ids,
                        ethics_context=ethics_context,
                        ethics_evaluation=ethics_evaluation,
                    )

            if specialist is None:
                return self._abort(
                    request=request,
                    plan=plan,
                    step=step,
                    step_started_event_id=started.event_id,
                    chain_event_id=chain_event.event_id,
                    event_ids=event_ids,
                    responses=responses,
                    artifact_event_ids=artifact_event_ids,
                    episodic_event_ids=episodic_event_ids,
                    failure_reason=f"unknown specialist_id: {step.specialist_id}",
                )

            try:
                response = specialist.run(step_request)
            except Exception as exc:
                return self._abort(
                    request=request,
                    plan=plan,
                    step=step,
                    step_started_event_id=started.event_id,
                    chain_event_id=chain_event.event_id,
                    event_ids=event_ids,
                    responses=responses,
                    artifact_event_ids=artifact_event_ids,
                    episodic_event_ids=episodic_event_ids,
                    failure_reason=str(exc),
                )
            if response.refusal_or_error:
                return self._abort(
                    request=request,
                    plan=plan,
                    step=step,
                    step_started_event_id=started.event_id,
                    chain_event_id=chain_event.event_id,
                    event_ids=event_ids,
                    responses=responses,
                    artifact_event_ids=artifact_event_ids,
                    episodic_event_ids=episodic_event_ids,
                    failure_reason=response.refusal_or_error,
                )

            artifact_ids = _response_artifact_ids(response)
            completed = self.ledger.append(
                ProvenanceEvent(
                    EventType.CHAIN_STEP_COMPLETED,
                    request.tenant_id,
                    "chain-executor",
                    {
                        "request_id": request.request_id,
                        "plan_id": plan.plan_id,
                        "step_id": step.step_id,
                        "specialist_id": step.specialist_id,
                        "claim_ids": [claim.claim_id for claim in response.claims],
                        "artifact_ids": artifact_ids,
                    },
                    parent_event_ids=[started.event_id],
                )
            )
            event_ids.append(completed.event_id)
            artifact_parent_id = completed.event_id
            episodic_entry_id: str | None = None
            if self.episodic_store is not None:
                episodic_entry_id, episodic_event_id = _write_episodic_entry(
                    self.episodic_store,
                    self.ledger,
                    request=request,
                    plan=plan,
                    step=step,
                    response=response,
                    parent_event_id=completed.event_id,
                    artifact_ids=artifact_ids,
                )
                event_ids.append(episodic_event_id)
                episodic_event_ids.append(episodic_event_id)
                artifact_parent_id = episodic_event_id
            recorded = _record_artifacts(
                self.ledger,
                response=response,
                tenant_id=request.tenant_id,
                parent_event_id=artifact_parent_id,
                episodic_entry_id=episodic_entry_id,
            )
            event_ids.extend(recorded)
            artifact_event_ids.extend(recorded)
            responses_by_step[step.step_id] = response
            responses.append(response)

        return ChainExecutionResult(
            chain_event_id=chain_event.event_id,
            responses=responses,
            event_ids=event_ids,
            artifact_event_ids=artifact_event_ids,
            episodic_event_ids=episodic_event_ids,
        )

    def _abort(
        self,
        *,
        request: SpecialistRequest,
        plan: ChainPlan,
        step: ChainStep,
        step_started_event_id: str,
        chain_event_id: str,
        event_ids: list[str],
        responses: list[SpecialistResponse],
        artifact_event_ids: list[str],
        episodic_event_ids: list[str],
        failure_reason: str,
    ) -> ChainExecutionResult:
        failed = self.ledger.append(
            ProvenanceEvent(
                EventType.CHAIN_STEP_FAILED,
                request.tenant_id,
                "chain-executor",
                {
                    "request_id": request.request_id,
                    "plan_id": plan.plan_id,
                    "step_id": step.step_id,
                    "specialist_id": step.specialist_id,
                    "failure_reason": failure_reason,
                },
                parent_event_ids=[step_started_event_id],
            )
        )
        aborted = self.ledger.append(
            ProvenanceEvent(
                EventType.CHAIN_ABORTED,
                request.tenant_id,
                "chain-executor",
                {
                    "request_id": request.request_id,
                    "plan_id": plan.plan_id,
                    "failed_step_id": step.step_id,
                    "failure_reason": failure_reason,
                },
                parent_event_ids=[failed.event_id],
            )
        )
        event_ids.extend([failed.event_id, aborted.event_id])
        return ChainExecutionResult(
            chain_event_id=chain_event_id,
            responses=responses,
            event_ids=event_ids,
            artifact_event_ids=artifact_event_ids,
            episodic_event_ids=episodic_event_ids,
            aborted=True,
            failure_reason=failure_reason,
            failed_step_id=step.step_id,
        )

    def _halt_for_ethics(
        self,
        *,
        request: SpecialistRequest,
        plan: ChainPlan,
        step: ChainStep,
        step_started_event_id: str,
        chain_event_id: str,
        event_ids: list[str],
        responses: list[SpecialistResponse],
        artifact_event_ids: list[str],
        episodic_event_ids: list[str],
        ethics_context: EthicsContext,
        ethics_evaluation: EthicsEvaluation,
    ) -> ChainExecutionResult:
        decision = ethics_evaluation.final_decision
        event_type = (
            EventType.CHAIN_STEP_REFUSED_BY_ETHICS
            if decision.decision == EthicsDecisionType.REFUSE
            else EventType.CHAIN_STEP_AWAITING_REVIEW
        )
        ethics_event = self.ledger.append(
            ProvenanceEvent(
                event_type,
                request.tenant_id,
                "ethics-kernel",
                {
                    "request_id": request.request_id,
                    "plan_id": plan.plan_id,
                    "step_id": step.step_id,
                    "specialist_id": step.specialist_id,
                    "decision": decision.decision.value,
                    "rule_id": decision.rule_id,
                    "reason": decision.reason,
                    "context_hash": ethics_context.context_hash(),
                    "evaluation": ethics_evaluation.to_payload(),
                },
                parent_event_ids=[step_started_event_id],
            )
        )
        failure_reason = f"{decision.decision.value}: {decision.rule_id}: {decision.reason}"
        aborted = self.ledger.append(
            ProvenanceEvent(
                EventType.CHAIN_ABORTED,
                request.tenant_id,
                "chain-executor",
                {
                    "request_id": request.request_id,
                    "plan_id": plan.plan_id,
                    "failed_step_id": step.step_id,
                    "failure_reason": failure_reason,
                },
                parent_event_ids=[ethics_event.event_id],
            )
        )
        event_ids.extend([ethics_event.event_id, aborted.event_id])
        return ChainExecutionResult(
            chain_event_id=chain_event_id,
            responses=responses,
            event_ids=event_ids,
            artifact_event_ids=artifact_event_ids,
            episodic_event_ids=episodic_event_ids,
            aborted=True,
            failure_reason=failure_reason,
            failed_step_id=step.step_id,
        )


def _bind_step_request(
    *,
    request: SpecialistRequest,
    plan: ChainPlan,
    step: ChainStep,
    chain_event_id: str,
    step_started_event_id: str,
    responses_by_step: dict[str, SpecialistResponse],
) -> SpecialistRequest:
    upstream = {
        step_id: _response_summary(responses_by_step[step_id]) for step_id in step.inputs_from
    }
    input_payload = {
        **request.input_payload,
        **step.parameters,
        "upstream": upstream,
        "chain": {
            "plan_id": plan.plan_id,
            "chain_event_id": chain_event_id,
            "step_id": step.step_id,
            "inputs_from": list(step.inputs_from),
        },
    }
    context = {
        **request.provenance_context,
        "parent_event_id": step_started_event_id,
        "chain_event_id": chain_event_id,
        "chain_step_id": step.step_id,
    }
    return SpecialistRequest(
        request_id=f"{request.request_id}:{step.step_id}",
        tenant_id=request.tenant_id,
        task_type=request.task_type,
        input_payload=input_payload,
        required_output_schema=request.required_output_schema,
        provenance_context=context,
    )


def _response_summary(response: SpecialistResponse) -> dict[str, Any]:
    artifact_records = []
    if response.cost_metadata:
        raw_records = response.cost_metadata.get("artifact_records", [])
        if isinstance(raw_records, list):
            artifact_records = [record for record in raw_records if isinstance(record, dict)]
    return {
        "specialist_id": response.specialist_id,
        "claim_ids": [claim.claim_id for claim in response.claims],
        "claims": [
            {
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "claim_text": claim.claim_text,
                "confidence": claim.confidence,
                "artifact_ids": list(claim.artifact_ids),
            }
            for claim in response.claims
        ],
        "artifact_ids": _response_artifact_ids(response),
        "artifact_records": artifact_records,
    }


def _ethics_context_for_step(
    *,
    request: SpecialistRequest,
    step: ChainStep,
    chain_event_id: str,
    step_started_event_id: str,
    specialist_registered: bool,
) -> EthicsContext:
    risk_tags = _collect_tags(
        "risk_tags", step.parameters, request.input_payload, request.provenance_context
    )
    learning_event_token = _learning_event_token(request)
    return EthicsContext(
        tenant_id=request.tenant_id,
        actor_id="chain-executor",
        request_id=request.request_id,
        action_type="specialist_call",
        provenance_parent_event_id=step_started_event_id,
        chain_id=chain_event_id,
        step_id=step.step_id,
        specialist_id=step.specialist_id,
        operation=_first_string(
            "operation", step.parameters, request.input_payload, request.provenance_context
        )
        or _first_string(
            "command", step.parameters, request.input_payload, request.provenance_context
        ),
        target_tenant_id=_first_string(
            "target_tenant_id", step.parameters, request.input_payload, request.provenance_context
        ),
        cross_tenant_grant_id=_first_string(
            "cross_tenant_grant_id",
            step.parameters,
            request.input_payload,
            request.provenance_context,
        ),
        writes_learning_memory=_truthy(
            "writes_learning_memory",
            step.parameters,
            request.input_payload,
            request.provenance_context,
        )
        or _first_string(
            "memory_write_kind", step.parameters, request.input_payload, request.provenance_context
        )
        in {"lora", "training", "learning"},
        learning_event_token=learning_event_token,
        governance_approval_id=_first_string(
            "governance_approval_id",
            step.parameters,
            request.input_payload,
            request.provenance_context,
        ),
        destructive=_truthy(
            "destructive", step.parameters, request.input_payload, request.provenance_context
        ),
        risk_tags=frozenset(risk_tags),
        specialist_registered=specialist_registered,
    )


def _first_string(key: str, *sources: dict[str, Any]) -> str | None:
    for source in sources:
        value = source.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def _truthy(key: str, *sources: dict[str, Any]) -> bool:
    for source in sources:
        value = source.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"1", "true", "yes"}:
            return True
    return False


def _collect_tags(key: str, *sources: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for source in sources:
        value = source.get(key)
        if isinstance(value, str):
            tags.append(value)
        elif isinstance(value, (list, tuple, set, frozenset)):
            tags.extend(str(item) for item in value)
    return tags


def _learning_event_token(request: SpecialistRequest) -> LearningEventToken | None:
    for source in (request.provenance_context, request.input_payload):
        value = source.get("learning_event_token")
        if isinstance(value, LearningEventToken):
            return value
    return None


def _response_artifact_ids(response: SpecialistResponse) -> list[str]:
    artifact_ids: list[str] = []
    for claim in response.claims:
        artifact_ids.extend(claim.artifact_ids)
    for record in (response.cost_metadata or {}).get("artifact_records", []):
        payload = _artifact_payload(record)
        artifact_id = payload.get("artifact_id")
        if artifact_id:
            artifact_ids.append(str(artifact_id))
    return sorted(set(artifact_ids))


def _record_artifacts(
    ledger: ProvenanceLedger,
    *,
    response: SpecialistResponse,
    tenant_id: str,
    parent_event_id: str,
    episodic_entry_id: str | None = None,
) -> list[str]:
    event_ids: list[str] = []
    for record in (response.cost_metadata or {}).get("artifact_records", []):
        payload = _artifact_payload(record)
        payload["parent_event_id"] = parent_event_id
        if episodic_entry_id is not None:
            payload["episodic_entry_id"] = episodic_entry_id
        event = ledger.append(
            ProvenanceEvent(
                EventType.ARTIFACT_PRODUCED,
                tenant_id,
                response.specialist_id,
                payload,
                parent_event_ids=[parent_event_id],
            )
        )
        event_ids.append(event.event_id)
    return event_ids


def _artifact_payload(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_payload"):
        return dict(record.to_payload())
    return dict(record)


def _request_parent_ids(request: SpecialistRequest) -> list[str]:
    parent_event_id = request.provenance_context.get("parent_event_id")
    return [str(parent_event_id)] if parent_event_id else []


def _write_episodic_entry(
    episodic_store: EpisodicStore,
    ledger: ProvenanceLedger,
    *,
    request: SpecialistRequest,
    plan: ChainPlan,
    step: ChainStep,
    response: SpecialistResponse,
    parent_event_id: str,
    artifact_ids: list[str],
) -> tuple[str, str]:
    entry_id = f"episodic-{request.request_id}-{step.step_id}"
    entry = episodic_store.write_text(
        entry_id=entry_id,
        tenant_id=request.tenant_id,
        raw_text=_episodic_raw_text(request=request, plan=plan, step=step, response=response),
        metadata={
            "specialist_id": step.specialist_id,
            "source_id": response.specialist_id,
            "anchor_id": request.provenance_context.get("trust_anchor_id"),
            "timestamp": response.claims[0].created_at.isoformat() if response.claims else "",
            "access_count": 0,
            "freshness": 1.0,
            "success_status": "succeeded",
            "request_id": request.request_id,
            "plan_id": plan.plan_id,
            "step_id": step.step_id,
        },
        chain_id=plan.plan_id,
        artifact_ids=artifact_ids,
    )
    event = ledger.append(
        ProvenanceEvent(
            EventType.EPISODIC_ENTRY_WRITTEN,
            request.tenant_id,
            "chain-executor",
            {
                "entry_id": entry.entry_id,
                "request_id": request.request_id,
                "plan_id": plan.plan_id,
                "step_id": step.step_id,
                "specialist_id": step.specialist_id,
                "chain_id": entry.chain_id,
                "artifact_ids": list(entry.artifact_ids),
                "embedding_dimensions": len(entry.embedding),
            },
            parent_event_ids=[parent_event_id],
        )
    )
    return entry.entry_id, event.event_id


def _episodic_raw_text(
    *,
    request: SpecialistRequest,
    plan: ChainPlan,
    step: ChainStep,
    response: SpecialistResponse,
) -> str:
    payload = {
        "request_id": request.request_id,
        "task_type": request.task_type,
        "plan_id": plan.plan_id,
        "step_id": step.step_id,
        "specialist_id": step.specialist_id,
        "input_payload": request.input_payload,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "claim_text": claim.claim_text,
                "confidence": claim.confidence,
                "artifact_ids": list(claim.artifact_ids),
            }
            for claim in response.claims
        ],
        "artifact_ids": _response_artifact_ids(response),
        "success_status": "succeeded",
    }
    return json_dumps(payload)


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
