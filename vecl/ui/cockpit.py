from __future__ import annotations

import base64
import importlib.util
import json
import os
import re
import secrets
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from vecl._compat import UTC
from vecl._env import load_dotenv_if_present
from vecl._paths import environment_directory
from vecl.ethics import EthicsKernel, default_ethics_rules
from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb._llm_inference import DEFAULT_ROUTING_MODEL_ID
from vecl.qb.chain_executor import ChainExecutionResult, ChainExecutor
from vecl.qb.model_driver import (
    ModelDriver,
    ModelDriverResult,
    ModelRouteResult,
    TextGenerationDriver,
    resolve_model_driver,
)
from vecl.qb.planner import ChainPlan, ChainStep
from vecl.qb.prompted_router import (
    RoutingDecision,
    RoutingParseError,
    build_routing_prompt_from_cards,
    parse_routing_response,
)
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistRequest, SpecialistResponse
from vecl.qb.specialist_contracts import CONFIG_OPERATIONS
from vecl.qb.tool_call import (
    ToolCallParseError,
    ToolCallProposal,
    ToolCallRunResult,
    ToolCallValidationConfig,
    build_tool_call_prompt,
    execute_tool_call,
    parse_tool_call_response,
)
from vecl.specialists.artifacts import ArtifactRecord, ContentAddressedStore
from vecl.specialists.blast_specialist import BLASTSpecialist, resolve_blast_binary
from vecl.specialists.stockfish import StockfishSpecialist, resolve_stockfish_binary
from vecl.specialists.sympy_specialist import SymPySpecialist
from vecl.specialists.terraform_specialist import TerraformSpecialist, resolve_terraform_binary
from vecl.specialists.timesfm_specialist import TimesFMSpecialist

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
LOCAL_TENANT_ID = "vecl-ui-local"
ROUTE_KIND = "route"
TOOL_CALL_KIND = "tool_call"
MANUAL_TOOL_CALL_KIND = "manual_tool_call"
SYNTHESIS_KIND = "synthesis"
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
MAX_UI_STOCKFISH_DEPTH = 20


@dataclass(frozen=True)
class Availability:
    available: bool
    status: str
    detail: str
    can_execute: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "detail": self.detail,
            "can_execute": self.available and self.can_execute,
        }


SpecialistFactory = Callable[[ContentAddressedStore], Specialist]
AvailabilityCheck = Callable[[], Availability]
DriverFactory = Callable[[str], ModelDriver]


@dataclass(frozen=True)
class SpecialistSpec:
    card: SpecialistCard
    factory: SpecialistFactory
    availability_check: AvailabilityCheck
    expected_artifact_type: str
    notes: str = ""

    def status_payload(self) -> dict[str, Any]:
        availability = self.availability_check()
        return {
            "specialist_id": self.card.specialist_id,
            "supported_task_types": sorted(self.card.supported_task_types),
            "trust_requirements": dict(self.card.trust_requirements),
            "cost_hint": self.card.cost_hint,
            "latency_hint": self.card.latency_hint,
            "version": self.card.version,
            "effective_trust": self.card.effective_trust,
            "description": self.card.description,
            "expected_artifact_type": self.expected_artifact_type,
            "notes": self.notes,
            **availability.to_dict(),
        }


@dataclass
class UIStep:
    step_id: str
    label: str
    status: str
    detail: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "created_at": self.created_at.isoformat(),
            "duration_ms": self.duration_ms,
        }


@dataclass
class UIModelCall:
    call_id: str
    kind: str
    provider: str
    model_id: str
    raw_text: str
    parsed: dict[str, Any]
    prompt: str
    usage: dict[str, Any]
    metadata: dict[str, Any]
    latency_ms: int | None
    finish_reason: str | None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_response(
        cls,
        *,
        kind: str,
        response: ModelDriverResult,
        prompt: str,
        parsed: dict[str, Any] | None = None,
    ) -> UIModelCall:
        return cls(
            call_id=f"call-{uuid4()}",
            kind=kind,
            provider=response.provider,
            model_id=response.model_id,
            raw_text=response.raw_text,
            parsed=dict(parsed or {}),
            prompt=prompt,
            usage=dict(response.usage),
            metadata=dict(response.metadata),
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
        )

    def estimated_cost_usd(self) -> float | None:
        value = self.metadata.get("estimated_cost_usd")
        if value is None:
            return None
        return float(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "kind": self.kind,
            "provider": self.provider,
            "model_id": self.model_id,
            "raw_text": self.raw_text,
            "parsed": self.parsed,
            "prompt": self.prompt,
            "usage": self.usage,
            "metadata": self.metadata,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "estimated_cost_usd": self.estimated_cost_usd(),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class UIArtifact:
    record: ArtifactRecord
    event_id: str

    def to_dict(self) -> dict[str, Any]:
        payload = self.record.to_payload()
        payload["event_id"] = self.event_id
        return payload


@dataclass
class RunState:
    run_id: str
    session_id: str
    prompt: str
    requested_driver: str
    status: str = "queued"
    task_type: str = ""
    selected_specialist: str | None = None
    proposed_payload: dict[str, Any] = field(default_factory=dict)
    validation_result: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    active_step: str = ""
    errors: list[str] = field(default_factory=list)
    steps: list[UIStep] = field(default_factory=list)
    model_calls: list[UIModelCall] = field(default_factory=list)
    artifacts: list[UIArtifact] = field(default_factory=list)
    provenance_events: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def cumulative_cost_usd(self) -> float | None:
        values = [call.estimated_cost_usd() for call in self.model_calls]
        costs = [value for value in values if value is not None]
        if not costs:
            return None
        return round(sum(costs), 8)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "prompt": self.prompt,
            "requested_driver": self.requested_driver,
            "status": self.status,
            "task_type": self.task_type,
            "selected_specialist": self.selected_specialist,
            "proposed_payload": self.proposed_payload,
            "validation_result": self.validation_result,
            "final_answer": self.final_answer,
            "active_step": self.active_step,
            "errors": list(self.errors),
            "steps": [step.to_dict() for step in self.steps],
            "model_calls": [call.to_dict() for call in self.model_calls],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "provenance_events": list(self.provenance_events),
            "cumulative_cost_usd": self.cumulative_cost_usd(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class SessionState:
    session_id: str
    title: str
    run_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "run_ids": list(self.run_ids),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class CockpitBackend:
    def __init__(
        self,
        *,
        artifact_root: Path | str | None = None,
        driver_factory: DriverFactory | None = None,
        specialist_specs: Sequence[SpecialistSpec] | None = None,
    ) -> None:
        root = artifact_root or environment_directory(
            "VECL_UI_ARTIFACT_ROOT", prefix="vecl-ui-artifacts-"
        )
        self.artifact_store = ContentAddressedStore(root)
        self._driver_factory = driver_factory or _default_driver_factory
        self._specialist_specs = {
            spec.card.specialist_id: spec
            for spec in (specialist_specs or default_specialist_specs())
        }
        self._runs: dict[str, RunState] = {}
        self._sessions: dict[str, SessionState] = {}
        self._artifact_index: dict[str, UIArtifact] = {}
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "vecl-ui-cockpit",
            "time": datetime.now(UTC).isoformat(),
            "artifact_root": str(self.artifact_store._root_path),
        }

    def list_drivers(self) -> list[dict[str, Any]]:
        load_dotenv_if_present()
        return [_driver_status_openai(), _driver_status_anthropic(), _driver_status_gemma()]

    def list_specialists(self) -> list[dict[str, Any]]:
        return [
            spec.status_payload()
            for spec in sorted(
                self._specialist_specs.values(), key=lambda item: item.card.specialist_id
            )
        ]

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            sessions = sorted(
                self._sessions.values(), key=lambda session: session.updated_at, reverse=True
            )
            return [session.to_dict() for session in sessions]

    def create_session(self, title: str = "New conversation") -> dict[str, Any]:
        session = SessionState(session_id=f"sess-{uuid4()}", title=title)
        with self._lock:
            self._sessions[session.session_id] = session
        return session.to_dict()

    def submit_chat(
        self,
        *,
        prompt: str,
        driver_provider: str,
        session_id: str | None = None,
        payload_override: dict[str, Any] | None = None,
        run_async: bool = True,
    ) -> dict[str, Any]:
        cleaned = prompt.strip()
        if not cleaned:
            raise ValueError("prompt must be non-empty")
        session = self._ensure_session(session_id, cleaned)
        run = RunState(
            run_id=f"run-{uuid4()}",
            session_id=session.session_id,
            prompt=cleaned,
            requested_driver=driver_provider.strip().lower() or "openai",
        )
        with self._lock:
            self._runs[run.run_id] = run
            session.run_ids.append(run.run_id)
            session.updated_at = datetime.now(UTC)
        if run_async:
            thread = threading.Thread(
                target=self._execute_run,
                args=(run.run_id, payload_override),
                name=run.run_id,
            )
            thread.daemon = True
            thread.start()
        else:
            self._execute_run(run.run_id, payload_override)
        return self.get_run(run.run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._require_run(run_id)
            return run.to_dict()

    def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            run = self._require_run(run_id)
            return list(run.provenance_events)

    def get_artifact_content(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            artifact = self._artifact_index.get(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        content = self.artifact_store.read_bytes(artifact.record)
        output_format = artifact.record.output_format.lower()
        raw_base64 = base64.b64encode(content).decode("ascii")
        try:
            text = content.decode()
            is_binary = False
        except UnicodeDecodeError:
            text = None
            is_binary = True
        if output_format in {"png", "jpg", "jpeg", "gif", "webp"}:
            is_binary = True
        return {
            "artifact": artifact.to_dict(),
            "is_binary": is_binary,
            "text": text,
            "raw_base64": raw_base64,
            "byte_length": len(content),
        }

    def _execute_run(self, run_id: str, payload_override: dict[str, Any] | None) -> None:
        ledger = ProvenanceLedger()
        try:
            kernel, _ = _ethics_kernel_for_run(ledger)
            self._mark_run(run_id, status="running", started_at=datetime.now(UTC))
            run = self._run_snapshot(run_id)
            driver = self._resolve_driver(run_id, run.requested_driver)
            self._append_step(
                run_id,
                "Driver selected",
                "completed",
                f"{driver.provider} / {driver.model_id}",
            )
            request_payload = build_request_payload(run.prompt, payload_override or {})
            self._set_payload(run_id, request_payload.task_type, request_payload.to_payload())
            if request_payload.warnings:
                self._append_step(
                    run_id,
                    "Payload warning",
                    "review",
                    "; ".join(request_payload.warnings),
                )
            request_event = ledger.append(
                ProvenanceEvent(
                    EventType.EVIDENCE_INGESTED,
                    LOCAL_TENANT_ID,
                    "vecl-ui-cockpit",
                    {
                        "run_id": run_id,
                        "prompt_hash": stable_hash(run.prompt),
                        "task_type": request_payload.task_type,
                    },
                )
            )
            self._append_event(run_id, request_event)
            request = SpecialistRequest(
                request_id=run_id,
                tenant_id=LOCAL_TENANT_ID,
                task_type=request_payload.task_type,
                input_payload=request_payload.payload,
                required_output_schema={},
                provenance_context={"parent_event_id": request_event.event_id},
            )
            tool_call = self._execute_tool_call_path(
                run_id,
                ledger,
                kernel,
                driver,
                request,
                request_payload,
            )
            if tool_call is None:
                self._finish_without_tool(
                    run_id,
                    ledger,
                    request,
                    parent_event_id=request_event.event_id,
                    answer="Tool-call proposal failed or could not be parsed; no specialist was executed.",
                )
                return
            selected_id = tool_call.proposal.specialist_id or "none"
            final_request = tool_call.validated.request if tool_call.validated else request
            self._synthesize_final(
                run_id,
                ledger,
                driver,
                final_request,
                tool_call.execution,
                selected_id,
            )
        except Exception as exc:
            self._append_error(run_id, str(exc))
            self._mark_run(
                run_id,
                status="failed",
                completed_at=datetime.now(UTC),
                active_step="",
            )

    def _resolve_driver(self, run_id: str, provider: str) -> ModelDriver:
        self._append_step(
            run_id,
            f"Resolving {provider} driver",
            "running",
        )
        driver = self._driver_factory(provider)
        return driver

    def _execute_tool_call_path(
        self,
        run_id: str,
        ledger: ProvenanceLedger,
        kernel: EthicsKernel,
        driver: ModelDriver,
        request: SpecialistRequest,
        request_payload: RequestPayload,
    ) -> ToolCallRunResult | None:
        cards = {spec.card.specialist_id: spec.card for spec in self._specialist_specs.values()}
        proposal_result = self._tool_call_proposal(
            run_id,
            driver,
            request,
            request_payload,
            cards,
        )
        if proposal_result is None:
            return None
        proposal, model_response = proposal_result
        self._set_tool_call_proposal(run_id, proposal)
        self._set_selected_specialist(run_id, proposal.specialist_id)
        self._append_step(
            run_id,
            f"Proposed specialist: {proposal.specialist_id or 'none'}",
            "completed",
            proposal.reasoning,
        )

        self._set_active(run_id, "Validating tool call...")
        self._append_step(run_id, "Validating tool call", "running")
        specialists = self._specialists_for_tool_call(run_id, proposal.specialist_id)
        config = _tool_call_validation_config(request)
        executor = ChainExecutor(
            ledger=ledger,
            specialists=dict(specialists),
            ethics_kernel=kernel,
        )
        result = execute_tool_call(
            proposal=proposal,
            model_response=model_response,
            cards=cards,
            specialists=specialists,
            ledger=ledger,
            config=config,
            executor=executor,
        )
        self._append_tool_call_events(run_id, ledger, result)
        if result.validated is None:
            reason = "; ".join(result.warnings) or "tool-call validation failed"
            self._set_validation(
                run_id,
                {
                    "ok": False,
                    "phase": "tool_call_validation",
                    "reason": reason,
                    "warnings": list(result.warnings),
                },
            )
            self._append_step(run_id, "Validating tool call", "failed", reason)
            self._append_error(run_id, reason)
            return result

        validated = result.validated
        self._set_task_type(run_id, validated.request.task_type)
        self._set_specialist_request_payload(run_id, validated.request)
        self._set_validation(
            run_id,
            {
                "ok": True,
                "phase": "tool_call_validation",
                "reason": "tool call validated",
                "specialist_id": validated.specialist_id,
                "warnings": list(result.warnings),
            },
        )
        detail = "; ".join(result.warnings) if result.warnings else "canonical payload accepted"
        self._append_step(run_id, "Validating tool call", "completed", detail)
        execution = result.execution
        if execution is None:
            return result
        self._append_step(
            run_id,
            _running_specialist_label(
                validated.specialist_id, dict(validated.request.input_payload)
            ),
            "completed" if not execution.aborted else "failed",
        )
        self._index_artifacts(run_id, ledger, execution.artifact_event_ids)
        self._append_ethics_status(run_id, ledger, execution)
        if execution.aborted:
            self._append_step(
                run_id,
                "Specialist execution aborted",
                "failed",
                execution.failure_reason or "chain aborted",
            )
            self._append_error(run_id, execution.failure_reason or "chain aborted")
        else:
            for artifact in self._run_snapshot(run_id).artifacts:
                self._append_step(
                    run_id,
                    f"Artifact produced: {artifact.record.output_format}",
                    "completed",
                    artifact.record.artifact_id,
                )
            self._append_step(run_id, f"{validated.specialist_id} completed", "completed")
        return result

    def _tool_call_proposal(
        self,
        run_id: str,
        driver: ModelDriver,
        request: SpecialistRequest,
        request_payload: RequestPayload,
        cards: dict[str, SpecialistCard],
    ) -> tuple[ToolCallProposal, ModelDriverResult] | None:
        if request_payload.source == "explicit_payload_editor":
            self._append_step(
                run_id,
                "Using explicit payload editor JSON",
                "completed",
                "manual payload is authoritative",
            )
            proposal = _proposal_from_explicit_payload(
                request_payload,
                tuple(cards.values()),
            )
            response = _manual_tool_call_response(proposal)
            self._append_model_call(
                run_id,
                UIModelCall.from_response(
                    kind=MANUAL_TOOL_CALL_KIND,
                    response=response,
                    prompt="payload editor JSON",
                    parsed=tool_call_proposal_payload(proposal),
                ),
            )
            return proposal, response

        cards_sequence = sorted(cards.values(), key=lambda card: card.specialist_id)
        prompt = build_tool_call_prompt(request, cards_sequence)
        self._set_active(run_id, f"Authoring tool call with {driver.provider}...")
        self._append_step(run_id, f"Authoring tool call with {driver.provider}", "running")
        try:
            response = driver.synthesize(prompt)
        except Exception as exc:
            self._append_step(run_id, "Tool-call authoring failed", "failed", str(exc))
            self._append_error(run_id, f"tool-call authoring failed: {exc}")
            return None
        try:
            proposal = parse_tool_call_response(response.raw_text)
        except ToolCallParseError as exc:
            self._append_model_call(
                run_id,
                UIModelCall.from_response(
                    kind=TOOL_CALL_KIND,
                    response=response,
                    prompt=prompt,
                    parsed={"parse_error": str(exc)},
                ),
            )
            self._append_step(run_id, "Tool-call parse failure", "failed", str(exc))
            self._append_error(run_id, f"tool-call parse failure: {exc}")
            return None
        self._append_model_call(
            run_id,
            UIModelCall.from_response(
                kind=TOOL_CALL_KIND,
                response=response,
                prompt=prompt,
                parsed=tool_call_proposal_payload(proposal),
            ),
        )
        return proposal, response

    def _specialists_for_tool_call(
        self, run_id: str, specialist_id: str | None
    ) -> dict[str, Specialist]:
        if specialist_id is None:
            return {}
        spec = self._specialist_specs.get(specialist_id)
        if spec is None:
            return {}
        availability = spec.availability_check()
        if not availability.available or not availability.can_execute:
            self._append_step(run_id, f"{specialist_id} unavailable", "failed", availability.detail)
            self._append_error(run_id, availability.detail)
            return {}
        self._append_step(run_id, f"{specialist_id} available", "completed", availability.detail)
        return {specialist_id: spec.factory(self.artifact_store)}

    def _route(
        self,
        run_id: str,
        ledger: ProvenanceLedger,
        driver: ModelDriver,
        request: SpecialistRequest,
    ) -> ModelRouteResult | None:
        cards = [spec.card for spec in self._specialist_specs.values()]
        prompt = build_routing_prompt_from_cards(cards, request)
        self._set_active(run_id, f"Routing with {driver.provider}...")
        self._append_step(run_id, f"Routing with {driver.provider}", "running")
        try:
            route = route_with_raw(driver, request, cards, prompt)
        except RoutingParseWithResponseError as exc:
            self._append_model_call(
                run_id,
                UIModelCall.from_response(
                    kind=ROUTE_KIND,
                    response=exc.response,
                    prompt=prompt,
                    parsed={"parse_error": str(exc)},
                ),
            )
            event = ledger.append(
                ProvenanceEvent(
                    EventType.ROUTING_FALLBACK,
                    request.tenant_id,
                    "vecl-ui-cockpit",
                    {
                        "request_id": request.request_id,
                        "query_hash": stable_hash(request.input_payload),
                        "parse_failure_reason": str(exc),
                        "fallback_specialist_id": None,
                        "model_id": driver.model_id,
                    },
                    parent_event_ids=[str(request.provenance_context["parent_event_id"])],
                )
            )
            self._append_event(run_id, event)
            self._append_step(run_id, "Parse failure", "failed", str(exc))
            self._append_error(run_id, f"routing parse failure: {exc}")
            return None
        except Exception as exc:
            self._append_step(run_id, "Routing failed", "failed", str(exc))
            self._append_error(run_id, f"routing failed: {exc}")
            return None
        self._append_model_call(
            run_id,
            UIModelCall.from_response(
                kind=ROUTE_KIND,
                response=route.response,
                prompt=prompt,
                parsed=routing_decision_payload(route.decision),
            ),
        )
        event = ledger.append(
            ProvenanceEvent(
                EventType.LLM_ROUTING_DECIDED,
                request.tenant_id,
                "vecl-ui-cockpit",
                {
                    "request_id": request.request_id,
                    "query_hash": stable_hash(request.input_payload),
                    "chosen_specialist_id": route.decision.specialist_id,
                    "model_id": route.response.model_id,
                    "provider": route.response.provider,
                    "reasoning": route.decision.reasoning,
                    "confidence": route.decision.confidence,
                },
                parent_event_ids=[str(request.provenance_context["parent_event_id"])],
            )
        )
        self._append_event(run_id, event)
        detail = route.decision.specialist_id or "no specialist"
        self._append_step(run_id, f"Proposed specialist: {detail}", "completed")
        return route

    def _bind_request_to_selected_specialist(
        self, run_id: str, request: SpecialistRequest, selected_id: str
    ) -> SpecialistRequest:
        spec = self._specialist_specs.get(selected_id)
        if spec is None:
            self._set_validation(run_id, {"ok": False, "reason": "unknown specialist"})
            return request
        if request.task_type in spec.card.supported_task_types:
            self._set_validation(run_id, {"ok": True, "reason": "task_type accepted"})
            return request
        if request.task_type == "general":
            task_type = sorted(spec.card.supported_task_types)[0]
            self._append_step(
                run_id,
                "Bound generic request",
                "completed",
                f"task_type={task_type}",
            )
            self._set_task_type(run_id, task_type)
            self._set_validation(run_id, {"ok": True, "reason": "bound generic task_type"})
            return SpecialistRequest(
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                task_type=task_type,
                input_payload=request.input_payload,
                required_output_schema=request.required_output_schema,
                provenance_context=request.provenance_context,
            )
        self._set_validation(
            run_id,
            {
                "ok": False,
                "reason": (
                    f"{selected_id} does not support task_type={request.task_type}; "
                    f"supports {sorted(spec.card.supported_task_types)}"
                ),
            },
        )
        return request

    def _execute_specialist(
        self,
        run_id: str,
        ledger: ProvenanceLedger,
        kernel: EthicsKernel,
        request: SpecialistRequest,
        selected_id: str,
    ) -> ChainExecutionResult | None:
        spec = self._specialist_specs.get(selected_id)
        if spec is None:
            self._append_step(run_id, "Validating tool call", "failed", "unknown specialist")
            self._append_error(run_id, f"unknown specialist_id: {selected_id}")
            return None
        availability = spec.availability_check()
        self._append_step(run_id, "Validating tool call", "completed", availability.detail)
        if not availability.available or not availability.can_execute:
            self._append_step(run_id, f"{selected_id} unavailable", "failed", availability.detail)
            self._append_error(run_id, availability.detail)
            return None
        specialist = spec.factory(self.artifact_store)
        plan = ChainPlan(
            f"ui-single-step-{selected_id}",
            request.task_type,
            (
                ChainStep(
                    "execute",
                    selected_id,
                    expected_artifact_type=spec.expected_artifact_type,
                ),
            ),
        )
        self._set_active(run_id, f"Running {selected_id}...")
        if selected_id == "terraform":
            operation = request.input_payload.get("operation") or request.input_payload.get(
                "command"
            )
            self._append_step(run_id, f"Running Terraform {operation or 'plan'}", "running")
        else:
            self._append_step(run_id, f"Running {selected_id}", "running")
        executor = ChainExecutor(
            ledger=ledger,
            specialists={selected_id: specialist},
            ethics_kernel=kernel,
        )
        execution = executor.execute(plan, request)
        self._append_execution_events(run_id, ledger, execution)
        self._index_artifacts(run_id, ledger, execution.artifact_event_ids)
        self._append_ethics_status(run_id, ledger, execution)
        if execution.aborted:
            self._append_step(
                run_id,
                "Specialist execution aborted",
                "failed",
                execution.failure_reason or "chain aborted",
            )
            self._append_error(run_id, execution.failure_reason or "chain aborted")
        else:
            for artifact in self._run_snapshot(run_id).artifacts:
                self._append_step(
                    run_id,
                    f"Artifact produced: {artifact.record.output_format}",
                    "completed",
                    artifact.record.artifact_id,
                )
            self._append_step(run_id, f"{selected_id} completed", "completed")
        return execution

    def _synthesize_final(
        self,
        run_id: str,
        ledger: ProvenanceLedger,
        driver: ModelDriver,
        request: SpecialistRequest,
        execution: ChainExecutionResult | None,
        selected_id: str,
    ) -> None:
        self._set_active(run_id, "Synthesizing final answer...")
        self._append_step(run_id, "Synthesizing final answer", "running")
        prompt = self._build_synthesis_prompt(run_id, request, execution, selected_id)
        try:
            response = driver.synthesize(prompt)
        except Exception as exc:
            answer = f"Specialist trace is available, but final model synthesis failed: {exc}"
            self._append_error(run_id, str(exc))
            self._record_final_event(
                run_id,
                ledger,
                request,
                selected_id=selected_id,
                parent_event_id=_parent_for_final(request, execution),
                status="synthesis_failed",
            )
            self._finish(run_id, answer=answer, status="failed")
            return
        self._append_model_call(
            run_id,
            UIModelCall.from_response(
                kind=SYNTHESIS_KIND,
                response=response,
                prompt=prompt,
                parsed={"final_answer": response.raw_text},
            ),
        )
        if not response.raw_text.strip():
            message = (
                "Final model synthesis returned empty text. "
                f"Raw response: {response.raw_text!r}; finish_reason={response.finish_reason!r}"
            )
            self._append_error(run_id, message)
            self._record_final_event(
                run_id,
                ledger,
                request,
                selected_id=selected_id,
                parent_event_id=_parent_for_final(request, execution),
                status="synthesis_failed",
            )
            self._finish(run_id, answer=message, status="failed")
            return
        self._record_final_event(
            run_id,
            ledger,
            request,
            selected_id=selected_id,
            parent_event_id=_parent_for_final(request, execution),
            status="completed" if execution is None or not execution.aborted else "aborted",
        )
        final_status = "completed" if execution is None or not execution.aborted else "completed"
        self._finish(run_id, answer=response.raw_text, status=final_status)

    def _finish_without_tool(
        self,
        run_id: str,
        ledger: ProvenanceLedger,
        request: SpecialistRequest,
        *,
        parent_event_id: str,
        answer: str,
    ) -> None:
        self._record_final_event(
            run_id,
            ledger,
            request,
            selected_id=None,
            parent_event_id=parent_event_id,
            status="no_tool",
        )
        self._finish(run_id, answer=answer, status="completed")

    def _build_synthesis_prompt(
        self,
        run_id: str,
        request: SpecialistRequest,
        execution: ChainExecutionResult | None,
        selected_id: str,
    ) -> str:
        run = self._run_snapshot(run_id)
        artifact_context: list[dict[str, Any]] = []
        for artifact in run.artifacts:
            try:
                content = self.artifact_store.read_bytes(artifact.record)
                text = content.decode(errors="replace")
            except Exception as exc:
                text = f"<artifact read failed: {exc}>"
            artifact_context.append(
                {
                    "artifact_id": artifact.record.artifact_id,
                    "output_hash": artifact.record.output_hash,
                    "output_format": artifact.record.output_format,
                    "producer_specialist_id": artifact.record.producer_specialist_id,
                    "content": text[:12000],
                }
            )
        response_context = []
        if execution is not None:
            response_context = [_response_payload(response) for response in execution.responses]
        payload = {
            "user_prompt": request.input_payload.get("query", run.prompt),
            "selected_specialist": selected_id,
            "task_type": request.task_type,
            "tool_call": run.proposed_payload,
            "tool_call_validation": run.validation_result,
            "chain_aborted": execution.aborted if execution is not None else None,
            "failure_reason": execution.failure_reason if execution is not None else None,
            "specialist_responses": response_context,
            "artifacts": artifact_context,
        }
        return (
            "You are VECL-QB's final synthesis driver. Use only the verified specialist "
            "responses and exact artifacts in this JSON context. If the chain was refused, "
            "aborted, or produced no artifact, say that plainly. Do not claim Terraform "
            "applied or destroyed anything; Terraform is plan/validate only here.\n\n"
            + json.dumps(payload, sort_keys=True, indent=2, default=str)
        )

    def _record_final_event(
        self,
        run_id: str,
        ledger: ProvenanceLedger,
        request: SpecialistRequest,
        *,
        selected_id: str | None,
        parent_event_id: str,
        status: str,
    ) -> None:
        event = ledger.append(
            ProvenanceEvent(
                EventType.FINAL_RESPONSE_RECORDED,
                request.tenant_id,
                "vecl-ui-cockpit",
                {
                    "request_id": request.request_id,
                    "run_id": run_id,
                    "selected_specialist": selected_id,
                    "status": status,
                    "model_call_ids": [
                        call.call_id for call in self._run_snapshot(run_id).model_calls
                    ],
                    "artifact_ids": [
                        artifact.record.artifact_id
                        for artifact in self._run_snapshot(run_id).artifacts
                    ],
                },
                parent_event_ids=[parent_event_id],
            )
        )
        self._append_event(run_id, event)

    def _append_execution_events(
        self, run_id: str, ledger: ProvenanceLedger, execution: ChainExecutionResult
    ) -> None:
        for event_id in execution.event_ids:
            event = ledger.get(event_id)
            if event is not None:
                self._append_event(run_id, event)

    def _append_tool_call_events(
        self, run_id: str, ledger: ProvenanceLedger, result: ToolCallRunResult
    ) -> None:
        if result.proposal_event_id is not None:
            event = ledger.get(result.proposal_event_id)
            if event is not None:
                self._append_event(run_id, event)
        if result.execution is not None:
            self._append_execution_events(run_id, ledger, result.execution)

    def _index_artifacts(
        self, run_id: str, ledger: ProvenanceLedger, artifact_event_ids: Sequence[str]
    ) -> None:
        for event_id in artifact_event_ids:
            event = ledger.get(event_id)
            if event is None:
                continue
            record = ArtifactRecord.from_payload(event.payload)
            artifact = UIArtifact(record=record, event_id=event.event_id)
            with self._lock:
                run = self._require_run(run_id)
                run.artifacts.append(artifact)
                run.updated_at = datetime.now(UTC)
                self._artifact_index[record.artifact_id] = artifact

    def _append_ethics_status(
        self, run_id: str, ledger: ProvenanceLedger, execution: ChainExecutionResult
    ) -> None:
        event_types = {
            ledger.require(event_id).event_type
            for event_id in execution.event_ids
            if ledger.get(event_id) is not None
        }
        if EventType.CHAIN_STEP_REFUSED_BY_ETHICS in event_types:
            event = next(
                ledger.require(event_id)
                for event_id in execution.event_ids
                if ledger.require(event_id).event_type == EventType.CHAIN_STEP_REFUSED_BY_ETHICS
            )
            self._append_step(
                run_id,
                "EthicsKernel REFUSE",
                "refused",
                str(event.payload.get("reason") or event.payload.get("rule_id") or ""),
            )
        elif EventType.CHAIN_STEP_AWAITING_REVIEW in event_types:
            event = next(
                ledger.require(event_id)
                for event_id in execution.event_ids
                if ledger.require(event_id).event_type == EventType.CHAIN_STEP_AWAITING_REVIEW
            )
            self._append_step(
                run_id,
                "EthicsKernel REQUIRE_HUMAN_REVIEW",
                "review",
                str(event.payload.get("reason") or event.payload.get("rule_id") or ""),
            )
        else:
            self._append_step(run_id, "EthicsKernel ALLOW", "completed")

    def _ensure_session(self, session_id: str | None, prompt: str) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            title = _session_title(prompt)
            session = SessionState(session_id=f"sess-{uuid4()}", title=title)
            self._sessions[session.session_id] = session
            return session

    def _require_run(self, run_id: str) -> RunState:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(run_id) from exc

    def _run_snapshot(self, run_id: str) -> RunState:
        with self._lock:
            return self._require_run(run_id)

    def _current_run_id(self) -> str:
        current = threading.current_thread()
        if current.name.startswith("run-"):
            return current.name
        with self._lock:
            if len(self._runs) == 1:
                return next(iter(self._runs))
        return ""

    def _mark_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        active_step: str | None = None,
    ) -> None:
        with self._lock:
            run = self._require_run(run_id)
            if status is not None:
                run.status = status
            if started_at is not None:
                run.started_at = started_at
            if completed_at is not None:
                run.completed_at = completed_at
            if active_step is not None:
                run.active_step = active_step
            run.updated_at = datetime.now(UTC)

    def _set_active(self, run_id: str, label: str) -> None:
        self._mark_run(run_id, active_step=label)

    def _set_payload(self, run_id: str, task_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.task_type = task_type
            run.proposed_payload = payload
            run.updated_at = datetime.now(UTC)

    def _set_tool_call_proposal(self, run_id: str, proposal: ToolCallProposal) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.proposed_payload["tool_call_proposal"] = tool_call_proposal_payload(proposal)
            run.proposed_payload["model_proposed_specialist_id"] = proposal.specialist_id
            run.proposed_payload["model_proposed_task_type"] = proposal.task_type
            run.updated_at = datetime.now(UTC)

    def _set_task_type(self, run_id: str, task_type: str) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.task_type = task_type
            run.proposed_payload["task_type"] = task_type
            if "specialist_request" in run.proposed_payload:
                run.proposed_payload["specialist_request"]["task_type"] = task_type
            run.updated_at = datetime.now(UTC)

    def _set_specialist_request_payload(self, run_id: str, request: SpecialistRequest) -> None:
        with self._lock:
            run = self._require_run(run_id)
            actual = {
                "task_type": request.task_type,
                "input_payload": dict(request.input_payload),
            }
            run.proposed_payload["actual_specialist_request"] = actual
            run.updated_at = datetime.now(UTC)

    def _set_selected_specialist(self, run_id: str, specialist_id: str | None) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.selected_specialist = specialist_id
            run.proposed_payload["model_proposed_specialist_id"] = specialist_id
            run.updated_at = datetime.now(UTC)

    def _set_validation(self, run_id: str, validation: dict[str, Any]) -> None:
        with self._lock:
            run = self._require_run(run_id)
            payload = dict(validation)
            payload["warnings"] = _dedupe_strings(
                [
                    *list(run.proposed_payload.get("warnings") or []),
                    *list(payload.get("warnings") or []),
                ]
            )
            run.updated_at = datetime.now(UTC)
            run.validation_result = payload

    def _append_step(
        self, run_id: str, label: str, status: str, detail: str = "", duration_ms: int | None = None
    ) -> None:
        if not run_id:
            return
        with self._lock:
            run = self._require_run(run_id)
            run.steps.append(
                UIStep(
                    step_id=f"step-{len(run.steps) + 1}",
                    label=label,
                    status=status,
                    detail=detail,
                    duration_ms=duration_ms,
                )
            )
            if status == "running":
                run.active_step = label
            run.updated_at = datetime.now(UTC)

    def _append_model_call(self, run_id: str, call: UIModelCall) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.model_calls.append(call)
            run.updated_at = datetime.now(UTC)

    def _append_event(self, run_id: str, event: ProvenanceEvent) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.provenance_events.append(event.to_dict())
            run.updated_at = datetime.now(UTC)

    def _append_error(self, run_id: str, error: str) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.errors.append(error)
            run.updated_at = datetime.now(UTC)

    def _finish(self, run_id: str, *, answer: str, status: str) -> None:
        with self._lock:
            run = self._require_run(run_id)
            run.final_answer = answer
            run.status = status
            run.completed_at = datetime.now(UTC)
            run.active_step = ""
            run.steps.append(
                UIStep(
                    step_id=f"step-{len(run.steps) + 1}",
                    label="Run completed" if status == "completed" else "Run failed",
                    status="completed" if status == "completed" else status,
                )
            )
            run.updated_at = datetime.now(UTC)


@dataclass(frozen=True)
class RequestPayload:
    task_type: str
    payload: dict[str, Any]
    notes: list[str]
    warnings: list[str] = field(default_factory=list)
    source: str = "heuristic_prompt"

    def to_payload(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "input_payload": self.payload,
            "payload": self.payload,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "source": self.source,
            "specialist_request": {
                "task_type": self.task_type,
                "input_payload": self.payload,
            },
        }


class RoutingParseWithResponseError(RoutingParseError):
    def __init__(self, message: str, response: ModelDriverResult) -> None:
        super().__init__(message)
        self.response = response


def route_with_raw(
    driver: ModelDriver,
    request: SpecialistRequest,
    cards: Sequence[SpecialistCard],
    prompt: str,
) -> ModelRouteResult:
    if isinstance(driver, TextGenerationDriver):
        response = driver.synthesize(prompt)
        try:
            decision = parse_routing_response(response.raw_text)
        except RoutingParseError as exc:
            raise RoutingParseWithResponseError(str(exc), response) from exc
        return ModelRouteResult(decision=decision, response=response)
    return driver.route(request, cards)


def routing_decision_payload(decision: RoutingDecision) -> dict[str, Any]:
    return {
        "specialist_id": decision.specialist_id,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "raw_response": decision.raw_response,
    }


def tool_call_proposal_payload(proposal: ToolCallProposal) -> dict[str, Any]:
    return {
        "specialist_id": proposal.specialist_id,
        "task_type": proposal.task_type,
        "input_payload": dict(proposal.input_payload),
        "confidence": proposal.confidence,
        "reasoning": proposal.reasoning,
        "raw_response": proposal.raw_response,
    }


def _proposal_from_explicit_payload(
    request_payload: RequestPayload, cards: Sequence[SpecialistCard]
) -> ToolCallProposal:
    specialist_id = _specialist_id_for_task_type(request_payload.task_type, cards)
    payload = {
        "specialist_id": specialist_id,
        "task_type": request_payload.task_type,
        "input_payload": dict(request_payload.payload),
        "confidence": None,
        "reasoning": "Explicit payload editor JSON.",
    }
    raw_response = json.dumps(payload, sort_keys=True, indent=2, default=str)
    return ToolCallProposal(
        specialist_id=specialist_id,
        task_type=request_payload.task_type,
        input_payload=dict(request_payload.payload),
        confidence=None,
        reasoning="Explicit payload editor JSON.",
        raw_response=raw_response,
    )


def _manual_tool_call_response(proposal: ToolCallProposal) -> ModelDriverResult:
    return ModelDriverResult(
        provider="vecl-ui",
        model_id="payload-editor",
        raw_text=proposal.raw_response,
        metadata={"source": "explicit_payload_editor"},
    )


def _specialist_id_for_task_type(task_type: str, cards: Sequence[SpecialistCard]) -> str | None:
    matches = sorted(card.specialist_id for card in cards if task_type in card.supported_task_types)
    return matches[0] if len(matches) == 1 else None


def _tool_call_validation_config(request: SpecialistRequest) -> ToolCallValidationConfig:
    return ToolCallValidationConfig(
        tenant_id=request.tenant_id,
        request_id=request.request_id,
        provenance_context=dict(request.provenance_context),
        terraform_config_dir=_terraform_config_dir(),
        max_stockfish_depth=MAX_UI_STOCKFISH_DEPTH,
    )


def _terraform_config_dir() -> str | None:
    value = os.environ.get("VECL_UI_TERRAFORM_CONFIG_DIR", "").strip()
    return value or None


def _running_specialist_label(specialist_id: str, payload: dict[str, Any]) -> str:
    if specialist_id == "terraform":
        operation = str(payload.get("operation") or payload.get("command") or "plan")
        return f"Running Terraform {operation}"
    if specialist_id == "stockfish":
        return f"Running Stockfish depth {payload.get('depth', 4)}"
    return f"Running {specialist_id}"


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def build_request_payload(prompt: str, override: dict[str, Any]) -> RequestPayload:
    if override:
        return _explicit_request_payload(prompt, override)
    payload = {"query": prompt}
    notes: list[str] = []
    warnings: list[str] = []
    task_type = _infer_task_type(prompt, payload, notes)
    _infer_payload_fields(prompt, task_type, payload, notes)
    _validate_payload_fields(task_type, payload, notes, warnings)
    return RequestPayload(
        task_type=task_type,
        payload=payload,
        notes=notes,
        warnings=warnings,
        source="heuristic_prompt",
    )


def _explicit_request_payload(prompt: str, override: dict[str, Any]) -> RequestPayload:
    notes = ["using explicit payload editor JSON as authoritative"]
    warnings = _explicit_payload_disagreement_warnings(prompt, override)
    task_type, input_payload = _payload_envelope(override)
    _validate_payload_fields(task_type, input_payload, notes, warnings)
    return RequestPayload(
        task_type=task_type,
        payload=input_payload,
        notes=notes,
        warnings=warnings,
        source="explicit_payload_editor",
    )


def _payload_envelope(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    task_type = str(payload.get("task_type") or "").strip()
    if "input_payload" in payload:
        raw_input = payload["input_payload"]
        if not task_type:
            raise ValueError("payload editor JSON with input_payload must include task_type")
        if not isinstance(raw_input, dict):
            raise ValueError("payload editor input_payload must be a JSON object")
        return task_type, dict(raw_input)
    if not task_type:
        raise ValueError(
            "payload editor JSON must include task_type with input_payload, "
            "or be wrapped by a selected specialist preset"
        )
    input_payload = {key: value for key, value in payload.items() if key != "task_type"}
    if not input_payload:
        raise ValueError("payload editor input payload must be non-empty")
    return task_type, input_payload


def default_specialist_specs() -> list[SpecialistSpec]:
    return [
        SpecialistSpec(
            SpecialistCard(
                "stockfish",
                {"chess_eval"},
                {"trust_anchor_id": "stockfish-v18", "description": "Stockfish UCI analysis."},
                cost_hint=1.0,
                latency_hint=1.0,
                version="stockfish-18",
                effective_trust=0.9,
                description="Analyze chess positions with Stockfish.",
            ),
            lambda store: StockfishSpecialist(artifact_store=store),
            _stockfish_availability,
            "txt",
            "Runs a local UCI engine and records the transcript.",
        ),
        SpecialistSpec(
            SpecialistCard(
                "sympy",
                {"symbolic_math", "sequence_math"},
                {"trust_anchor_id": "sympy-v1.14", "description": "Symbolic algebra."},
                cost_hint=0.05,
                latency_hint=0.2,
                version="sympy-1.14",
                effective_trust=0.95,
                description="Solve, simplify, factor, integrate, differentiate, and plot.",
            ),
            lambda store: SymPySpecialist(artifact_store=store),
            _sympy_availability,
            "tex",
            "Produces LaTeX and an optional PNG plot.",
        ),
        SpecialistSpec(
            SpecialistCard(
                "blast",
                {"sequence_alignment"},
                {"trust_anchor_id": "blast-v2.17", "description": "NCBI BLAST+ search."},
                cost_hint=0.7,
                latency_hint=1.5,
                version="blast-2.17",
                effective_trust=0.9,
                description="Run local BLAST nucleotide alignments against a local database.",
            ),
            lambda store: BLASTSpecialist(artifact_store=store),
            _blast_availability,
            "tsv",
            "Requires local blastn and a caller-supplied database.",
        ),
        SpecialistSpec(
            SpecialistCard(
                "terraform",
                {"infrastructure_plan"},
                {
                    "trust_anchor_id": "terraform-cli",
                    "description": "Terraform validate and plan only.",
                },
                cost_hint=0.5,
                latency_hint=0.8,
                version="terraform-plan-only",
                effective_trust=0.95,
                description="Validate or plan a local Terraform module; mutation is disabled.",
            ),
            lambda store: TerraformSpecialist(artifact_store=store),
            _terraform_availability,
            "json",
            "Apply, destroy, state mutation, and workspace mutation are refused.",
        ),
        SpecialistSpec(
            SpecialistCard(
                "timesfm",
                {"demand_forecast", "tool_request"},
                {"trust_anchor_id": "timesfm-v2.5", "description": "TimesFM forecasting."},
                cost_hint=2.0,
                latency_hint=3.0,
                version="timesfm-2.5-200m-pytorch",
                effective_trust=0.9,
                description="Forecast numeric time series with a local TimesFM runner.",
            ),
            lambda store: TimesFMSpecialist(artifact_store=store),
            _timesfm_availability,
            "json",
            "Local execution is opt-in to avoid accidental model downloads.",
        ),
    ]


def _default_driver_factory(provider: str) -> ModelDriver:
    selected = provider.strip().lower()
    if selected == "gemma" and not _local_gemma_enabled():
        raise ValueError(
            "Gemma interactive local execution is disabled. Use Vertex/batch eval scripts or "
            "set VECL_UI_ENABLE_LOCAL_GEMMA=1 with local substrate dependencies installed."
        )
    return resolve_model_driver(selected)


def _driver_status_openai() -> dict[str, Any]:
    load_dotenv_if_present()
    model_id = os.environ.get("VECL_OPENAI_MODEL", "").strip()
    has_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    available = bool(model_id and has_key)
    return {
        "provider": "openai",
        "label": "OpenAI",
        "model_id": model_id or "(set VECL_OPENAI_MODEL)",
        "available": available,
        "enabled": available,
        "detail": "Ready from .env" if available else "Set VECL_OPENAI_MODEL and OPENAI_API_KEY",
        "env": {"VECL_OPENAI_MODEL": bool(model_id), "OPENAI_API_KEY": has_key},
    }


def _driver_status_anthropic() -> dict[str, Any]:
    load_dotenv_if_present()
    model_id = os.environ.get("VECL_ANTHROPIC_MODEL", "").strip()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    available = bool(model_id and has_key)
    return {
        "provider": "anthropic",
        "label": "Anthropic",
        "model_id": model_id or "(set VECL_ANTHROPIC_MODEL)",
        "available": available,
        "enabled": available,
        "detail": (
            "Ready from .env" if available else "Set VECL_ANTHROPIC_MODEL and ANTHROPIC_API_KEY"
        ),
        "env": {"VECL_ANTHROPIC_MODEL": bool(model_id), "ANTHROPIC_API_KEY": has_key},
    }


def _driver_status_gemma() -> dict[str, Any]:
    load_dotenv_if_present()
    model_id = os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID)
    deps = _module_exists("transformers") and _module_exists("torch")
    has_token = bool(os.environ.get("HF_TOKEN", "").strip())
    enabled = _local_gemma_enabled()
    available = bool(enabled and deps and has_token)
    if available:
        detail = "Local Gemma enabled for interactive routing/synthesis."
    elif enabled:
        detail = "Local Gemma requested, but torch/transformers or HF_TOKEN are missing."
    else:
        detail = "Vertex/batch only by default; local UI execution is disabled."
    return {
        "provider": "gemma",
        "label": "Gemma",
        "model_id": model_id,
        "available": available,
        "enabled": available,
        "detail": detail,
        "env": {
            "VECL_UI_ENABLE_LOCAL_GEMMA": enabled,
            "HF_TOKEN": has_token,
            "torch_transformers": deps,
        },
    }


def _stockfish_availability() -> Availability:
    try:
        path = resolve_stockfish_binary()
    except Exception as exc:
        return Availability(False, "missing", str(exc), can_execute=False)
    return Availability(True, "available", f"binary: {path}")


def _sympy_availability() -> Availability:
    if _module_exists("sympy"):
        return Availability(True, "available", "sympy import is available")
    return Availability(False, "missing", "sympy is not importable", can_execute=False)


def _terraform_availability() -> Availability:
    try:
        path = resolve_terraform_binary()
    except Exception as exc:
        return Availability(False, "missing", str(exc), can_execute=False)
    return Availability(True, "available", f"binary: {path}")


def _blast_availability() -> Availability:
    try:
        path = resolve_blast_binary("blastn")
    except Exception as exc:
        return Availability(False, "missing", str(exc), can_execute=False)
    return Availability(True, "available", f"binary: {path}")


def _timesfm_availability() -> Availability:
    enabled = os.environ.get("VECL_UI_ENABLE_LOCAL_TIMESFM", "").strip() == "1"
    if not enabled:
        return Availability(
            False,
            "disabled",
            "Local TimesFM execution is disabled by default to avoid model downloads.",
            can_execute=False,
        )
    missing = [name for name in ("timesfm", "torch", "numpy") if not _module_exists(name)]
    if missing:
        return Availability(
            False,
            "missing",
            f"missing optional dependencies: {', '.join(missing)}",
            can_execute=False,
        )
    return Availability(True, "available", "TimesFM dependencies are importable")


def _ethics_kernel_for_run(
    ledger: ProvenanceLedger,
) -> tuple[EthicsKernel, list[ProvenanceEvent]]:
    env_name = "VECL_UI_ETHICS_ADMIN_TOKEN"
    existing = os.environ.get(env_name)
    token = existing or secrets.token_urlsafe(24)
    os.environ[env_name] = token
    kernel = EthicsKernel(ledger=ledger, admin_token_env=env_name)
    events: list[ProvenanceEvent] = []
    try:
        for rule in default_ethics_rules():
            events.append(kernel.install_rule(rule, token))
    finally:
        if existing is None:
            os.environ.pop(env_name, None)
    return kernel, events


def _infer_task_type(prompt: str, payload: dict[str, Any], notes: list[str]) -> str:
    lowered = prompt.lower()
    if payload.get("fen") or _extract_fen(prompt) or "stockfish" in lowered or "chess" in lowered:
        notes.append("inferred chess_eval from chess/FEN language")
        return "chess_eval"
    if "terraform" in lowered or any(
        word in lowered for word in ("tfstate", "apply", "destroy", "plan-only")
    ):
        notes.append("inferred infrastructure_plan from Terraform language")
        return "infrastructure_plan"
    if "blast" in lowered or "fasta" in lowered or payload.get("query_sequence"):
        notes.append("inferred sequence_alignment from BLAST/FASTA language")
        return "sequence_alignment"
    if "timesfm" in lowered or "forecast" in lowered or "demand" in lowered:
        notes.append("inferred demand_forecast from forecast language")
        return "demand_forecast"
    if any(
        word in lowered
        for word in ("sympy", "simplify", "factor", "differentiate", "integrate", "solve", "plot")
    ):
        notes.append("inferred symbolic_math from math operation language")
        return "symbolic_math"
    if "expression" in payload or "operation" in payload:
        notes.append("inferred symbolic_math from payload fields")
        return "symbolic_math"
    notes.append("no specialist-specific task type inferred")
    return "general"


def _infer_payload_fields(
    prompt: str, task_type: str, payload: dict[str, Any], notes: list[str]
) -> None:
    lowered = prompt.lower()
    if task_type in {
        "chess_eval",
        "infrastructure_plan",
        "symbolic_math",
        "sequence_alignment",
        "demand_forecast",
    } and _is_configuration_prompt(lowered):
        payload.setdefault("operation", "configure")
        payload.setdefault("configuration_template", True)
        notes.append("marked request as specialist configuration/template")
        return
    if task_type == "chess_eval":
        if "fen" not in payload:
            fen = _extract_fen(prompt)
            if fen:
                payload["fen"] = fen
                notes.append("extracted FEN from prompt")
            elif "starting position" in lowered:
                payload["fen"] = STARTING_FEN
                notes.append("used standard chess starting position FEN")
        if "depth" not in payload:
            depth = _extract_depth(prompt)
            if depth is not None:
                payload["depth"] = depth
                notes.append("extracted Stockfish depth from prompt")
        payload.setdefault("depth", 4)
    elif task_type == "infrastructure_plan":
        payload.setdefault("operation", _terraform_operation_from_text(lowered))
        operation = str(payload.get("operation") or "").lower()
        if operation in {"apply", "destroy", "state", "import", "workspace", "taint", "untaint"}:
            payload.setdefault("destructive", True)
            tags = set(str(tag) for tag in payload.get("risk_tags", []))
            tags.add("terraform_mutation")
            payload["risk_tags"] = sorted(tags)
            notes.append("marked Terraform mutation for ethics refusal")
        if "config_dir" not in payload:
            config_dir = _extract_assignment(prompt, "config_dir") or _extract_assignment(
                prompt, "path"
            )
            if config_dir:
                payload["config_dir"] = config_dir
                notes.append("extracted config_dir from prompt")
    elif task_type == "symbolic_math":
        payload.setdefault("operation", _sympy_operation_from_text(lowered))
        if "expression" not in payload:
            expression = _extract_expression(prompt)
            if expression:
                payload["expression"] = expression
                notes.append("extracted symbolic expression from prompt")
    elif task_type == "sequence_alignment":
        if "query_sequence" not in payload and "query_fasta" not in payload:
            sequence = _extract_sequence(prompt)
            if sequence:
                payload["query_sequence"] = sequence
                notes.append("extracted candidate sequence from prompt")
    elif task_type == "demand_forecast":
        if "values" not in payload and "history" not in payload and "series" not in payload:
            values = [float(value) for value in _NUMBER_RE.findall(prompt)]
            if len(values) >= 8:
                payload["values"] = values
                notes.append("extracted numeric history from prompt")
        if "horizon" not in payload:
            horizon = _extract_horizon(prompt)
            if horizon is not None:
                payload["horizon"] = horizon
                notes.append("extracted forecast horizon")


def _validate_payload_fields(
    task_type: str,
    payload: dict[str, Any],
    notes: list[str],
    warnings: list[str],
) -> None:
    operation = str(payload.get("operation") or "").strip().lower()
    if operation in CONFIG_OPERATIONS or payload.get("configuration_template"):
        return
    if task_type == "chess_eval":
        if "fen" in payload:
            fen = str(payload["fen"]).strip()
            if not _is_valid_fen(fen):
                raise ValueError(f"invalid FEN in specialist payload: {fen}")
            payload["fen"] = fen
        if "depth" in payload:
            depth = _positive_int(payload["depth"], "depth")
            if depth > MAX_UI_STOCKFISH_DEPTH:
                warnings.append(f"Stockfish depth clamped from {depth} to {MAX_UI_STOCKFISH_DEPTH}")
                depth = MAX_UI_STOCKFISH_DEPTH
            payload["depth"] = depth
    elif task_type == "infrastructure_plan":
        operation = str(payload.get("operation") or payload.get("command") or "").lower()
        if operation in {"apply", "destroy", "state", "import", "workspace", "taint", "untaint"}:
            notes.append("Terraform mutation will be refused by governance")


def _explicit_payload_disagreement_warnings(prompt: str, payload: dict[str, Any]) -> list[str]:
    task_type, input_payload = _payload_envelope(payload)
    notes: list[str] = []
    prompt_task = _infer_task_type(prompt, {"query": prompt}, notes)
    warnings: list[str] = []
    if prompt_task != "general" and prompt_task != task_type:
        warnings.append(
            f"natural language appears to request {prompt_task}, but explicit payload uses {task_type}; using explicit payload"
        )
    prompt_fen = _extract_fen(prompt)
    payload_fen = input_payload.get("fen")
    if prompt_fen and payload_fen and prompt_fen != str(payload_fen).strip():
        warnings.append("natural language FEN differs from payload FEN; using explicit payload")
    prompt_depth = _extract_depth(prompt)
    payload_depth = input_payload.get("depth")
    if prompt_depth is not None and payload_depth is not None:
        try:
            parsed_payload_depth = int(payload_depth)
        except (TypeError, ValueError):
            parsed_payload_depth = None
        if parsed_payload_depth is not None and prompt_depth != parsed_payload_depth:
            warnings.append(
                f"natural language depth {prompt_depth} differs from payload depth {parsed_payload_depth}; using explicit payload"
            )
    return warnings


def _terraform_operation_from_text(lowered: str) -> str:
    for operation in (
        "destroy",
        "apply",
        "validate",
        "state",
        "import",
        "workspace",
        "taint",
        "untaint",
        "plan",
    ):
        if operation in lowered:
            return operation
    return "plan"


def _is_configuration_prompt(lowered: str) -> bool:
    return any(
        phrase in lowered
        for phrase in (
            "configuration",
            "configure",
            "config",
            "payload template",
            "schema",
            "input fields",
            "how to call",
            "tool contract",
        )
    )


def _sympy_operation_from_text(lowered: str) -> str:
    if "differentiate" in lowered or "derivative" in lowered:
        return "differentiate"
    if "integrate" in lowered or "integral" in lowered:
        return "integrate"
    if "factor" in lowered:
        return "factor"
    if "solve" in lowered:
        return "solve"
    if "plot" in lowered:
        return "plot"
    return "simplify"


def _extract_fen(prompt: str) -> str | None:
    for segment in re.split(r"[\"`,\n]", prompt):
        tokens = segment.split()
        for index in range(0, max(0, len(tokens) - 5)):
            candidate = " ".join(tokens[index : index + 6]).strip()
            if _is_valid_fen(candidate):
                return candidate
    return None


def _is_valid_fen(fen: str) -> bool:
    parts = fen.split()
    if len(parts) != 6:
        return False
    board, side, castling, en_passant, halfmove, fullmove = parts
    ranks = board.split("/")
    if len(ranks) != 8:
        return False
    for rank in ranks:
        total = 0
        previous_digit = False
        for char in rank:
            if char.isdigit():
                if char == "0" or previous_digit:
                    return False
                total += int(char)
                previous_digit = True
            elif char in "pnbrqkPNBRQK":
                total += 1
                previous_digit = False
            else:
                return False
        if total != 8:
            return False
    if side not in {"w", "b"}:
        return False
    if castling != "-" and (
        not re.fullmatch(r"[KQkq]+", castling) or len(set(castling)) != len(castling)
    ):
        return False
    if en_passant != "-" and not re.fullmatch(r"[a-h][36]", en_passant):
        return False
    return halfmove.isdigit() and fullmove.isdigit() and int(fullmove) > 0


def _extract_depth(prompt: str) -> int | None:
    match = re.search(r"\bdepth\s*(?:=|:)?\s*(\d+)\b", prompt, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _extract_assignment(prompt: str, key: str) -> str | None:
    match = re.search(rf"{re.escape(key)}\s*=\s*([^\s,]+)", prompt)
    if match:
        return match.group(1).strip("'\"")
    return None


def _extract_expression(prompt: str) -> str | None:
    fenced = re.findall(r"`([^`]+)`", prompt)
    if fenced:
        return str(fenced[0]).strip()
    match = re.search(r"(?:expression|expr)\s*[:=]\s*(.+)$", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    for word in ("simplify", "factor", "differentiate", "integrate", "solve", "plot"):
        match = re.search(rf"{word}\s+(.+)$", prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_sequence(prompt: str) -> str | None:
    candidates = re.findall(r"\b[ACGTUNacgtun]{12,}\b", prompt)
    if not candidates:
        return None
    return str(max(candidates, key=len)).upper().replace("U", "T")


def _extract_horizon(prompt: str) -> int | None:
    match = re.search(r"horizon\s*[:=]?\s*(\d+)", prompt, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _response_payload(response: SpecialistResponse) -> dict[str, Any]:
    return {
        "request_id": response.request_id,
        "specialist_id": response.specialist_id,
        "tenant_id": response.tenant_id,
        "refusal_or_error": response.refusal_or_error,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "specialist_id": claim.specialist_id,
                "claim_text": claim.claim_text,
                "claim_type": claim.claim_type,
                "confidence": claim.confidence,
                "evidence_ids": claim.evidence_ids,
                "source_ids": claim.source_ids,
                "artifact_ids": claim.artifact_ids,
                "assumptions": claim.assumptions,
                "limitations": claim.limitations,
                "high_risk": claim.high_risk,
            }
            for claim in response.claims
        ],
        "cost_metadata": response.cost_metadata,
    }


def _parent_for_final(request: SpecialistRequest, execution: ChainExecutionResult | None) -> str:
    if execution is None:
        return str(request.provenance_context["parent_event_id"])
    return execution.event_ids[-1] if execution.event_ids else execution.chain_event_id


def _session_title(prompt: str) -> str:
    title = " ".join(prompt.split())
    if len(title) > 48:
        return title[:45].rstrip() + "..."
    return title or "New conversation"


def _local_gemma_enabled() -> bool:
    return os.environ.get("VECL_UI_ENABLE_LOCAL_GEMMA", "").strip() == "1"


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(stripped)
        if not match:
            raise
        loaded = json.loads(match.group(0))
    if not isinstance(loaded, dict):
        raise ValueError("JSON payload must be an object")
    return loaded
