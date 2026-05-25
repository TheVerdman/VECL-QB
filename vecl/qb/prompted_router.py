from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb._llm_inference import DEFAULT_ROUTING_MODEL_ID, gemma_route_once
from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistRequest
from vecl.qb.specialist_contracts import contract_for_specialist

ROUTING_SYSTEM_PROMPT_TEMPLATE = """You are VECL-QB's specialist router.
Choose exactly one registered specialist when a request clearly needs that tool.
Return only JSON with this schema:
{{"tool": "<specialist_id or null>", "confidence": <0.0-1.0>, "reasoning": "<brief reason>"}}

Registered specialists:
{specialists}

Request:
task_type: {task_type}
query: {query}
payload: {payload}
"""

_FENCED_RE = re.compile(r"```(?:json|python|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FUNCTION_RE = re.compile(r"(?P<name>[A-Za-z_][\w.]*)\s*\((?P<args>.*)\)\s*$", re.DOTALL)
_KEY_VALUE_RE = re.compile(r"(\w+)\s*=")
NO_TOOL_VALUES = {"", "none", "null", "no_tool", "no-specialist", "no specialist", "n/a"}


@dataclass(frozen=True)
class RoutingDecision:
    specialist_id: str | None
    confidence: float | None
    reasoning: str
    raw_response: str


class RoutingParseError(ValueError):
    pass


class PromptedLLMRouter:
    def __init__(
        self,
        *,
        ledger: ProvenanceLedger | None = None,
        inference_fn: Callable[[str], str] = gemma_route_once,
        model_id: str = DEFAULT_ROUTING_MODEL_ID,
        fallback_router: QBRouter | None = None,
        max_specialists: int | None = None,
    ) -> None:
        self.ledger = ledger
        self.inference_fn = inference_fn
        self.model_id = model_id
        self.max_specialists = max_specialists
        self._fallback_router = fallback_router or QBRouter(max_specialists=max_specialists)
        self._registry: dict[str, tuple[SpecialistCard, Specialist]] = {}

    def register_specialist(self, card: SpecialistCard, specialist: Specialist) -> None:
        self._registry[card.specialist_id] = (card, specialist)
        self._fallback_router.register_specialist(card, specialist)

    @property
    def specialists(self) -> dict[str, Specialist]:
        return {
            specialist_id: specialist
            for specialist_id, (_card, specialist) in self._registry.items()
        }

    @property
    def cards(self) -> dict[str, SpecialistCard]:
        return {
            specialist_id: card for specialist_id, (card, _specialist) in self._registry.items()
        }

    def route(
        self, request: SpecialistRequest, max_specialists: int | None = None
    ) -> list[Specialist]:
        prompt = build_routing_prompt(list(self._registry.values()), request)
        raw_response = self.inference_fn(prompt)
        try:
            decision = parse_routing_response(raw_response)
            specialists = self._specialists_for_decision(decision, max_specialists)
            self._append_decision_event(request, decision)
            return specialists
        except RoutingParseError as exc:
            fallback = self._fallback_router.route(request, max_specialists=max_specialists)
            self._append_fallback_event(request, str(exc), fallback)
            return fallback

    def _specialists_for_decision(
        self, decision: RoutingDecision, max_specialists: int | None
    ) -> list[Specialist]:
        if decision.specialist_id is None:
            return []
        registered = self._registry.get(decision.specialist_id)
        if registered is None:
            raise RoutingParseError(f"unknown specialist_id: {decision.specialist_id}")
        limit = max_specialists if max_specialists is not None else self.max_specialists
        specialists = [registered[1]]
        return specialists if limit is None else specialists[:limit]

    def _append_decision_event(self, request: SpecialistRequest, decision: RoutingDecision) -> None:
        if self.ledger is None:
            return
        self.ledger.append(
            ProvenanceEvent(
                EventType.LLM_ROUTING_DECIDED,
                request.tenant_id,
                "prompted-llm-router",
                {
                    "request_id": request.request_id,
                    "query_hash": _request_query_hash(request),
                    "chosen_specialist_id": decision.specialist_id,
                    "model_id": self.model_id,
                    "reasoning": decision.reasoning,
                    "confidence": decision.confidence,
                },
                parent_event_ids=_parent_event_ids(request),
            )
        )

    def _append_fallback_event(
        self, request: SpecialistRequest, reason: str, fallback: list[Specialist]
    ) -> None:
        if self.ledger is None:
            return
        self.ledger.append(
            ProvenanceEvent(
                EventType.ROUTING_FALLBACK,
                request.tenant_id,
                "prompted-llm-router",
                {
                    "request_id": request.request_id,
                    "query_hash": _request_query_hash(request),
                    "parse_failure_reason": reason,
                    "fallback_specialist_id": fallback[0].specialist_id if fallback else None,
                    "model_id": self.model_id,
                },
                parent_event_ids=_parent_event_ids(request),
            )
        )


def build_routing_prompt(
    entries: list[tuple[SpecialistCard, Specialist]], request: SpecialistRequest
) -> str:
    return build_routing_prompt_from_cards([card for card, _specialist in entries], request)


def build_routing_prompt_from_cards(cards: list[SpecialistCard], request: SpecialistRequest) -> str:
    serialized = "\n".join(serialize_specialist_card(card) for card in cards)
    payload = json.dumps(request.input_payload, sort_keys=True, default=str)
    query = str(request.input_payload.get("query") or request.input_payload.get("prompt") or "")
    return ROUTING_SYSTEM_PROMPT_TEMPLATE.format(
        specialists=serialized or "[]",
        task_type=request.task_type,
        query=query,
        payload=payload,
    )


def serialize_specialist_card(card: SpecialistCard) -> str:
    contract = contract_for_specialist(card.specialist_id)
    chunk = {
        "id": card.specialist_id,
        "supported_task_types": sorted(card.supported_task_types),
        "description": card.description
        or str(card.trust_requirements.get("description", "")).strip()
        or f"Specialist for {', '.join(sorted(card.supported_task_types))}",
        "cost_hint": card.cost_hint,
        "latency_hint": card.latency_hint,
        "version": card.version,
        "effective_trust": card.effective_trust,
    }
    if contract is not None:
        chunk["payload_contract"] = contract
    return json.dumps(chunk, sort_keys=True, separators=(",", ":"))


def parse_routing_response(response: str) -> RoutingDecision:
    candidates = _response_candidates(response)
    for candidate in candidates:
        try:
            payload = _parse_candidate(candidate)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if payload is None:
            continue
        return _decision_from_payload(payload, response)
    raise RoutingParseError("could not parse LLM routing response")


def _response_candidates(response: str) -> list[str]:
    stripped = response.strip()
    fenced = [match.group(1).strip() for match in _FENCED_RE.finditer(stripped)]
    return [*fenced, stripped]


def _parse_candidate(candidate: str) -> dict[str, Any] | None:
    if not candidate:
        return None
    try:
        loaded = json.loads(candidate)
    except json.JSONDecodeError:
        loaded = _parse_function_call(candidate)
    if not isinstance(loaded, dict):
        return None
    return loaded


def _parse_function_call(candidate: str) -> dict[str, Any]:
    match = _FUNCTION_RE.match(candidate.strip())
    if not match:
        raise RoutingParseError("not a function-call response")
    args = match.group("args").strip()
    if args.startswith("{"):
        loaded = json.loads(args)
        if not isinstance(loaded, dict):
            raise RoutingParseError("function-call payload must be an object")
        return loaded
    pythonish = _KEY_VALUE_RE.sub(r'"\1":', args)
    loaded = ast.literal_eval("{" + pythonish + "}")
    if not isinstance(loaded, dict):
        raise RoutingParseError("function-call payload must be a mapping")
    return loaded


def _decision_from_payload(payload: dict[str, Any], raw_response: str) -> RoutingDecision:
    tool = payload.get("tool", payload.get("specialist_id", payload.get("name")))
    if isinstance(tool, dict):
        tool = tool.get("name") or tool.get("specialist_id") or tool.get("tool")
    if tool is None:
        specialist_id = None
    else:
        specialist_id = str(tool).strip()
        if specialist_id.lower() in NO_TOOL_VALUES:
            specialist_id = None
    confidence = payload.get("confidence")
    parsed_confidence = float(confidence) if confidence is not None else None
    reasoning = str(payload.get("reasoning") or payload.get("reason") or "")
    return RoutingDecision(
        specialist_id=specialist_id,
        confidence=parsed_confidence,
        reasoning=reasoning,
        raw_response=raw_response,
    )


def _request_query_hash(request: SpecialistRequest) -> str:
    return stable_hash(
        {
            "request_id": request.request_id,
            "task_type": request.task_type,
            "input_payload": request.input_payload,
        }
    )


def _parent_event_ids(request: SpecialistRequest) -> list[str]:
    parent_event_id = request.provenance_context.get("parent_event_id")
    return [str(parent_event_id)] if parent_event_id else []
