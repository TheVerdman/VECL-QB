from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.chain_executor import ChainExecutionResult, ChainExecutor
from vecl.qb.model_driver import ModelDriver, ModelDriverResult
from vecl.qb.planner import ChainPlan, ChainStep
from vecl.qb.prompted_router import NO_TOOL_VALUES, serialize_specialist_card
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistRequest
from vecl.qb.specialist_contracts import (
    is_config_request,
    payload_contracts_for_specialists,
)
from vecl.specialists.openroad_specialist import OPENROAD_ALLOWED_OPERATIONS
from vecl.specialists.sympy_specialist import SUPPORTED_OPERATIONS as SYMPY_OPERATIONS
from vecl.specialists.terraform_specialist import TERRAFORM_ALLOWED_OPERATIONS
from vecl.specialists.yosys_specialist import YOSYS_ALLOWED_OPERATIONS

TOOL_CALL_PROMPT_TEMPLATE = """You are VECL-QB's tool-call author.
Choose at most one registered specialist and produce the exact JSON payload VECL should validate.
Do not execute tools. Do not invent specialist ids or unsupported fields.
Return only JSON with this schema:
{{"specialist_id":"<specialist id or null>","task_type":"<task type or null>","input_payload":{{...}},"confidence":0.0,"reasoning":"<brief reason>"}}

Registered specialists:
{specialists}

Specialist input payload contracts:
{payload_contracts}

If the user asks for a specialist configuration, schema, payload template, or how to call
a specialist, select that specialist and set input_payload.operation to "configure".
Configuration requests do not require normal execution fields such as FEN, sequence,
database, time-series values, expression, or Terraform config_dir.

Request:
task_type_hint: {task_type}
query: {query}
payload_hint: {payload}
"""

_FENCED_RE = re.compile(r"```(?:json|python|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_FUNCTION_RE = re.compile(r"(?P<name>[A-Za-z_][\w.-]*)\s*\((?P<args>.*)\)\s*$", re.DOTALL)
_KEY_VALUE_RE = re.compile(r"(\w+)\s*=")


@dataclass(frozen=True)
class ToolCallProposal:
    specialist_id: str | None
    task_type: str | None
    input_payload: dict[str, Any]
    confidence: float | None
    reasoning: str
    raw_response: str


@dataclass(frozen=True)
class ValidatedToolCall:
    specialist_id: str
    request: SpecialistRequest
    plan: ChainPlan
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCallRunResult:
    proposal: ToolCallProposal
    model_response: ModelDriverResult
    validated: ValidatedToolCall | None
    execution: ChainExecutionResult | None
    proposal_event_id: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCallValidationConfig:
    tenant_id: str
    request_id: str
    provenance_context: dict[str, Any] = field(default_factory=dict)
    terraform_config_dir: str | Path | None = None
    max_stockfish_depth: int = 20


class ToolCallParseError(ValueError):
    pass


class ToolCallValidationError(ValueError):
    pass


def propose_tool_call(
    driver: ModelDriver,
    request: SpecialistRequest,
    specialist_cards: Sequence[SpecialistCard],
) -> tuple[ToolCallProposal, ModelDriverResult]:
    prompt = build_tool_call_prompt(request, specialist_cards)
    response = driver.synthesize(prompt)
    return parse_tool_call_response(response.raw_text), response


def build_tool_call_prompt(
    request: SpecialistRequest, specialist_cards: Sequence[SpecialistCard]
) -> str:
    serialized = "\n".join(serialize_specialist_card(card) for card in specialist_cards)
    payload = json.dumps(request.input_payload, sort_keys=True, default=str)
    query = str(request.input_payload.get("query") or request.input_payload.get("prompt") or "")
    return TOOL_CALL_PROMPT_TEMPLATE.format(
        specialists=serialized or "[]",
        payload_contracts=_payload_contracts(specialist_cards),
        task_type=request.task_type,
        query=query,
        payload=payload,
    )


def parse_tool_call_response(response: str) -> ToolCallProposal:
    for candidate in _response_candidates(response):
        try:
            payload = _load_object(candidate)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if payload is None:
            continue
        return _proposal_from_payload(payload, response)
    raise ToolCallParseError("could not parse LLM tool-call response")


def validate_tool_call(
    proposal: ToolCallProposal,
    *,
    cards: Mapping[str, SpecialistCard],
    config: ToolCallValidationConfig,
) -> ValidatedToolCall:
    if proposal.specialist_id is None:
        raise ToolCallValidationError("proposal did not select a specialist")
    card = cards.get(proposal.specialist_id)
    if card is None:
        raise ToolCallValidationError(f"unknown specialist_id: {proposal.specialist_id}")

    task_type = proposal.task_type or _default_task_type(card)
    if task_type not in card.supported_task_types:
        raise ToolCallValidationError(
            f"task_type {task_type!r} is not supported by {proposal.specialist_id}"
        )
    payload, warnings = _canonical_payload(proposal, task_type, config)
    request = SpecialistRequest(
        config.request_id,
        config.tenant_id,
        task_type,
        payload,
        {},
        dict(config.provenance_context),
    )
    plan = single_step_plan(proposal.specialist_id, task_type)
    return ValidatedToolCall(
        specialist_id=proposal.specialist_id,
        request=request,
        plan=plan,
        warnings=tuple(warnings),
    )


def execute_tool_call(
    *,
    proposal: ToolCallProposal,
    model_response: ModelDriverResult,
    cards: Mapping[str, SpecialistCard],
    specialists: Mapping[str, Specialist],
    ledger: ProvenanceLedger,
    config: ToolCallValidationConfig,
    executor: ChainExecutor | None = None,
) -> ToolCallRunResult:
    proposal_event = _append_proposal_event(ledger, proposal, model_response, config)
    try:
        validated = validate_tool_call(proposal, cards=cards, config=config)
    except ToolCallValidationError as exc:
        return ToolCallRunResult(
            proposal=proposal,
            model_response=model_response,
            validated=None,
            execution=None,
            proposal_event_id=proposal_event.event_id,
            warnings=(str(exc),),
        )
    runner = executor or ChainExecutor(ledger=ledger, specialists=dict(specialists))
    result = runner.execute(validated.plan, validated.request)
    return ToolCallRunResult(
        proposal=proposal,
        model_response=model_response,
        validated=validated,
        execution=result,
        proposal_event_id=proposal_event.event_id,
        warnings=validated.warnings,
    )


def single_step_plan(specialist_id: str, task_type: str) -> ChainPlan:
    expected_artifact_type = {
        "stockfish": "txt",
        "sympy": "tex",
        "blast": "tsv",
        "terraform": "json",
        "timesfm": "json",
        "yosys": "v",
        "openroad": "txt",
    }.get(specialist_id, "")
    return ChainPlan(
        f"single-step-{specialist_id}",
        task_type,
        (ChainStep("execute", specialist_id, expected_artifact_type=expected_artifact_type),),
    )


def _payload_contracts(specialist_cards: Sequence[SpecialistCard]) -> str:
    selected = payload_contracts_for_specialists(
        tuple(card.specialist_id for card in specialist_cards)
    )
    return json.dumps(selected, sort_keys=True, separators=(",", ":")) if selected else "{}"


def _canonical_payload(
    proposal: ToolCallProposal,
    task_type: str,
    config: ToolCallValidationConfig,
) -> tuple[dict[str, Any], list[str]]:
    payload = dict(proposal.input_payload)
    warnings: list[str] = []
    if proposal.specialist_id == "stockfish":
        return _canonical_stockfish_payload(payload, config, warnings), warnings
    if proposal.specialist_id == "sympy":
        return _canonical_sympy_payload(payload, warnings), warnings
    if proposal.specialist_id == "blast":
        return _canonical_blast_payload(payload, warnings), warnings
    if proposal.specialist_id == "terraform":
        return _canonical_terraform_payload(payload, config, warnings), warnings
    if proposal.specialist_id == "timesfm":
        return _canonical_timesfm_payload(payload, warnings), warnings
    if proposal.specialist_id == "yosys":
        return _canonical_yosys_payload(payload, warnings), warnings
    if proposal.specialist_id == "openroad":
        return _canonical_openroad_payload(payload, warnings), warnings
    return payload, warnings


def _canonical_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": str(payload.get("query") or ""),
        "operation": "configure",
        "configuration_template": True,
    }


def _canonical_stockfish_payload(
    payload: dict[str, Any], config: ToolCallValidationConfig, warnings: list[str]
) -> dict[str, Any]:
    if is_config_request(payload):
        return _canonical_config_payload(payload)
    fen = str(payload.get("fen") or "").strip().strip("`'\",")
    if not _valid_fen(fen):
        raise ToolCallValidationError("stockfish payload requires a valid six-field FEN")
    depth = _int_payload(payload, "depth", default=4)
    if depth < 1:
        raise ToolCallValidationError("stockfish depth must be positive")
    if depth > config.max_stockfish_depth:
        warnings.append(f"stockfish depth clamped from {depth} to {config.max_stockfish_depth}")
        depth = config.max_stockfish_depth
    result = {
        "query": str(payload.get("query") or ""),
        "fen": fen,
        "depth": depth,
    }
    if payload.get("movetime_ms") is not None:
        result["movetime_ms"] = _int_payload(payload, "movetime_ms", default=0)
    return result


def _canonical_sympy_payload(payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if is_config_request(payload):
        return _canonical_config_payload(payload)
    operation = str(payload.get("operation") or "simplify").strip().lower()
    if operation not in SYMPY_OPERATIONS:
        raise ToolCallValidationError(f"unsupported SymPy operation: {operation}")
    expression = str(payload.get("expression") or "").strip()
    if not expression:
        raise ToolCallValidationError("sympy payload requires expression")
    result = {
        "operation": operation,
        "expression": expression,
    }
    if payload.get("variable") is not None:
        result["variable"] = str(payload["variable"])
    return result


def _canonical_blast_payload(payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if is_config_request(payload):
        return _canonical_config_payload(payload)
    database = str(payload.get("database") or "").strip()
    if not database:
        raise ToolCallValidationError("blast payload requires database")
    query_keys = ("query_fasta", "query_sequence", "query_sequences")
    if not any(payload.get(key) for key in query_keys):
        raise ToolCallValidationError(
            "blast payload requires query_sequence, query_sequences, or query_fasta"
        )
    result: dict[str, Any] = {
        "query": str(payload.get("query") or ""),
        "database": database,
    }
    for key in query_keys:
        if payload.get(key) is not None:
            result[key] = payload[key]
    max_target_seqs = _int_payload(payload, "max_target_seqs", default=5)
    if max_target_seqs < 1:
        raise ToolCallValidationError("blast max_target_seqs must be positive")
    result["max_target_seqs"] = max_target_seqs
    task = str(payload.get("task") or "").strip()
    if task:
        allowed_tasks = {"blastn", "blastn-short", "megablast", "dc-megablast"}
        if task not in allowed_tasks:
            raise ToolCallValidationError(f"unsupported BLAST task: {task}")
        result["task"] = task
    if payload.get("evalue") is not None:
        result["evalue"] = str(payload["evalue"])
    return result


def _canonical_terraform_payload(
    payload: dict[str, Any], config: ToolCallValidationConfig, warnings: list[str]
) -> dict[str, Any]:
    if is_config_request(payload):
        return _canonical_config_payload(payload)
    operation = str(payload.get("operation") or payload.get("command") or "plan").strip().lower()
    if operation.startswith("terraform "):
        operation = operation.removeprefix("terraform ").strip()
    operation = operation.split()[0] if operation else "plan"
    if operation not in TERRAFORM_ALLOWED_OPERATIONS:
        raise ToolCallValidationError(f"terraform operation is not allowed: {operation}")
    if config.terraform_config_dir is None:
        raise ToolCallValidationError("terraform_config_dir is required for Terraform tool calls")
    if payload.get("config_dir") is not None and str(payload["config_dir"]) != str(
        config.terraform_config_dir
    ):
        warnings.append("ignored model-proposed Terraform config_dir in favor of runtime config")
    variables = payload.get("variables") or {}
    if not isinstance(variables, dict):
        raise ToolCallValidationError("terraform variables must be an object")
    return {
        "query": str(payload.get("query") or ""),
        "operation": operation,
        "config_dir": str(config.terraform_config_dir),
        "variables": {str(key): value for key, value in sorted(variables.items())},
    }


def _canonical_timesfm_payload(payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if is_config_request(payload):
        return _canonical_config_payload(payload)
    horizon = _int_payload(payload, "horizon", default=8)
    if horizon < 1:
        raise ToolCallValidationError("timesfm horizon must be positive")
    if horizon > 256:
        raise ToolCallValidationError("timesfm horizon must be <= 256")
    result: dict[str, Any] = {
        "query": str(payload.get("query") or ""),
        "horizon": horizon,
        "period": str(payload.get("period") or "week"),
    }
    if "series" in payload:
        if not isinstance(payload["series"], list):
            raise ToolCallValidationError("timesfm series must be a list")
        result["series"] = payload["series"]
    else:
        values = payload.get("values", payload.get("history"))
        if not isinstance(values, list | tuple):
            raise ToolCallValidationError("timesfm payload requires values, history, or series")
        parsed_values = [float(value) for value in values]
        if len(parsed_values) < 8:
            raise ToolCallValidationError("timesfm requires at least 8 historical observations")
        if any(not math.isfinite(value) for value in parsed_values):
            raise ToolCallValidationError("timesfm values must be finite")
        result["values"] = parsed_values
    if payload.get("series_id") is not None:
        result["series_id"] = str(payload["series_id"])
    if payload.get("current_inventory") is not None:
        result["current_inventory"] = float(payload["current_inventory"])
    return result


def _canonical_yosys_payload(payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if is_config_request(payload):
        return _canonical_config_payload(payload)
    operation = str(payload.get("operation") or payload.get("command") or "synthesize")
    operation = operation.strip().lower()
    if operation.startswith("yosys "):
        operation = operation.removeprefix("yosys ").strip()
    operation = operation.split()[0] if operation else "synthesize"
    if operation not in YOSYS_ALLOWED_OPERATIONS:
        raise ToolCallValidationError(f"unsupported Yosys operation: {operation}")
    top_module = str(payload.get("top_module") or "").strip()
    if not top_module:
        raise ToolCallValidationError("yosys payload requires top_module")
    result: dict[str, Any] = {
        "query": str(payload.get("query") or ""),
        "operation": "synthesize" if operation in {"synth", "compile"} else operation,
        "top_module": top_module,
    }
    source_keys = ("verilog_text", "verilog_sources", "rtl_files", "source_files")
    if not any(payload.get(key) for key in source_keys):
        raise ToolCallValidationError(
            "yosys payload requires verilog_text, verilog_sources, rtl_files, or source_files"
        )
    for key in source_keys:
        if payload.get(key) is not None:
            result[key] = payload[key]
    if payload.get("filename") is not None:
        result["filename"] = str(payload["filename"])
    return result


def _canonical_openroad_payload(payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    if is_config_request(payload):
        return _canonical_config_payload(payload)
    operation = str(payload.get("operation") or payload.get("command") or "analyze")
    operation = operation.strip().lower()
    if operation.startswith("openroad "):
        operation = operation.removeprefix("openroad ").strip()
    operation = operation.split()[0] if operation else "analyze"
    if operation not in OPENROAD_ALLOWED_OPERATIONS:
        raise ToolCallValidationError(f"unsupported OpenROAD operation: {operation}")
    top_module = str(payload.get("top_module") or "").strip()
    if not top_module:
        raise ToolCallValidationError("openroad payload requires top_module")
    result: dict[str, Any] = {
        "query": str(payload.get("query") or ""),
        "operation": operation,
        "top_module": top_module,
    }
    netlist_keys = ("netlist_text", "netlist_path", "netlist_from")
    if not any(payload.get(key) for key in netlist_keys):
        raise ToolCallValidationError(
            "openroad payload requires netlist_text, netlist_path, or netlist_from"
        )
    for key in (
        "netlist_text",
        "netlist_path",
        "netlist_from",
        "liberty_files",
        "lef_files",
        "sdc_file",
        "die_area",
        "core_area",
    ):
        if payload.get(key) is not None:
            result[key] = payload[key]
    return result


def _append_proposal_event(
    ledger: ProvenanceLedger,
    proposal: ToolCallProposal,
    model_response: ModelDriverResult,
    config: ToolCallValidationConfig,
) -> ProvenanceEvent:
    return ledger.append(
        ProvenanceEvent(
            EventType.LLM_TOOL_CALL_PROPOSED,
            config.tenant_id,
            "model-driver",
            {
                "request_id": config.request_id,
                "specialist_id": proposal.specialist_id,
                "task_type": proposal.task_type,
                "input_payload_hash": stable_hash(proposal.input_payload),
                "confidence": proposal.confidence,
                "reasoning": proposal.reasoning,
                "provider": model_response.provider,
                "model_id": model_response.model_id,
                "raw_response": proposal.raw_response,
            },
            parent_event_ids=_parent_event_ids(config.provenance_context),
        )
    )


def _proposal_from_payload(payload: dict[str, Any], raw_response: str) -> ToolCallProposal:
    if isinstance(payload.get("function_call"), dict):
        payload = {**payload, **dict(payload["function_call"])}
    specialist = payload.get("specialist_id", payload.get("tool", payload.get("name")))
    if isinstance(specialist, dict):
        specialist = (
            specialist.get("name") or specialist.get("specialist_id") or specialist.get("tool")
        )
    specialist_id = None if specialist is None else str(specialist).strip()
    if specialist_id and specialist_id.lower() in NO_TOOL_VALUES:
        specialist_id = None
    task_type = payload.get("task_type")
    input_payload = payload.get("input_payload", payload.get("payload", payload.get("arguments")))
    if input_payload is None:
        input_payload = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "specialist_id",
                "tool",
                "name",
                "function_call",
                "task_type",
                "confidence",
                "reasoning",
            }
        }
    parsed_payload = _payload_object(input_payload)
    confidence = payload.get("confidence")
    return ToolCallProposal(
        specialist_id=specialist_id,
        task_type=str(task_type).strip() if task_type is not None and str(task_type) else None,
        input_payload=parsed_payload,
        confidence=float(confidence) if confidence is not None else None,
        reasoning=str(payload.get("reasoning") or payload.get("reason") or ""),
        raw_response=raw_response,
    )


def _response_candidates(response: str) -> list[str]:
    stripped = response.strip()
    fenced = [match.group(1).strip() for match in _FENCED_RE.finditer(stripped)]
    json_object = _JSON_OBJECT_RE.search(stripped)
    candidates = [*fenced]
    candidates.append(stripped)
    if json_object is not None:
        candidates.append(json_object.group(0))
    return candidates


def _load_object(candidate: str) -> dict[str, Any] | None:
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
        loaded = ast.literal_eval(candidate)
        if not isinstance(loaded, dict):
            raise ToolCallParseError("tool-call payload must be an object")
        return loaded
    name = match.group("name")
    args = match.group("args").strip()
    if args.startswith("{"):
        loaded = json.loads(args)
    else:
        pythonish = _KEY_VALUE_RE.sub(r'"\1":', args)
        loaded = ast.literal_eval("{" + pythonish + "}")
    if not isinstance(loaded, dict):
        raise ToolCallParseError("function-call payload must be an object")
    loaded.setdefault("specialist_id", name)
    return loaded


def _payload_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        loaded = _load_object(value)
        if loaded is not None:
            return loaded
    raise ToolCallParseError("input_payload must be an object")


def _valid_fen(fen: str) -> bool:
    fields = fen.split()
    if len(fields) != 6:
        return False
    rows = fields[0].split("/")
    return len(rows) == 8 and fields[1] in {"w", "b"}


def _int_payload(payload: dict[str, Any], key: str, *, default: int) -> int:
    value = payload.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ToolCallValidationError(f"{key} must be an integer") from exc


def _default_task_type(card: SpecialistCard) -> str:
    return sorted(card.supported_task_types)[0]


def _parent_event_ids(provenance_context: dict[str, Any]) -> list[str]:
    parent_event_id = provenance_context.get("parent_event_id")
    return [str(parent_event_id)] if parent_event_id else []
