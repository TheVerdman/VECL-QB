#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vecl._paths import environment_directory
from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb._llm_inference import DEFAULT_ROUTING_MODEL_ID, gemma_route_once
from vecl.qb.orchestrator import QBOrchestrator
from vecl.qb.planner import ChainPlan, ChainPlanner, ChainStep
from vecl.qb.prompted_router import PromptedLLMRouter
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import SpecialistRequest
from vecl.qb.verifier import VerificationPolicy
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.sympy_specialist import SymPySpecialist
from vecl.specialists.timesfm_specialist import (
    DeterministicTimesFMRunner,
    LocalTimesFMRunner,
    TimesFMRunner,
    TimesFMSpecialist,
)

ANSWER_PROMPT_TEMPLATE = """You are VECL-QB's inventory planning answerer.
A specialist chain has already run. Use the TimesFM forecast and SymPy cumulative-demand result below.
Do not invent a new forecast. Do not present the forecast as guaranteed.
Return only JSON with this schema:
{{"reorder_needed": true|false, "forecast_sum": <number>, "current_inventory": <number>, "recommendation": "<short answer>", "used_timesfm": true, "used_sympy": true}}

Question:
{question}

Current inventory:
{current_inventory}

TimesFM forecast JSON artifact:
{forecast_json}

SymPy cumulative-demand claim:
{sympy_claim_text}
"""

_FENCED_RE = re.compile(r"```(?:json|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SYMPY_RESULT_RE = re.compile(r"\bresult=([^\s;]+)")


@dataclass(frozen=True)
class DemandAnswerContext:
    question: str
    current_inventory: float
    forecast_json: dict[str, Any]
    sympy_claim_text: str
    expected_reorder_needed: bool
    expected_forecast_sum: float
    artifact_ids: tuple[str, ...]


@dataclass(frozen=True)
class DemandAnswer:
    reorder_needed: bool
    forecast_sum: float
    current_inventory: float
    recommendation: str
    used_timesfm: bool
    used_sympy: bool
    raw_response: str


class DemandAnswerParseError(ValueError):
    pass


def main() -> int:
    fixture_path = Path(os.environ.get("VECL_TIMESFM_DEMAND_EVAL_FIXTURE", _default_fixture_path()))
    entries = json.loads(fixture_path.read_text())
    model_id = os.environ.get("VECL_ROUTING_MODEL_ID", DEFAULT_ROUTING_MODEL_ID)
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is required for the TimesFM demand eval.", flush=True)
        return 2
    max_entries = int(os.environ.get("VECL_TIMESFM_DEMAND_EVAL_MAX_ENTRIES", "6"))
    entries = entries[:max_entries]
    min_answer_rate = float(os.environ.get("VECL_TIMESFM_DEMAND_MIN_ANSWER_RATE", "0.75"))
    runner = LocalTimesFMRunner(
        model_id=os.environ.get("TIMESFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
    )
    summary = run_demand_eval(
        entries=entries,
        runner=runner,
        route_fn=gemma_route_once,
        answer_fn=gemma_route_once,
        model_id=model_id,
    )
    print("TIMESFM_DEMAND_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["timesfm_routing_rate"] < 0.9:
        return 1
    if summary["timesfm_chain_success_rate"] < 0.9:
        return 1
    if summary["timesfm_answer_success_rate"] < min_answer_rate:
        return 1
    if summary["out_of_domain_timesfm_rate"] > 0.1:
        return 1
    if summary["fallback_events"] != 0:
        return 1
    return 0


def run_demand_eval(
    *,
    entries: Sequence[dict[str, Any]],
    runner: TimesFMRunner | None = None,
    route_fn: Callable[[str], str],
    answer_fn: Callable[[str], str],
    model_id: str,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    artifact_root = artifact_root or environment_directory(
        "VECL_ARTIFACT_STORE", prefix="vecl-timesfm-demand-artifacts-"
    )
    store = ContentAddressedStore(artifact_root)
    ledger = ProvenanceLedger()
    router = PromptedLLMRouter(ledger=ledger, inference_fn=route_fn, model_id=model_id)
    timesfm = TimesFMSpecialist(
        task_types=("demand_forecast",),
        runner=runner or DeterministicTimesFMRunner(),
        artifact_store=store,
    )
    sympy = SymPySpecialist(task_types=("demand_forecast",), artifact_store=store)
    router.register_specialist(timesfm_card(), timesfm)
    router.register_specialist(sympy_card(), sympy)
    planner = ChainPlanner({"demand_forecast": timesfm_inventory_chain_plan()})
    orchestrator = QBOrchestrator(
        router,
        ledger,
        policy=VerificationPolicy(required_claim_types={"time_series_forecast", "symbolic_math"}),
    )

    demand_total = 0
    demand_routed = 0
    chains_started = 0
    chains_passed = 0
    answer_calls = 0
    answers_passed = 0
    parse_failures = 0
    out_total = 0
    out_routed_to_timesfm = 0
    max_prompt_chars = 0
    rows: list[dict[str, Any]] = []

    for index, entry in enumerate(entries, start=1):
        route_request_event = ledger.append(
            ProvenanceEvent(
                EventType.EVIDENCE_INGESTED,
                "timesfm-demand-eval",
                "timesfm-demand-eval",
                {"entry_id": entry["id"], "purpose": "preflight-routing"},
            )
        )
        route_request = SpecialistRequest(
            f"timesfm-route-{index}",
            "timesfm-demand-eval",
            entry["task_type"],
            entry["input_payload"],
            {},
            {"parent_event_id": route_request_event.event_id},
        )
        routed = router.route(route_request, max_specialists=1)
        routed_ids = [specialist.specialist_id for specialist in routed]
        if entry["expected_specialist_id"] != "timesfm":
            out_total += 1
            out_routed_to_timesfm += int("timesfm" in routed_ids)
            row = {
                "id": entry["id"],
                "index": index,
                "expected": None,
                "routed_ids": routed_ids,
                "chain_status": "NOT_RUN",
                "answer_status": "NOT_RUN",
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            continue

        demand_total += 1
        if routed_ids != ["timesfm"]:
            row = {
                "id": entry["id"],
                "index": index,
                "expected": "timesfm",
                "routed_ids": routed_ids,
                "chain_status": "NOT_RUN",
                "answer_status": "ROUTING_FAILED",
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            continue

        demand_routed += 1
        artifact_start = len(ledger.find_by_type(EventType.ARTIFACT_PRODUCED))
        plan = planner.plan(route_request, router.specialists)
        result = orchestrator.run_task(
            tenant_id="timesfm-demand-eval",
            task_type=entry["task_type"],
            input_payload=entry["input_payload"],
            chain_plan=plan,
        )
        chains_started += 1
        chain_passed = result.verification_status == "PASSED"
        chains_passed += int(chain_passed)
        artifact_payloads = [
            event.payload
            for event in ledger.find_by_type(EventType.ARTIFACT_PRODUCED)[artifact_start:]
        ]
        context = build_demand_answer_context(
            entry=entry,
            chain_answer_text=result.answer_text,
            artifact_payloads=artifact_payloads,
        )
        prompt = build_answer_prompt(context)
        max_prompt_chars = max(max_prompt_chars, len(prompt))
        answer_calls += 1
        raw_answer = answer_fn(prompt)
        try:
            answer = parse_demand_answer(raw_answer)
            score = score_demand_answer(context, answer)
            answer_passed = bool(score["passed"] and chain_passed)
            answers_passed += int(answer_passed)
            answer_status = "PASSED" if answer_passed else "FAILED"
        except DemandAnswerParseError as exc:
            parse_failures += 1
            score = {"passed": False, "reason": str(exc)}
            answer_status = "PARSE_FAILED"
        row = {
            "id": entry["id"],
            "index": index,
            "expected": "timesfm",
            "routed_ids": routed_ids,
            "chain_status": result.verification_status,
            "answer_status": answer_status,
            "expected_reorder_needed": context.expected_reorder_needed,
            "expected_forecast_sum": context.expected_forecast_sum,
            "score": score,
            "prompt_chars": len(prompt),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = demand_eval_summary(
        ledger=ledger,
        model_id=model_id,
        demand_total=demand_total,
        demand_routed=demand_routed,
        chains_started=chains_started,
        chains_passed=chains_passed,
        answer_calls=answer_calls,
        answers_passed=answers_passed,
        parse_failures=parse_failures,
        out_total=out_total,
        out_routed_to_timesfm=out_routed_to_timesfm,
        max_prompt_chars=max_prompt_chars,
    )
    summary["rows"] = rows
    return summary


def timesfm_inventory_chain_plan() -> ChainPlan:
    return ChainPlan(
        "timesfm-sympy-inventory",
        "demand_forecast",
        (
            ChainStep("forecast", "timesfm", expected_artifact_type="json"),
            ChainStep(
                "cumulative",
                "sympy",
                inputs_from=("forecast",),
                parameters={
                    "operation": "simplify",
                    "expression_from": "timesfm_forecast_sum",
                },
                expected_artifact_type="tex",
            ),
        ),
    )


def timesfm_card() -> SpecialistCard:
    return SpecialistCard(
        "timesfm",
        {"demand_forecast"},
        {
            "description": (
                "Use for inventory, demand, sales, sensor, or other time-series "
                "forecasting with point forecasts and prediction intervals."
            )
        },
        cost_hint=0.7,
        latency_hint=0.8,
        version="timesfm-2.5-200m-pytorch",
        effective_trust=0.9,
        description="Forecast inventory or demand time series with TimesFM.",
    )


def sympy_card() -> SpecialistCard:
    return SpecialistCard(
        "sympy",
        {"demand_forecast"},
        {
            "description": (
                "Use only as a downstream calculator for exact arithmetic over an "
                "existing forecast, such as cumulative demand or maxima."
            )
        },
        cost_hint=0.1,
        latency_hint=0.1,
        version="sympy-1.14",
        effective_trust=0.95,
        description="Compute exact arithmetic over forecast outputs.",
    )


def build_demand_answer_context(
    *,
    entry: dict[str, Any],
    chain_answer_text: str,
    artifact_payloads: list[dict[str, Any]],
) -> DemandAnswerContext:
    forecast_json: dict[str, Any] | None = None
    artifact_ids: list[str] = []
    for payload in artifact_payloads:
        artifact_ids.append(str(payload["artifact_id"]))
        if payload.get("output_format") == "json" and forecast_json is None:
            forecast_json = json.loads(read_artifact_payload_text(payload))
    if forecast_json is None:
        raise ValueError("TimesFM JSON artifact is required for answer context")
    sympy_claim_text = _sympy_claim_text(chain_answer_text)
    expected_forecast_sum = _forecast_sum_from_sympy(sympy_claim_text)
    return DemandAnswerContext(
        question=str(entry["input_payload"].get("query", "")),
        current_inventory=float(entry["input_payload"]["current_inventory"]),
        forecast_json=forecast_json,
        sympy_claim_text=sympy_claim_text,
        expected_reorder_needed=bool(entry["expected_reorder_needed"]),
        expected_forecast_sum=expected_forecast_sum,
        artifact_ids=tuple(sorted(artifact_ids)),
    )


def build_answer_prompt(context: DemandAnswerContext) -> str:
    return ANSWER_PROMPT_TEMPLATE.format(
        question=context.question,
        current_inventory=context.current_inventory,
        forecast_json=json.dumps(context.forecast_json, sort_keys=True),
        sympy_claim_text=context.sympy_claim_text,
    )


def read_artifact_payload_text(payload: dict[str, Any]) -> str:
    path = Path(str(payload["output_path"]))
    content = path.read_bytes()
    expected_hash = str(payload["output_hash"])
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("artifact content hash mismatch")
    return content.decode()


def parse_demand_answer(response: str) -> DemandAnswer:
    for candidate in _json_candidates(response):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        recommendation = str(payload.get("recommendation", "")).strip()
        if not recommendation:
            continue
        try:
            return DemandAnswer(
                reorder_needed=_parse_bool(payload["reorder_needed"]),
                forecast_sum=float(payload["forecast_sum"]),
                current_inventory=float(payload["current_inventory"]),
                recommendation=recommendation,
                used_timesfm=bool(payload.get("used_timesfm", False)),
                used_sympy=bool(payload.get("used_sympy", False)),
                raw_response=response,
            )
        except (KeyError, TypeError, ValueError):
            continue
    raise DemandAnswerParseError("could not parse Gemma inventory answer")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "no", "0"}:
        return False
    raise ValueError("expected boolean")


def score_demand_answer(context: DemandAnswerContext, answer: DemandAnswer) -> dict[str, Any]:
    reorder_matches = answer.reorder_needed == context.expected_reorder_needed
    forecast_sum_matches = abs(answer.forecast_sum - context.expected_forecast_sum) <= max(
        1.0, abs(context.expected_forecast_sum) * 0.05
    )
    inventory_matches = abs(answer.current_inventory - context.current_inventory) <= 1e-6
    recommendation_mentions_inventory = (
        "inventory" in answer.recommendation.lower() or "stock" in answer.recommendation.lower()
    )
    passed = (
        answer.used_timesfm
        and answer.used_sympy
        and reorder_matches
        and forecast_sum_matches
        and inventory_matches
    )
    return {
        "passed": passed,
        "used_timesfm": answer.used_timesfm,
        "used_sympy": answer.used_sympy,
        "reorder_matches": reorder_matches,
        "forecast_sum_matches": forecast_sum_matches,
        "inventory_matches": inventory_matches,
        "recommendation_mentions_inventory": recommendation_mentions_inventory,
    }


def demand_eval_summary(
    *,
    ledger: ProvenanceLedger,
    model_id: str,
    demand_total: int,
    demand_routed: int,
    chains_started: int,
    chains_passed: int,
    answer_calls: int,
    answers_passed: int,
    parse_failures: int,
    out_total: int,
    out_routed_to_timesfm: int,
    max_prompt_chars: int,
) -> dict[str, Any]:
    decision_events = ledger.find_by_type(EventType.LLM_ROUTING_DECIDED)
    fallback_events = ledger.find_by_type(EventType.ROUTING_FALLBACK)
    artifact_events = ledger.find_by_type(EventType.ARTIFACT_PRODUCED)
    return {
        "model_id": model_id,
        "demand_total": demand_total,
        "timesfm_routing_rate": demand_routed / demand_total if demand_total else 0.0,
        "timesfm_chain_success_rate": chains_passed / demand_total if demand_total else 0.0,
        "timesfm_answer_success_rate": answers_passed / demand_total if demand_total else 0.0,
        "out_of_domain_timesfm_rate": (out_routed_to_timesfm / out_total if out_total else 0.0),
        "chains_started": chains_started,
        "chains_passed": chains_passed,
        "answer_calls": answer_calls,
        "answers_passed": answers_passed,
        "answer_parse_failures": parse_failures,
        "decision_events": len(decision_events),
        "fallback_events": len(fallback_events),
        "artifact_events": len(artifact_events),
        "max_prompt_chars": max_prompt_chars,
        "total_entries": demand_total + out_total,
    }


def _json_candidates(response: str) -> list[str]:
    stripped = response.strip()
    candidates = [match.group(1).strip() for match in _FENCED_RE.finditer(stripped)]
    candidates.append(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    return candidates


def _sympy_claim_text(chain_answer_text: str) -> str:
    marker = "operation=simplify"
    index = chain_answer_text.find(marker)
    if index < 0:
        raise ValueError("SymPy cumulative-demand claim is required")
    return chain_answer_text[index:]


def _forecast_sum_from_sympy(sympy_claim_text: str) -> float:
    match = _SYMPY_RESULT_RE.search(sympy_claim_text)
    if not match:
        raise ValueError("SymPy claim did not contain result=<forecast_sum>")
    return float(match.group(1).strip())


def _default_fixture_path() -> Path:
    packaged = Path(__file__).with_name("timesfm_demand_eval_v0.json")
    if packaged.exists():
        return packaged
    return Path(__file__).parents[1] / "tests" / "qb" / "fixtures" / "timesfm_demand_eval_v0.json"


if __name__ == "__main__":
    raise SystemExit(main())
