#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl.qb.model_driver import ModelDriver, resolve_model_driver  # noqa: E402
from vecl.qb.router import SpecialistCard  # noqa: E402
from vecl.qb.specialist import SpecialistRequest  # noqa: E402

TERRAFORM_SYNTHESIS_PROMPT = """You are VECL-QB's infrastructure reviewer.
Use only the verified Terraform artifact below. Return only JSON:
{"used_terraform":true,"applied":false,"create":1,"answer":"<short answer>"}

User request:
What would this Terraform module change?

Verified Terraform artifact:
{"format_version":"1.2","resource_changes":[{"address":"terraform_data.inventory","change":{"actions":["create"]}}]}
"""

STOCKFISH_SYNTHESIS_PROMPT = """You are VECL-QB's chess analyst.
Use only the verified Stockfish artifact below. Return only JSON:
{"used_stockfish":true,"best_move":"g8f6","answer":"<short answer>"}

User request:
What is the best move in this position?

Verified Stockfish artifact:
bestmove g8f6
info depth 12 score cp 20 pv g8f6
"""


@dataclass(frozen=True)
class SynthesisCase:
    case_id: str
    prompt: str
    required_fields: dict[str, Any]


def main() -> int:
    fixture_path = Path(os.environ.get("VECL_MODEL_DRIVER_EVAL_FIXTURE", _default_fixture_path()))
    entries = json.loads(fixture_path.read_text())
    driver = resolve_model_driver()
    summary = run_model_driver_eval(entries=entries, driver=driver)
    print("MODEL_DRIVER_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if summary["route_success_rate"] < 1.0:
        return 1
    if summary["synthesis_success_rate"] < 1.0:
        return 1
    if summary["route_parse_failures"] != 0:
        return 1
    if summary["synthesis_parse_failures"] != 0:
        return 1
    return 0


def run_model_driver_eval(
    *, entries: Sequence[dict[str, Any]], driver: ModelDriver
) -> dict[str, Any]:
    cards = [terraform_card(), stockfish_card()]
    route_rows: list[dict[str, Any]] = []
    route_success = 0
    route_parse_failures = 0
    for index, entry in enumerate(entries, start=1):
        request = SpecialistRequest(
            f"model-driver-eval-{index}",
            "model-driver-eval",
            entry["task_type"],
            dict(entry["input_payload"]),
            {},
            {"parent_event_id": f"evt-model-driver-eval-{index}"},
        )
        try:
            result = driver.route(request, cards)
            actual = result.decision.specialist_id
            raw_text = result.response.raw_text
        except Exception as exc:
            route_parse_failures += 1
            actual = None
            raw_text = ""
            result = None
            reason = str(exc)
        else:
            reason = result.decision.reasoning
        expected = entry.get("expected_specialist_id")
        passed = actual == expected
        route_success += int(passed)
        row = {
            "id": entry["id"],
            "index": index,
            "expected_specialist_id": expected,
            "actual_specialist_id": actual,
            "passed": passed,
            "reasoning": reason,
            "raw_text": raw_text,
        }
        route_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    synthesis_rows: list[dict[str, Any]] = []
    synthesis_success = 0
    synthesis_parse_failures = 0
    for case in synthesis_cases():
        response = driver.synthesize(case.prompt)
        try:
            payload = _parse_json_object(response.raw_text)
        except ValueError as exc:
            synthesis_parse_failures += 1
            passed = False
            reasons = [str(exc)]
            payload = {}
        else:
            reasons = _score_required_fields(payload, case.required_fields)
            passed = not reasons
        synthesis_success += int(passed)
        row = {
            "id": case.case_id,
            "passed": passed,
            "reasons": reasons,
            "raw_text": response.raw_text,
            "usage": response.usage,
            "metadata": response.metadata,
        }
        synthesis_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    return {
        "provider": driver.provider,
        "model_id": driver.model_id,
        "route_total": len(entries),
        "route_success": route_success,
        "route_parse_failures": route_parse_failures,
        "route_success_rate": _rate(route_success, len(entries)),
        "synthesis_total": len(synthesis_rows),
        "synthesis_success": synthesis_success,
        "synthesis_parse_failures": synthesis_parse_failures,
        "synthesis_success_rate": _rate(synthesis_success, len(synthesis_rows)),
        "route_rows": route_rows,
        "synthesis_rows": synthesis_rows,
    }


def synthesis_cases() -> list[SynthesisCase]:
    return [
        SynthesisCase(
            "terraform_plan_synthesis",
            TERRAFORM_SYNTHESIS_PROMPT,
            {"used_terraform": True, "applied": False, "create": 1},
        ),
        SynthesisCase(
            "stockfish_synthesis",
            STOCKFISH_SYNTHESIS_PROMPT,
            {"used_stockfish": True, "best_move": "g8f6"},
        ),
    ]


def terraform_card() -> SpecialistCard:
    return SpecialistCard(
        "terraform",
        {"infrastructure_plan"},
        {"description": "Terraform validate and plan only; apply and destroy are disabled."},
        cost_hint=0.5,
        latency_hint=0.6,
        version="terraform-plan-only",
        effective_trust=0.95,
        description="Terraform plan-only infrastructure specialist.",
    )


def stockfish_card() -> SpecialistCard:
    return SpecialistCard(
        "stockfish",
        {"chess_eval"},
        {"description": "Chess engine analysis from FEN."},
        cost_hint=1.0,
        latency_hint=1.0,
        version="stockfish-18",
        effective_trust=0.9,
        description="Analyze chess positions with Stockfish 18.",
    )


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response did not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response JSON was not an object")
    return payload


def _score_required_fields(payload: dict[str, Any], required: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key, expected in required.items():
        if payload.get(key) != expected:
            reasons.append(f"{key} mismatch")
    return reasons


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _default_fixture_path() -> str:
    return str(
        Path(__file__).parents[1] / "tests" / "qb" / "fixtures" / "model_driver_eval_v0.json"
    )


if __name__ == "__main__":
    raise SystemExit(main())
