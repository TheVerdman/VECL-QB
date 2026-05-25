from __future__ import annotations

import json
import re

from scripts.timesfm_demand_eval_gemma import (
    DemandAnswer,
    DemandAnswerContext,
    _default_fixture_path,
    parse_demand_answer,
    run_demand_eval,
    score_demand_answer,
    timesfm_card,
    timesfm_inventory_chain_plan,
)
from vecl.specialists.timesfm_specialist import DeterministicTimesFMRunner


def test_timesfm_chain_plan_and_card_shape() -> None:
    plan = timesfm_inventory_chain_plan()
    card = timesfm_card()

    assert plan.plan_id == "timesfm-sympy-inventory"
    assert [step.step_id for step in plan.steps] == ["forecast", "cumulative"]
    assert plan.steps[1].inputs_from == ("forecast",)
    assert card.specialist_id == "timesfm"
    assert "Forecast" in card.description


def test_parse_demand_answer_accepts_fenced_json() -> None:
    answer = parse_demand_answer(
        """
        ```json
        {
          "reorder_needed": true,
          "forecast_sum": 552,
          "current_inventory": 450,
          "recommendation": "Increase inventory before the next four weeks.",
          "used_timesfm": true,
          "used_sympy": true
        }
        ```
        """
    )

    assert answer.reorder_needed
    assert answer.forecast_sum == 552
    assert answer.used_timesfm
    assert answer.used_sympy


def test_demand_eval_with_mock_inference_routes_chains_and_answers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    entries = json.loads(_default_fixture_path().read_text())

    summary = run_demand_eval(
        entries=entries,
        runner=DeterministicTimesFMRunner(),
        route_fn=_mock_route,
        answer_fn=_mock_answer,
        model_id="mock-gemma-31b",
        artifact_root=tmp_path,
    )

    assert summary["timesfm_routing_rate"] == 1.0
    assert summary["timesfm_chain_success_rate"] == 1.0
    assert summary["timesfm_answer_success_rate"] == 1.0
    assert summary["out_of_domain_timesfm_rate"] == 0.0
    assert summary["decision_events"] == len(entries)
    assert summary["fallback_events"] == 0
    assert summary["artifact_events"] == 12
    assert all(row["answer_status"] in {"PASSED", "NOT_RUN"} for row in summary["rows"])


def test_score_demand_answer_requires_timesfm_and_sympy() -> None:
    context = DemandAnswerContext(
        question="Should we increase inventory?",
        current_inventory=450.0,
        forecast_json={"series": [{"forecast_sum": 552.0}]},
        sympy_claim_text="operation=simplify; result=552.0",
        expected_reorder_needed=True,
        expected_forecast_sum=552.0,
        artifact_ids=("artifact-forecast",),
    )
    bad_answer = DemandAnswer(
        reorder_needed=True,
        forecast_sum=552.0,
        current_inventory=450.0,
        recommendation="Increase inventory.",
        used_timesfm=True,
        used_sympy=False,
        raw_response="{}",
    )

    assert not score_demand_answer(context, bad_answer)["passed"]


def _mock_route(prompt: str) -> str:
    request = prompt.split("Request:", 1)[1]
    if "history" in request and "current_inventory" in request:
        return json.dumps(
            {
                "tool": "timesfm",
                "confidence": 0.98,
                "reasoning": "Demand history needs time-series forecasting.",
            }
        )
    return json.dumps(
        {
            "tool": None,
            "confidence": 0.91,
            "reasoning": "No forecasting specialist is appropriate.",
        }
    )


def _mock_answer(prompt: str) -> str:
    inventory = _extract_number_after(prompt, "Current inventory:")
    forecast_sum = _extract_sympy_result(prompt)
    return json.dumps(
        {
            "reorder_needed": forecast_sum > inventory,
            "forecast_sum": forecast_sum,
            "current_inventory": inventory,
            "recommendation": (
                "Increase inventory because forecast demand exceeds stock."
                if forecast_sum > inventory
                else "Inventory is enough for the forecast window."
            ),
            "used_timesfm": True,
            "used_sympy": True,
        }
    )


def _extract_number_after(text: str, marker: str) -> float:
    tail = text.split(marker, 1)[1].strip()
    return float(tail.splitlines()[0])


def _extract_sympy_result(text: str) -> float:
    match = re.search(r"\bresult=([^\s;]+)", text)
    if match is None:
        raise AssertionError("mock prompt did not contain a SymPy result")
    return float(match.group(1))
