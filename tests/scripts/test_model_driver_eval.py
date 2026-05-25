from __future__ import annotations

import json
from pathlib import Path

from scripts.model_driver_eval import run_model_driver_eval
from vecl.qb.model_driver import ModelDriverResult, ModelRouteResult
from vecl.qb.prompted_router import parse_routing_response
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import SpecialistRequest


class FakeDriver:
    provider = "fake"
    model_id = "fake-model"

    def route(
        self, request: SpecialistRequest, specialist_cards: list[SpecialistCard]
    ) -> ModelRouteResult:
        del specialist_cards
        query = str(request.input_payload.get("query") or "")
        if "Terraform" in query:
            raw = '{"tool":"terraform","confidence":0.95,"reasoning":"infrastructure"}'
        elif "chess" in query:
            raw = '{"tool":"stockfish","confidence":0.95,"reasoning":"chess"}'
        else:
            raw = '{"tool":null,"confidence":0.8,"reasoning":"unsupported"}'
        return ModelRouteResult(
            decision=parse_routing_response(raw),
            response=ModelDriverResult(self.provider, self.model_id, raw),
        )

    def synthesize(self, prompt: str) -> ModelDriverResult:
        if "Terraform artifact" in prompt:
            raw = json.dumps(
                {
                    "used_terraform": True,
                    "applied": False,
                    "create": 1,
                    "answer": "One resource would be created.",
                }
            )
        else:
            raw = json.dumps(
                {
                    "used_stockfish": True,
                    "best_move": "g8f6",
                    "answer": "Stockfish recommends g8f6.",
                }
            )
        return ModelDriverResult(self.provider, self.model_id, raw)


def test_model_driver_eval_scores_route_and_synthesis() -> None:
    entries = json.loads(
        (Path(__file__).parents[1] / "qb" / "fixtures" / "model_driver_eval_v0.json").read_text()
    )

    summary = run_model_driver_eval(entries=entries, driver=FakeDriver())

    assert summary["route_success_rate"] == 1.0
    assert summary["synthesis_success_rate"] == 1.0
    assert summary["route_parse_failures"] == 0
    assert summary["synthesis_parse_failures"] == 0
    assert [row["actual_specialist_id"] for row in summary["route_rows"]] == [
        "terraform",
        "stockfish",
        None,
    ]
