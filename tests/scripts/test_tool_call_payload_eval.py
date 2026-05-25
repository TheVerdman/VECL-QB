from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.tool_call_payload_eval import run_tool_call_payload_eval
from vecl.provenance.events import stable_hash
from vecl.qb.model_driver import ModelDriverResult
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.specialists.artifacts import ContentAddressedStore

FEN = "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2"


class FakeDriver:
    provider = "fake"
    model_id = "fake-tool-caller"

    def synthesize(self, prompt: str) -> ModelDriverResult:
        if "tool-call author" in prompt and "depth 12" in prompt:
            return _result(
                {
                    "specialist_id": "stockfish",
                    "task_type": "chess_eval",
                    "input_payload": {
                        "query": "analyze the FEN",
                        "fen": FEN,
                        "depth": 12,
                    },
                    "confidence": 0.99,
                    "reasoning": "The user asked for chess analysis.",
                }
            )
        if "tool-call author" in prompt and "symbolic math" in prompt:
            return _result(
                {
                    "specialist_id": "sympy",
                    "task_type": "symbolic_math",
                    "input_payload": {
                        "operation": "simplify",
                        "expression": "(x + 1)^2",
                    },
                    "confidence": 0.97,
                    "reasoning": "The user asked for exact symbolic simplification.",
                }
            )
        return ModelDriverResult(
            raw_text="Final answer uses the verified specialist output.",
            provider=self.provider,
            model_id=self.model_id,
        )


class FakeArtifactSpecialist(Specialist):
    def __init__(
        self,
        specialist_id: str,
        task_type: str,
        output_format: str,
        store: ContentAddressedStore,
    ) -> None:
        self.specialist_id = specialist_id
        self.task_type = task_type
        self.output_format = output_format
        self.store = store
        self.seen_payloads: list[dict[str, Any]] = []

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == self.task_type

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        self.seen_payloads.append(dict(request.input_payload))
        record = self.store.write_text(
            json.dumps(request.input_payload, sort_keys=True),
            producer_specialist_id=self.specialist_id,
            producer_version="test",
            input_hash=stable_hash(request.input_payload),
            output_format=self.output_format,
            parent_event_id=str(request.provenance_context["parent_event_id"]),
        )
        claim = SpecialistClaim(
            f"claim-{self.specialist_id}",
            self.specialist_id,
            f"{self.specialist_id} result",
            self.task_type,
            0.9,
            evidence_ids=[stable_hash(request.input_payload)],
            artifact_ids=[record.artifact_id],
        )
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [claim],
            request.tenant_id,
            cost_metadata={"artifact_records": [record.to_payload()]},
        )


def test_run_tool_call_payload_eval_with_fake_driver_executes_and_synthesizes(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    stockfish = FakeArtifactSpecialist("stockfish", "chess_eval", "txt", store)
    sympy = FakeArtifactSpecialist("sympy", "symbolic_math", "tex", store)
    entries = [
        {
            "id": "stockfish_depth_12",
            "prompt": "Analyze this chess position with Stockfish at depth 12.",
            "expected_specialist_id": "stockfish",
            "expected_task_type": "chess_eval",
        },
        {
            "id": "sympy_simplify",
            "prompt": "Use symbolic math to simplify (x + 1)^2.",
            "expected_specialist_id": "sympy",
            "expected_task_type": "symbolic_math",
        },
    ]

    summary = run_tool_call_payload_eval(
        entries=entries,
        driver=FakeDriver(),
        cards={
            "stockfish": _card("stockfish", "chess_eval"),
            "sympy": _card("sympy", "symbolic_math"),
        },
        specialists={"stockfish": stockfish, "sympy": sympy},
    )

    assert summary["proposal_success_rate"] == 1.0
    assert summary["validation_success_rate"] == 1.0
    assert summary["execution_success_rate"] == 1.0
    assert summary["synthesis_success_rate"] == 1.0
    assert summary["tool_call_events"] == 2
    assert summary["artifact_events"] == 2
    assert stockfish.seen_payloads[0]["depth"] == 12
    assert sympy.seen_payloads[0]["operation"] == "simplify"


def _result(payload: dict[str, Any]) -> ModelDriverResult:
    return ModelDriverResult(
        raw_text=json.dumps(payload), provider="fake", model_id="fake-tool-caller"
    )


def _card(specialist_id: str, task_type: str) -> SpecialistCard:
    return SpecialistCard(
        specialist_id,
        {task_type},
        {"description": f"{specialist_id} specialist"},
        cost_hint=1.0,
        latency_hint=1.0,
        version="test",
        effective_trust=0.9,
        description=f"{specialist_id} specialist",
    )
