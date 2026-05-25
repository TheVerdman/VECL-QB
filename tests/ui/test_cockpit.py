from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vecl.provenance.events import stable_hash
from vecl.qb.model_driver import ModelDriverResult, TextGenerationDriver
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.ui.cockpit import Availability, CockpitBackend, SpecialistSpec
from vecl.ui.server import CockpitRequestHandler


class FakeDriver(TextGenerationDriver):
    provider = "fake"
    model_id = "fake-model"

    def __init__(
        self,
        *,
        route_text: str = '{"tool":"sympy","confidence":0.91,"reasoning":"math"}',
        synthesis_text: str = "Final answer from verified context.",
    ) -> None:
        self.route_text = route_text
        self.synthesis_text = synthesis_text
        self.prompts: list[str] = []

    def synthesize(self, prompt: str) -> ModelDriverResult:
        self.prompts.append(prompt)
        raw_text = self.route_text if "Registered specialists:" in prompt else self.synthesis_text
        return ModelDriverResult(
            provider=self.provider,
            model_id=self.model_id,
            raw_text=raw_text,
            usage={"input_tokens": 10, "output_tokens": 5},
            latency_ms=1,
            metadata={"estimated_cost_usd": 0.00001},
        )


class CountingTerraformSpecialist(Specialist):
    specialist_id = "terraform"

    def __init__(self) -> None:
        self.calls = 0

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "infrastructure_plan"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        self.calls += 1
        claim = SpecialistClaim(
            "claim-terraform",
            self.specialist_id,
            "terraform should not run",
            "terraform_plan",
            0.5,
            evidence_ids=["test"],
        )
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [claim],
            request.tenant_id,
        )


class RecordingStockfishSpecialist(Specialist):
    specialist_id = "stockfish"

    def __init__(self, store: ContentAddressedStore) -> None:
        self.store = store
        self.calls = 0
        self.seen_payloads: list[dict[str, Any]] = []

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "chess_eval"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        self.calls += 1
        self.seen_payloads.append(dict(request.input_payload))
        depth = int(request.input_payload["depth"])
        fen = str(request.input_payload["fen"])
        transcript = "\n".join(
            [
                "$ stockfish",
                "uci",
                "uciok",
                f"position fen {fen}",
                f"go depth {depth}",
                "bestmove e7e6",
                "",
            ]
        )
        record = self.store.write_text(
            transcript,
            producer_specialist_id=self.specialist_id,
            producer_version="stockfish-test",
            input_hash=stable_hash({"fen": fen, "depth": depth}),
            output_format="txt",
            parent_event_id=str(request.provenance_context["parent_event_id"]),
        )
        claim = SpecialistClaim(
            "claim-stockfish-test",
            self.specialist_id,
            "bestmove=e7e6",
            "chess_eval",
            0.9,
            evidence_ids=[stable_hash({"fen": fen})],
            artifact_ids=[record.artifact_id],
        )
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [claim],
            request.tenant_id,
            cost_metadata={"artifact_records": [record.to_payload()]},
        )


def test_backend_runs_sympy_and_exposes_trace(tmp_path: Path) -> None:
    driver = FakeDriver()
    backend = CockpitBackend(
        artifact_root=tmp_path,
        driver_factory=lambda _provider: driver,
    )

    run = backend.submit_chat(
        prompt="Simplify expression: x**2 + 2*x + 1",
        driver_provider="fake",
        payload_override={
            "task_type": "symbolic_math",
            "operation": "simplify",
            "expression": "x**2 + 2*x + 1",
        },
        run_async=False,
    )

    assert run["status"] == "completed"
    assert run["selected_specialist"] == "sympy"
    assert run["final_answer"] == "Final answer from verified context."
    assert [call["kind"] for call in run["model_calls"]] == ["manual_tool_call", "synthesis"]
    assert run["model_calls"][0]["parsed"]["input_payload"]["expression"] == "x**2 + 2*x + 1"
    assert run["artifacts"]
    assert run["artifacts"][0]["output_format"] == "tex"
    artifact = backend.get_artifact_content(str(run["artifacts"][0]["artifact_id"]))
    assert "\\documentclass" in str(artifact["text"])
    event_types = [event["event_type"] for event in run["provenance_events"]]
    assert "LLMToolCallProposed" in event_types
    assert "ChainStarted" in event_types
    assert "ArtifactProduced" in event_types
    assert "ReplayBatchPrepared" in event_types
    assert run["cumulative_cost_usd"] == 0.00001


def test_stockfish_payload_depth_12_reaches_specialist_and_trace(tmp_path: Path) -> None:
    stockfish_holder: dict[str, RecordingStockfishSpecialist] = {}

    def stockfish_factory(store: ContentAddressedStore) -> RecordingStockfishSpecialist:
        specialist = RecordingStockfishSpecialist(store)
        stockfish_holder["specialist"] = specialist
        return specialist

    spec = SpecialistSpec(
        SpecialistCard(
            "stockfish",
            {"chess_eval"},
            {"trust_anchor_id": "stockfish-v18"},
            cost_hint=1.0,
            latency_hint=1.0,
            version="stockfish-test",
            effective_trust=0.9,
            description="Stockfish test specialist.",
        ),
        stockfish_factory,
        lambda: Availability(True, "available", "test stockfish"),
        "txt",
    )
    driver = FakeDriver(
        route_text='{"tool":"stockfish","confidence":0.99,"reasoning":"explicit chess payload"}',
        synthesis_text="Stockfish completed.",
    )
    backend = CockpitBackend(
        artifact_root=tmp_path,
        driver_factory=lambda _provider: driver,
        specialist_specs=[spec],
    )
    payload = {
        "task_type": "chess_eval",
        "input_payload": {
            "query": "Analyze this position with Stockfish and explain the best move.",
            "fen": "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",
            "depth": 12,
        },
    }

    run = backend.submit_chat(
        prompt="Please simplify this expression, but use depth 4 if this is chess.",
        driver_provider="fake",
        payload_override=payload,
        run_async=False,
    )

    specialist = stockfish_holder["specialist"]
    assert specialist.calls == 1
    assert specialist.seen_payloads[0]["depth"] == 12
    assert (
        specialist.seen_payloads[0]["fen"]
        == "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2"
    )
    actual = run["proposed_payload"]["actual_specialist_request"]
    assert actual["task_type"] == "chess_eval"
    assert actual["input_payload"]["depth"] == 12
    assert actual["input_payload"]["fen"] == specialist.seen_payloads[0]["fen"]
    assert run["proposed_payload"]["source"] == "explicit_payload_editor"
    assert run["proposed_payload"]["warnings"]
    artifact = backend.get_artifact_content(str(run["artifacts"][0]["artifact_id"]))
    assert "go depth 12" in str(artifact["text"])


def test_model_authored_tool_call_uses_phase_8e_path(tmp_path: Path) -> None:
    stockfish_holder: dict[str, RecordingStockfishSpecialist] = {}

    def stockfish_factory(store: ContentAddressedStore) -> RecordingStockfishSpecialist:
        specialist = RecordingStockfishSpecialist(store)
        stockfish_holder["specialist"] = specialist
        return specialist

    spec = SpecialistSpec(
        SpecialistCard(
            "stockfish",
            {"chess_eval"},
            {"trust_anchor_id": "stockfish-v18"},
            cost_hint=1.0,
            latency_hint=1.0,
            version="stockfish-test",
            effective_trust=0.9,
            description="Stockfish test specialist.",
        ),
        stockfish_factory,
        lambda: Availability(True, "available", "test stockfish"),
        "txt",
    )
    tool_call_text = json.dumps(
        {
            "specialist_id": "stockfish",
            "task_type": "chess_eval",
            "input_payload": {
                "query": "Analyze this position with Stockfish.",
                "fen": "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",
                "depth": 12,
            },
            "confidence": 0.99,
            "reasoning": "FEN and depth were provided.",
        }
    )
    driver = FakeDriver(route_text=tool_call_text, synthesis_text="Stockfish completed.")
    backend = CockpitBackend(
        artifact_root=tmp_path,
        driver_factory=lambda _provider: driver,
        specialist_specs=[spec],
    )

    run = backend.submit_chat(
        prompt=(
            "Analyze this chess FEN at depth 12: "
            "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2"
        ),
        driver_provider="fake",
        run_async=False,
    )

    specialist = stockfish_holder["specialist"]
    assert specialist.seen_payloads[0]["depth"] == 12
    assert [call["kind"] for call in run["model_calls"]] == ["tool_call", "synthesis"]
    assert run["model_calls"][0]["raw_text"] == tool_call_text
    actual = run["proposed_payload"]["actual_specialist_request"]
    assert actual["input_payload"]["depth"] == 12
    assert actual["input_payload"]["fen"] == specialist.seen_payloads[0]["fen"]
    event_types = [event["event_type"] for event in run["provenance_events"]]
    assert "LLMToolCallProposed" in event_types
    assert "LLMRoutingDecided" not in event_types
    artifact = backend.get_artifact_content(str(run["artifacts"][0]["artifact_id"]))
    assert "go depth 12" in str(artifact["text"])


def test_terraform_mutation_is_refused_before_specialist_call(tmp_path: Path) -> None:
    terraform = CountingTerraformSpecialist()
    spec = SpecialistSpec(
        SpecialistCard(
            "terraform",
            {"infrastructure_plan"},
            {"trust_anchor_id": "terraform-cli"},
            cost_hint=0.5,
            latency_hint=0.8,
            version="terraform-plan-only",
            effective_trust=0.95,
            description="Terraform plan-only.",
        ),
        lambda _store: terraform,
        lambda: Availability(True, "available", "test terraform"),
        "json",
    )
    driver = FakeDriver(
        route_text='{"tool":"terraform","confidence":0.99,"reasoning":"terraform mutation"}',
        synthesis_text="Terraform mutation was refused.",
    )
    backend = CockpitBackend(
        artifact_root=tmp_path,
        driver_factory=lambda _provider: driver,
        specialist_specs=[spec],
    )

    run = backend.submit_chat(
        prompt="Apply this Terraform module now.",
        driver_provider="fake",
        payload_override={
            "task_type": "infrastructure_plan",
            "operation": "apply",
            "destructive": True,
            "risk_tags": ["terraform_mutation"],
        },
        run_async=False,
    )

    assert run["status"] == "completed"
    assert run["selected_specialist"] == "terraform"
    assert terraform.calls == 0
    assert run["artifacts"] == []
    assert run["validation_result"]["ok"] is False
    assert "terraform operation is not allowed" in run["validation_result"]["reason"]
    assert any(
        step["label"] == "Validating tool call"
        and step["status"] == "failed"
        and "terraform operation is not allowed" in step["detail"]
        for step in run["steps"]
    )
    event_types = [event["event_type"] for event in run["provenance_events"]]
    assert "LLMToolCallProposed" in event_types
    assert "ChainStepRefusedByEthics" not in event_types
    assert "ArtifactProduced" not in event_types
    assert run["final_answer"] == "Terraform mutation was refused."


def test_empty_final_synthesis_marks_run_failed(tmp_path: Path) -> None:
    driver = FakeDriver(synthesis_text="")
    backend = CockpitBackend(
        artifact_root=tmp_path,
        driver_factory=lambda _provider: driver,
    )

    run = backend.submit_chat(
        prompt="Simplify expression: x + x",
        driver_provider="fake",
        payload_override={
            "task_type": "symbolic_math",
            "input_payload": {"operation": "simplify", "expression": "x + x"},
        },
        run_async=False,
    )

    assert run["status"] == "failed"
    assert "empty text" in run["final_answer"]
    assert any(step["label"] == "Run failed" for step in run["steps"])
    assert not any(
        step["label"] == "Run completed" and step["status"] == "completed" for step in run["steps"]
    )


class MemorySocket:
    def __init__(self, request: bytes) -> None:
        self._reader = BytesIO(request)
        self.writer = BytesIO()

    def makefile(self, mode: str, buffering: int | None = None) -> BytesIO:
        del buffering
        if "r" in mode:
            return self._reader
        return self.writer

    def sendall(self, data: bytes) -> None:
        self.writer.write(data)


def test_http_handler_health_smoke(tmp_path: Path) -> None:
    backend = CockpitBackend(
        artifact_root=tmp_path,
        driver_factory=lambda _provider: FakeDriver(),
    )
    request = b"GET /api/health HTTP/1.1\r\nHost: localhost\r\n\r\n"
    socket = MemorySocket(request)
    server = SimpleNamespace(backend=backend)

    CockpitRequestHandler(socket, ("127.0.0.1", 1), server)  # type: ignore[arg-type]

    raw = socket.writer.getvalue()
    assert b"200" in raw.split(b"\r\n", 1)[0]
    body = raw.split(b"\r\n\r\n", 1)[1]
    payload: dict[str, Any] = json.loads(body.decode())
    assert payload["ok"] is True
    assert payload["service"] == "vecl-ui-cockpit"
