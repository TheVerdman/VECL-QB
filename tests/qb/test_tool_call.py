from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.model_driver import ModelDriverResult
from vecl.qb.router import SpecialistCard
from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.qb.tool_call import (
    ToolCallProposal,
    ToolCallValidationConfig,
    ToolCallValidationError,
    build_tool_call_prompt,
    execute_tool_call,
    parse_tool_call_response,
    validate_tool_call,
)
from vecl.specialists.artifacts import ContentAddressedStore

FEN = "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2"


class RecordingSpecialist(Specialist):
    def __init__(self, specialist_id: str, task_type: str, store: ContentAddressedStore) -> None:
        self.specialist_id = specialist_id
        self.task_type = task_type
        self.store = store
        self.seen_payload: dict[str, Any] | None = None

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == self.task_type

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        self.seen_payload = dict(request.input_payload)
        record = self.store.write_text(
            json.dumps(request.input_payload, sort_keys=True),
            producer_specialist_id=self.specialist_id,
            producer_version="test",
            input_hash=stable_hash(request.input_payload),
            output_format="txt",
            parent_event_id=str(request.provenance_context["parent_event_id"]),
        )
        claim = SpecialistClaim(
            f"claim-{self.specialist_id}",
            self.specialist_id,
            f"{self.specialist_id} saw payload",
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


def test_parse_tool_call_response_accepts_json_fenced_and_function_shapes() -> None:
    plain = parse_tool_call_response(
        json.dumps(
            {
                "specialist_id": "stockfish",
                "task_type": "chess_eval",
                "input_payload": {"fen": FEN, "depth": 12},
                "confidence": 0.93,
            }
        )
    )
    fenced = parse_tool_call_response(
        "```json\n"
        + json.dumps(
            {
                "tool": "sympy",
                "task_type": "symbolic_math",
                "arguments": {"operation": "simplify", "expression": "(x + 1)^2"},
            }
        )
        + "\n```"
    )
    function_call = parse_tool_call_response(
        f'stockfish({{"task_type":"chess_eval","fen":"{FEN}","depth":12}})'
    )

    assert plain.specialist_id == "stockfish"
    assert plain.input_payload["depth"] == 12
    assert fenced.specialist_id == "sympy"
    assert fenced.input_payload["expression"] == "(x + 1)^2"
    assert function_call.specialist_id == "stockfish"
    assert function_call.input_payload["fen"] == FEN


def test_validate_stockfish_payload_preserves_requested_depth() -> None:
    validated = validate_tool_call(
        ToolCallProposal("stockfish", "chess_eval", {"fen": FEN, "depth": 12}, 0.9, "", "{}"),
        cards={"stockfish": _card("stockfish", "chess_eval")},
        config=_config(),
    )

    assert validated.request.input_payload["fen"] == FEN
    assert validated.request.input_payload["depth"] == 12
    assert validated.plan.steps[0].specialist_id == "stockfish"
    assert validated.plan.steps[0].expected_artifact_type == "txt"


def test_tool_call_prompt_includes_payload_contracts() -> None:
    prompt = build_tool_call_prompt(
        SpecialistRequest("req", "tenant", "unknown", {"query": "simplify (x + 1)^2"}, {}, {}),
        [
            _card("sympy", "symbolic_math"),
            _card("stockfish", "chess_eval"),
            _card("blast", "sequence_alignment"),
            _card("terraform", "infrastructure_plan"),
            _card("timesfm", "demand_forecast"),
        ],
    )

    assert '"expression":"exact symbolic expression from the user request"' in prompt
    assert '"depth":"positive integer search depth when the user requests one"' in prompt
    assert '"database":"local BLAST database path or name supplied by the caller/runtime"' in prompt
    assert '"values_or_series":"values, history, or series of numeric observations"' in prompt
    assert '"forbidden":["apply","destroy","state mutation","workspace mutation"]' in prompt


@pytest.mark.parametrize(
    ("specialist_id", "task_type"),
    [
        ("stockfish", "chess_eval"),
        ("sympy", "symbolic_math"),
        ("blast", "sequence_alignment"),
        ("terraform", "infrastructure_plan"),
        ("timesfm", "demand_forecast"),
        ("yosys", "hardware_synthesis"),
        ("openroad", "physical_design"),
    ],
)
def test_validate_configuration_payloads_for_all_registered_specialists(
    specialist_id: str, task_type: str
) -> None:
    validated = validate_tool_call(
        ToolCallProposal(
            specialist_id,
            task_type,
            {"operation": "configure", "query": f"Give me a {specialist_id} configuration."},
            0.9,
            "",
            "{}",
        ),
        cards={specialist_id: _card(specialist_id, task_type)},
        config=_config(),
    )

    assert validated.request.input_payload["operation"] == "configure"
    assert validated.request.input_payload["configuration_template"] is True


def test_validate_blast_and_timesfm_execution_payloads() -> None:
    blast = validate_tool_call(
        ToolCallProposal(
            "blast",
            "sequence_alignment",
            {
                "database": "/tmp/blast-db/demo",
                "query_sequence": "ACGTACGTACGT",
                "max_target_seqs": 3,
                "task": "blastn-short",
            },
            0.9,
            "",
            "{}",
        ),
        cards={"blast": _card("blast", "sequence_alignment")},
        config=_config(),
    )
    timesfm = validate_tool_call(
        ToolCallProposal(
            "timesfm",
            "demand_forecast",
            {"values": [1, 2, 3, 4, 5, 6, 7, 8], "horizon": 4, "period": "week"},
            0.9,
            "",
            "{}",
        ),
        cards={"timesfm": _card("timesfm", "demand_forecast")},
        config=_config(),
    )

    assert blast.request.input_payload["database"] == "/tmp/blast-db/demo"
    assert blast.request.input_payload["max_target_seqs"] == 3
    assert timesfm.request.input_payload["values"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert timesfm.request.input_payload["horizon"] == 4


def test_validate_yosys_and_openroad_execution_payloads() -> None:
    yosys = validate_tool_call(
        ToolCallProposal(
            "yosys",
            "hardware_synthesis",
            {
                "operation": "synth",
                "top_module": "vecl_topk_helper",
                "verilog_text": "module vecl_topk_helper(); endmodule",
            },
            0.9,
            "",
            "{}",
        ),
        cards={"yosys": _card("yosys", "hardware_synthesis")},
        config=_config(),
    )
    openroad = validate_tool_call(
        ToolCallProposal(
            "openroad",
            "physical_design",
            {
                "operation": "analyze",
                "top_module": "vecl_topk_helper",
                "netlist_from": "synthesize",
            },
            0.9,
            "",
            "{}",
        ),
        cards={"openroad": _card("openroad", "physical_design")},
        config=_config(),
    )

    assert yosys.request.input_payload["operation"] == "synthesize"
    assert yosys.request.input_payload["top_module"] == "vecl_topk_helper"
    assert openroad.request.input_payload["operation"] == "analyze"
    assert openroad.request.input_payload["netlist_from"] == "synthesize"


def test_validate_stockfish_payload_clamps_depth_with_warning() -> None:
    validated = validate_tool_call(
        ToolCallProposal("stockfish", "chess_eval", {"fen": FEN, "depth": 99}, 0.9, "", "{}"),
        cards={"stockfish": _card("stockfish", "chess_eval")},
        config=_config(max_stockfish_depth=20),
    )

    assert validated.request.input_payload["depth"] == 20
    assert validated.warnings == ("stockfish depth clamped from 99 to 20",)


def test_validate_rejects_bad_fen_and_terraform_apply() -> None:
    with pytest.raises(ToolCallValidationError, match="valid six-field FEN"):
        validate_tool_call(
            ToolCallProposal("stockfish", "chess_eval", {"fen": "not a fen"}, 0.9, "", "{}"),
            cards={"stockfish": _card("stockfish", "chess_eval")},
            config=_config(),
        )

    with pytest.raises(ToolCallValidationError, match="terraform operation is not allowed"):
        validate_tool_call(
            ToolCallProposal(
                "terraform", "infrastructure_plan", {"operation": "apply"}, 0.9, "", "{}"
            ),
            cards={"terraform": _card("terraform", "infrastructure_plan")},
            config=_config(terraform_config_dir="/tmp/tf"),
        )


def test_validate_terraform_uses_runtime_config_dir() -> None:
    validated = validate_tool_call(
        ToolCallProposal(
            "terraform",
            "infrastructure_plan",
            {"operation": "plan", "config_dir": "/model/path", "variables": {"name": "demo"}},
            0.9,
            "",
            "{}",
        ),
        cards={"terraform": _card("terraform", "infrastructure_plan")},
        config=_config(terraform_config_dir="/runtime/path"),
    )

    assert validated.request.input_payload["config_dir"] == "/runtime/path"
    assert validated.request.input_payload["variables"] == {"name": "demo"}
    assert validated.warnings == (
        "ignored model-proposed Terraform config_dir in favor of runtime config",
    )


def test_execute_tool_call_runs_validated_request_through_chain_executor(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    specialist = RecordingSpecialist("stockfish", "chess_eval", store)
    ledger = ProvenanceLedger()
    parent = ledger.append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant", "test", {"request_id": "request"})
    )
    proposal = ToolCallProposal(
        "stockfish",
        "chess_eval",
        {"fen": FEN, "depth": 12, "query": "analyze this"},
        0.98,
        "FEN plus requested depth",
        '{"specialist_id":"stockfish"}',
    )

    run = execute_tool_call(
        proposal=proposal,
        model_response=ModelDriverResult(
            raw_text=proposal.raw_response,
            provider="test",
            model_id="fake-model",
        ),
        cards={"stockfish": _card("stockfish", "chess_eval")},
        specialists={"stockfish": specialist},
        ledger=ledger,
        config=_config(provenance_context={"parent_event_id": parent.event_id}),
    )

    assert run.execution is not None
    assert not run.execution.aborted
    assert specialist.seen_payload is not None
    assert specialist.seen_payload["depth"] == 12
    assert specialist.seen_payload["fen"] == FEN
    assert len(ledger.find_by_type(EventType.LLM_TOOL_CALL_PROPOSED)) == 1
    assert len(ledger.find_by_type(EventType.CHAIN_STARTED)) == 1
    assert len(ledger.find_by_type(EventType.ARTIFACT_PRODUCED)) == 1


def _card(specialist_id: str, task_type: str) -> SpecialistCard:
    return SpecialistCard(
        specialist_id,
        {task_type},
        {"description": f"{specialist_id} specialist"},
        cost_hint=1.0,
        latency_hint=1.0,
        version="test",
        effective_trust=0.9,
    )


def _config(
    *,
    provenance_context: dict[str, Any] | None = None,
    terraform_config_dir: str | None = None,
    max_stockfish_depth: int = 20,
) -> ToolCallValidationConfig:
    return ToolCallValidationConfig(
        tenant_id="tenant",
        request_id="request",
        provenance_context=provenance_context or {},
        terraform_config_dir=terraform_config_dir,
        max_stockfish_depth=max_stockfish_depth,
    )
