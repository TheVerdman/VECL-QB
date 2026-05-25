from __future__ import annotations

import json
from pathlib import Path

from scripts.terraform_eval_gemma import (
    build_answer_prompt,
    parse_terraform_answer,
    run_terraform_eval,
    score_terraform_answer,
    stockfish_card,
    terraform_card,
    terraform_chain_plan,
)
from vecl.provenance.events import stable_hash
from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.specialists.artifacts import ContentAddressedStore


class FakeTerraformSpecialist(Specialist):
    specialist_id = "terraform"

    def __init__(self, store: ContentAddressedStore) -> None:
        self.store = store

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "infrastructure_plan"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        operation = str(request.input_payload.get("operation") or "plan")
        if operation == "validate":
            payload = {"format_version": "1.2", "terraform_version": "1.15.4", "valid": True}
            claim_type = "terraform_validation"
        else:
            payload = {
                "format_version": "1.2",
                "terraform_version": "1.15.4",
                "resource_changes": [
                    {
                        "address": "terraform_data.inventory",
                        "type": "terraform_data",
                        "name": "inventory",
                        "change": {"actions": ["create"]},
                    }
                ],
            }
            claim_type = "terraform_plan"
        record = self.store.write_text(
            json.dumps(payload, sort_keys=True),
            producer_specialist_id=self.specialist_id,
            producer_version="terraform-test",
            input_hash=stable_hash(request.input_payload),
            output_format="json",
            parent_event_id=str(request.provenance_context["parent_event_id"]),
        )
        claim = SpecialistClaim(
            "claim-" + stable_hash(payload)[:12],
            self.specialist_id,
            "terraform_result=" + json.dumps(payload, sort_keys=True),
            claim_type,
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


class FakeStockfishSpecialist(Specialist):
    specialist_id = "stockfish"

    def __init__(self, store: ContentAddressedStore) -> None:
        self.store = store

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "chess_eval"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        transcript = "bestmove g8f6\n"
        record = self.store.write_text(
            transcript,
            producer_specialist_id=self.specialist_id,
            producer_version="stockfish-test",
            input_hash=stable_hash(request.input_payload),
            output_format="txt",
            parent_event_id=str(request.provenance_context["parent_event_id"]),
        )
        claim = SpecialistClaim(
            "claim-stockfish-test",
            self.specialist_id,
            "bestmove=g8f6; eval_cp=20; pv=g8f6",
            "chess_eval",
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


def test_terraform_chain_plan_and_card_shape() -> None:
    plan = terraform_chain_plan()
    card = terraform_card()

    assert plan.steps[0].specialist_id == "terraform"
    assert card.specialist_id == "terraform"
    assert stockfish_card().specialist_id == "stockfish"
    assert "apply and destroy are disabled" in card.description


def test_parse_and_score_terraform_answer() -> None:
    answer = parse_terraform_answer(
        """
        ```json
        {"operation":"plan","used_terraform":true,"applied":false,"valid":null,
         "change_summary":{"create":1,"update":0,"delete":0,"replace":0,"read":0,"no_op":0},
         "answer":"One resource would be created."}
        ```
        """
    )
    context = {
        "operation": "plan",
        "expected_kind": "plan",
        "expected_change_summary": {"create": 1},
    }

    score = score_terraform_answer(context, answer)

    assert score["passed"] is True


def test_build_answer_prompt_includes_plan_only_instruction() -> None:
    prompt = build_answer_prompt(
        {
            "question": "What would change?",
            "artifact_json": {"resource_changes": []},
        }
    )

    assert "plan-only" in prompt
    assert "Do not claim that Terraform applied changes" in prompt


def test_run_terraform_eval_with_mock_inference_blocks_mutations(tmp_path: Path) -> None:
    entries = json.loads(
        (Path(__file__).parents[1] / "qb" / "fixtures" / "terraform_eval_v0.json").read_text()
    )
    store = ContentAddressedStore(tmp_path / "artifacts")
    fake = FakeTerraformSpecialist(store)

    def route_fn(prompt: str) -> str:
        if "best move" in prompt:
            return '{"tool": null, "confidence": 0.9, "reasoning": "not infrastructure"}'
        return '{"tool": "terraform", "confidence": 0.95, "reasoning": "Terraform request"}'

    def answer_fn(prompt: str) -> str:
        if '"valid": true' in prompt:
            return json.dumps(
                {
                    "operation": "validate",
                    "used_terraform": True,
                    "applied": False,
                    "valid": True,
                    "change_summary": {
                        "create": 0,
                        "update": 0,
                        "delete": 0,
                        "replace": 0,
                        "read": 0,
                        "no_op": 0,
                    },
                    "answer": "The module is valid.",
                }
            )
        return json.dumps(
            {
                "operation": "plan",
                "used_terraform": True,
                "applied": False,
                "valid": None,
                "change_summary": {
                    "create": 1,
                    "update": 0,
                    "delete": 0,
                    "replace": 0,
                    "read": 0,
                    "no_op": 0,
                },
                "answer": "One terraform_data resource would be created.",
            }
        )

    summary = run_terraform_eval(
        entries=entries,
        route_fn=route_fn,
        answer_fn=answer_fn,
        model_id="mock-gemma",
        artifact_root=tmp_path,
        terraform_specialist=fake,
    )

    assert summary["plan_chain_success_rate"] == 1.0
    assert summary["answer_success_rate"] == 1.0
    assert summary["mutation_safe_rate"] == 1.0
    assert summary["blocked_specialist_calls"] == 0
    assert summary["out_of_domain_terraform_rate"] == 0.0
    assert summary["decision_events"] == len(entries)


def test_run_terraform_eval_with_stockfish_and_sqlite_persistence(tmp_path: Path) -> None:
    entries = json.loads(
        (
            Path(__file__).parents[1] / "qb" / "fixtures" / "terraform_stockfish_eval_v0.json"
        ).read_text()
    )
    store = ContentAddressedStore(tmp_path / "artifacts")
    fake_terraform = FakeTerraformSpecialist(store)
    fake_stockfish = FakeStockfishSpecialist(store)

    def route_fn(prompt: str) -> str:
        if "best move" in prompt or "chess engine" in prompt:
            return '{"tool": "stockfish", "confidence": 0.96, "reasoning": "chess request"}'
        if "Will it rain" in prompt:
            return '{"tool": null, "confidence": 0.9, "reasoning": "no weather specialist"}'
        return '{"tool": "terraform", "confidence": 0.95, "reasoning": "Terraform request"}'

    def answer_fn(prompt: str) -> str:
        if '"valid": true' in prompt:
            return json.dumps(
                {
                    "operation": "validate",
                    "used_terraform": True,
                    "applied": False,
                    "valid": True,
                    "change_summary": {
                        "create": 0,
                        "update": 0,
                        "delete": 0,
                        "replace": 0,
                        "read": 0,
                        "no_op": 0,
                    },
                    "answer": "The module is valid.",
                }
            )
        return json.dumps(
            {
                "operation": "plan",
                "used_terraform": True,
                "applied": False,
                "valid": None,
                "change_summary": {
                    "create": 1,
                    "update": 0,
                    "delete": 0,
                    "replace": 0,
                    "read": 0,
                    "no_op": 0,
                },
                "answer": "One terraform_data resource would be created.",
            }
        )

    summary = run_terraform_eval(
        entries=entries,
        route_fn=route_fn,
        answer_fn=answer_fn,
        model_id="mock-gemma",
        artifact_root=tmp_path,
        terraform_specialist=fake_terraform,
        stockfish_specialist=fake_stockfish,
        ledger_path=tmp_path / "ledger.sqlite",
    )

    assert summary["plan_chain_success_rate"] == 1.0
    assert summary["answer_success_rate"] == 1.0
    assert summary["mutation_safe_rate"] == 1.0
    assert summary["stockfish_routing_success_rate"] == 1.0
    assert summary["blocked_specialist_calls"] == 0
    assert summary["out_of_domain_terraform_rate"] == 0.0
    assert summary["persisted_artifact_restore_count"] == summary["artifact_events"]
