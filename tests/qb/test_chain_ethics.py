from __future__ import annotations

import pytest

from vecl.ethics import EthicsKernel, default_ethics_rules
from vecl.provenance.events import EventType, ProvenanceEvent
from vecl.provenance.ledger import ProvenanceLedger
from vecl.qb.chain_executor import ChainExecutor
from vecl.qb.planner import ChainPlan, ChainStep
from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse


class CountingSpecialist(Specialist):
    def __init__(self, specialist_id: str = "counter") -> None:
        self.specialist_id = specialist_id
        self.calls = 0

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type == "ethics_test"

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        self.calls += 1
        claim = SpecialistClaim(
            "claim-counter",
            self.specialist_id,
            "counter ran",
            "counter",
            0.9,
            evidence_ids=["fixture"],
        )
        return SpecialistResponse(
            request.request_id, self.specialist_id, [claim], request.tenant_id
        )


def _kernel(monkeypatch: pytest.MonkeyPatch, ledger: ProvenanceLedger) -> EthicsKernel:
    monkeypatch.setenv("VECL_ETHICS_ADMIN_TOKEN", "admin")
    kernel = EthicsKernel(ledger=ledger)
    for rule in default_ethics_rules():
        kernel.install_rule(rule, "admin")
    return kernel


def _request(
    ledger: ProvenanceLedger, input_payload: dict[str, object] | None = None
) -> SpecialistRequest:
    parent = ledger.append(
        ProvenanceEvent(EventType.EVIDENCE_INGESTED, "tenant-a", "test", {"request_id": "req"})
    )
    return SpecialistRequest(
        "req",
        "tenant-a",
        "ethics_test",
        input_payload or {},
        {},
        {"parent_event_id": parent.event_id},
    )


def _plan(parameters: dict[str, object] | None = None) -> ChainPlan:
    return ChainPlan(
        "ethics-plan",
        "ethics_test",
        (ChainStep("step-1", "counter", parameters=parameters or {}),),
    )


def _terraform_plan(parameters: dict[str, object] | None = None) -> ChainPlan:
    return ChainPlan(
        "terraform-ethics-plan",
        "ethics_test",
        (ChainStep("step-1", "terraform", parameters=parameters or {}),),
    )


def test_chain_step_refused_by_ethics_never_invokes_specialist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = ProvenanceLedger()
    kernel = _kernel(monkeypatch, ledger)
    specialist = CountingSpecialist()
    executor = ChainExecutor(
        ledger=ledger,
        specialists={specialist.specialist_id: specialist},
        ethics_kernel=kernel,
    )

    result = executor.execute(_plan({"target_tenant_id": "tenant-b"}), _request(ledger))

    refused = ledger.find_by_type(EventType.CHAIN_STEP_REFUSED_BY_ETHICS)
    aborted = ledger.find_by_type(EventType.CHAIN_ABORTED)
    assert result.aborted
    assert specialist.calls == 0
    assert refused[0].payload["rule_id"] == "TENANT_NO_CROSS_ACCESS"
    assert aborted[0].parent_event_ids == [refused[0].event_id]
    assert ledger.find_by_type(EventType.CHAIN_STEP_FAILED) == []


def test_chain_step_awaiting_review_never_invokes_specialist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = ProvenanceLedger()
    kernel = _kernel(monkeypatch, ledger)
    specialist = CountingSpecialist()
    executor = ChainExecutor(
        ledger=ledger,
        specialists={specialist.specialist_id: specialist},
        ethics_kernel=kernel,
    )

    result = executor.execute(_plan({"destructive": True}), _request(ledger))

    awaiting = ledger.find_by_type(EventType.CHAIN_STEP_AWAITING_REVIEW)
    assert result.aborted
    assert specialist.calls == 0
    assert awaiting[0].payload["rule_id"] == "DESTRUCTIVE_ACTION_REQUIRES_APPROVAL"
    assert result.failed_step_id == "step-1"


def test_chain_step_allowed_by_ethics_invokes_specialist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = ProvenanceLedger()
    kernel = _kernel(monkeypatch, ledger)
    specialist = CountingSpecialist()
    executor = ChainExecutor(
        ledger=ledger,
        specialists={specialist.specialist_id: specialist},
        ethics_kernel=kernel,
    )

    result = executor.execute(_plan(), _request(ledger))

    assert not result.aborted
    assert specialist.calls == 1
    assert ledger.find_by_type(EventType.CHAIN_STEP_COMPLETED)
    assert ledger.find_by_type(EventType.CHAIN_STEP_REFUSED_BY_ETHICS) == []


def test_terraform_apply_is_refused_before_specialist_even_with_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = ProvenanceLedger()
    kernel = _kernel(monkeypatch, ledger)
    specialist = CountingSpecialist("terraform")
    executor = ChainExecutor(
        ledger=ledger,
        specialists={specialist.specialist_id: specialist},
        ethics_kernel=kernel,
    )

    result = executor.execute(
        _terraform_plan(
            {
                "operation": "apply",
                "destructive": True,
                "governance_approval_id": "GOV-123",
            }
        ),
        _request(ledger),
    )

    refused = ledger.find_by_type(EventType.CHAIN_STEP_REFUSED_BY_ETHICS)
    assert result.aborted
    assert specialist.calls == 0
    assert refused[0].payload["rule_id"] == "TERRAFORM_MUTATION_DISABLED"
