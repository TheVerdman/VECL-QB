from __future__ import annotations

from datetime import datetime

import pytest

from vecl._compat import UTC
from vecl.ethics import (
    EthicsContext,
    EthicsDecisionType,
    EthicsKernel,
    default_ethics_rules,
)
from vecl.provenance.events import EventType
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.tokens import create_learning_event_token


def _kernel(monkeypatch: pytest.MonkeyPatch) -> tuple[EthicsKernel, ProvenanceLedger]:
    monkeypatch.setenv("VECL_ETHICS_ADMIN_TOKEN", "admin")
    ledger = ProvenanceLedger()
    kernel = EthicsKernel(ledger=ledger)
    for rule in default_ethics_rules():
        kernel.install_rule(rule, "admin")
    return kernel, ledger


def _context(**overrides: object) -> EthicsContext:
    data: dict[str, object] = {
        "tenant_id": "tenant-a",
        "actor_id": "chain-executor",
        "request_id": "req-1",
        "action_type": "specialist_call",
        "provenance_parent_event_id": "evt-parent",
        "specialist_id": "stockfish",
        "specialist_registered": True,
    }
    data.update(overrides)
    return EthicsContext(**data)  # type: ignore[arg-type]


def test_rule_install_requires_governance_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECL_ETHICS_ADMIN_TOKEN", "admin")
    ledger = ProvenanceLedger()
    kernel = EthicsKernel(ledger=ledger)

    with pytest.raises(PermissionError):
        kernel.install_rule(default_ethics_rules()[0], "wrong")

    assert ledger.find_by_type(EventType.ETHICS_RULE_INSTALLED) == []


def test_rule_install_and_revoke_emit_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECL_ETHICS_ADMIN_TOKEN", "admin")
    ledger = ProvenanceLedger()
    kernel = EthicsKernel(ledger=ledger)
    rule = default_ethics_rules()[0]

    installed = kernel.install_rule(rule, "admin")
    revoked = kernel.revoke_rule(rule.rule_id, "admin", reason="superseded")

    assert installed.payload["rule_id"] == "PROVENANCE_REQUIRED"
    assert revoked.payload["reason"] == "superseded"
    assert kernel.active_rule_ids == ()
    with pytest.raises(ValueError, match="already used"):
        kernel.install_rule(rule, "admin")


def test_kernel_fails_closed_when_no_rules_are_installed() -> None:
    ledger = ProvenanceLedger()
    kernel = EthicsKernel(ledger=ledger)

    evaluation = kernel.evaluate(_context())

    assert evaluation.final_decision.decision == EthicsDecisionType.REFUSE
    assert evaluation.final_decision.rule_id == "NO_RULES_INSTALLED"


def test_default_rules_allow_clean_specialist_call(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel, _ledger = _kernel(monkeypatch)

    evaluation = kernel.evaluate(_context())

    assert evaluation.final_decision.decision == EthicsDecisionType.ALLOW


@pytest.mark.parametrize(
    ("context", "rule_id"),
    [
        (_context(provenance_parent_event_id=""), "PROVENANCE_REQUIRED"),
        (_context(specialist_registered=False), "SPECIALIST_MUST_BE_REGISTERED"),
        (_context(target_tenant_id="tenant-b"), "TENANT_NO_CROSS_ACCESS"),
        (
            _context(
                writes_learning_memory=True,
                target_tenant_id="tenant-b",
                cross_tenant_grant_id="grant-1",
            ),
            "NO_LORA_WRITE_WITHOUT_LEARNING_TOKEN",
        ),
        (
            _context(
                writes_learning_memory=True,
                target_tenant_id="tenant-b",
                learning_event_token=create_learning_event_token(
                    batch_id="batch",
                    tenant_id="tenant-a",
                    source_set_hash="src",
                    provenance_root_hash="root",
                    min_authority=0.1,
                    min_score=0.1,
                    max_slots=1,
                    policy_version="policy-v1",
                    trust_policy_version="trust-v1",
                    now=datetime.now(UTC),
                ),
            ),
            "NO_CROSS_TENANT_LEARNING",
        ),
        (_context(risk_tags=frozenset({"privacy_abuse"})), "RISK_TAG_REFUSAL"),
        (
            _context(
                specialist_id="terraform",
                operation="apply",
                destructive=True,
                governance_approval_id="GOV-123",
            ),
            "TERRAFORM_MUTATION_DISABLED",
        ),
    ],
)
def test_default_rules_refuse_blocked_contexts(
    monkeypatch: pytest.MonkeyPatch, context: EthicsContext, rule_id: str
) -> None:
    kernel, _ledger = _kernel(monkeypatch)

    evaluation = kernel.evaluate(context)

    assert evaluation.final_decision.decision == EthicsDecisionType.REFUSE
    assert evaluation.final_decision.rule_id == rule_id


@pytest.mark.parametrize(
    ("context", "rule_id"),
    [
        (_context(destructive=True), "DESTRUCTIVE_ACTION_REQUIRES_APPROVAL"),
        (_context(risk_tags=frozenset({"bioengineering_misuse"})), "RISK_TAG_REVIEW"),
    ],
)
def test_default_rules_require_human_review(
    monkeypatch: pytest.MonkeyPatch, context: EthicsContext, rule_id: str
) -> None:
    kernel, _ledger = _kernel(monkeypatch)

    evaluation = kernel.evaluate(context)

    assert evaluation.final_decision.decision == EthicsDecisionType.REQUIRE_HUMAN_REVIEW
    assert evaluation.final_decision.rule_id == rule_id


def test_learning_write_with_valid_token_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel, _ledger = _kernel(monkeypatch)
    token = create_learning_event_token(
        batch_id="batch",
        tenant_id="tenant-a",
        source_set_hash="src",
        provenance_root_hash="root",
        min_authority=0.1,
        min_score=0.1,
        max_slots=1,
        policy_version="policy-v1",
        trust_policy_version="trust-v1",
        now=datetime.now(UTC),
    )

    evaluation = kernel.evaluate(_context(writes_learning_memory=True, learning_event_token=token))

    assert evaluation.final_decision.decision == EthicsDecisionType.ALLOW
