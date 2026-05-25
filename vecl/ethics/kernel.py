from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from vecl._compat import StrEnum
from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import ProvenanceLedger
from vecl.runtime.tokens import LearningEventToken, validate_learning_event_token

DEFAULT_REFUSE_RISK_TAGS = frozenset(
    {
        "csam",
        "credential_theft",
        "cyber_abuse",
        "dangerous_chemical_synthesis",
        "dangerous_weaponization",
        "disabled_operation",
        "malware",
        "privacy_abuse",
        "secret_exfiltration",
        "self_harm_instruction",
        "terraform_mutation",
    }
)
DEFAULT_REVIEW_RISK_TAGS = frozenset(
    {
        "bioengineering_misuse",
        "external_side_effect",
        "high_impact_professional_side_effect",
        "political_manipulation",
    }
)


class EthicsDecisionType(StrEnum):
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"
    REQUIRE_HUMAN_REVIEW = "REQUIRE_HUMAN_REVIEW"


@dataclass(frozen=True)
class EthicsDecision:
    rule_id: str
    decision: EthicsDecisionType
    reason: str

    def to_payload(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "decision": self.decision.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EthicsEvaluation:
    final_decision: EthicsDecision
    rule_decisions: tuple[EthicsDecision, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "final_decision": self.final_decision.to_payload(),
            "rule_decisions": [decision.to_payload() for decision in self.rule_decisions],
        }


@dataclass(frozen=True)
class EthicsContext:
    tenant_id: str
    actor_id: str
    request_id: str
    action_type: str
    provenance_parent_event_id: str
    chain_id: str | None = None
    step_id: str | None = None
    specialist_id: str | None = None
    operation: str | None = None
    target_tenant_id: str | None = None
    cross_tenant_grant_id: str | None = None
    writes_learning_memory: bool = False
    learning_event_token: LearningEventToken | None = None
    governance_approval_id: str | None = None
    destructive: bool = False
    risk_tags: frozenset[str] = field(default_factory=frozenset)
    specialist_registered: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "risk_tags", frozenset(_normalize_tag(tag) for tag in self.risk_tags)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_payload(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "request_id": self.request_id,
            "action_type": self.action_type,
            "provenance_parent_event_id": self.provenance_parent_event_id,
            "chain_id": self.chain_id,
            "step_id": self.step_id,
            "specialist_id": self.specialist_id,
            "operation": self.operation,
            "target_tenant_id": self.target_tenant_id,
            "cross_tenant_grant_id": self.cross_tenant_grant_id,
            "writes_learning_memory": self.writes_learning_memory,
            "learning_event_token_id": (
                self.learning_event_token.event_id if self.learning_event_token else None
            ),
            "governance_approval_id": self.governance_approval_id,
            "destructive": self.destructive,
            "risk_tags": sorted(self.risk_tags),
            "specialist_registered": self.specialist_registered,
            "metadata": dict(self.metadata),
        }

    def context_hash(self) -> str:
        return stable_hash(self.to_payload())


RuleFunction = Callable[[EthicsContext], EthicsDecision]


@dataclass(frozen=True)
class EthicsRule:
    rule_id: str
    version: str
    description: str
    evaluate: RuleFunction

    def __post_init__(self) -> None:
        if not self.rule_id or not self.version or not self.description:
            raise ValueError("ethics rule id, version, and description must be non-empty")

    @property
    def rule_hash(self) -> str:
        return stable_hash(
            {
                "rule_id": self.rule_id,
                "version": self.version,
                "description": self.description,
            }
        )

    def decide(self, context: EthicsContext) -> EthicsDecision:
        decision = self.evaluate(context)
        if decision.rule_id != self.rule_id:
            raise ValueError("ethics rule returned a mismatched rule_id")
        return decision


class EthicsKernel:
    def __init__(
        self,
        *,
        ledger: ProvenanceLedger,
        governance_tenant_id: str = "system",
        actor: str = "ethics-kernel",
        admin_token_env: str = "VECL_ETHICS_ADMIN_TOKEN",
    ) -> None:
        if not governance_tenant_id:
            raise ValueError("governance_tenant_id must be non-empty")
        self.ledger = ledger
        self.governance_tenant_id = governance_tenant_id
        self.actor = actor
        self.admin_token_env = admin_token_env
        self._rules: dict[str, EthicsRule] = {}
        self._revoked_rule_ids: set[str] = set()
        # Rule ids are single-use: replacements must use a new id/version path.
        self._seen_rule_ids: set[str] = set()

    @property
    def active_rule_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._rules))

    def install_rule(self, rule: EthicsRule, governance_signature: str) -> ProvenanceEvent:
        self._require_valid_signature(governance_signature)
        if rule.rule_id in self._seen_rule_ids:
            raise ValueError(f"ethics rule id already used: {rule.rule_id}")
        self._seen_rule_ids.add(rule.rule_id)
        self._rules[rule.rule_id] = rule
        return self.ledger.append(
            ProvenanceEvent(
                EventType.ETHICS_RULE_INSTALLED,
                self.governance_tenant_id,
                self.actor,
                {
                    "rule_id": rule.rule_id,
                    "version": rule.version,
                    "description": rule.description,
                    "rule_hash": rule.rule_hash,
                    "governance_signature_hash": stable_hash(governance_signature),
                },
            )
        )

    def revoke_rule(
        self, rule_id: str, governance_signature: str, *, reason: str
    ) -> ProvenanceEvent:
        self._require_valid_signature(governance_signature)
        if rule_id not in self._rules:
            raise KeyError(rule_id)
        rule = self._rules.pop(rule_id)
        self._revoked_rule_ids.add(rule_id)
        return self.ledger.append(
            ProvenanceEvent(
                EventType.ETHICS_RULE_REVOKED,
                self.governance_tenant_id,
                self.actor,
                {
                    "rule_id": rule.rule_id,
                    "version": rule.version,
                    "rule_hash": rule.rule_hash,
                    "reason": reason,
                    "governance_signature_hash": stable_hash(governance_signature),
                },
            )
        )

    def evaluate(self, context: EthicsContext) -> EthicsEvaluation:
        decisions = tuple(self._rules[rule_id].decide(context) for rule_id in sorted(self._rules))
        if not decisions:
            decisions = (
                EthicsDecision(
                    "NO_RULES_INSTALLED",
                    EthicsDecisionType.REFUSE,
                    "ethics kernel has no active rules",
                ),
            )
        final = sorted(
            decisions,
            key=lambda decision: (-_decision_priority(decision.decision), decision.rule_id),
        )[0]
        return EthicsEvaluation(final_decision=final, rule_decisions=decisions)

    def _require_valid_signature(self, governance_signature: str) -> None:
        expected = os.environ.get(self.admin_token_env)
        if not expected or governance_signature != expected:
            raise PermissionError("invalid ethics governance signature")


def default_ethics_rules() -> tuple[EthicsRule, ...]:
    return (
        EthicsRule(
            "PROVENANCE_REQUIRED",
            "v1",
            "Actions require request, tenant, actor, and provenance parent identifiers.",
            _provenance_required,
        ),
        EthicsRule(
            "SPECIALIST_MUST_BE_REGISTERED",
            "v1",
            "Specialist calls require a registered specialist.",
            _specialist_must_be_registered,
        ),
        EthicsRule(
            "TENANT_NO_CROSS_ACCESS",
            "v1",
            "Cross-tenant action targets require an explicit cross-tenant grant.",
            _tenant_no_cross_access,
        ),
        EthicsRule(
            "NO_LORA_WRITE_WITHOUT_LEARNING_TOKEN",
            "v1",
            "Learning-memory writes require a valid LearningEventToken.",
            _no_lora_write_without_learning_token,
        ),
        EthicsRule(
            "NO_CROSS_TENANT_LEARNING",
            "v1",
            "Learning-memory writes may not cross tenant boundaries without a grant.",
            _no_cross_tenant_learning,
        ),
        EthicsRule(
            "DESTRUCTIVE_ACTION_REQUIRES_APPROVAL",
            "v1",
            "External side effects require an explicit governance approval id.",
            _destructive_action_requires_approval,
        ),
        EthicsRule(
            "TERRAFORM_MUTATION_DISABLED",
            "v1",
            "Terraform apply, destroy, and state mutation are disabled in Phase 8.",
            _terraform_mutation_disabled,
        ),
        EthicsRule(
            "RISK_TAG_REFUSAL",
            "v1",
            "Refuse actions carrying non-negotiable risk tags.",
            _risk_tag_refusal,
        ),
        EthicsRule(
            "RISK_TAG_REVIEW",
            "v1",
            "Require human review for ambiguous or high-stakes risk tags.",
            _risk_tag_review,
        ),
    )


def _allow(rule_id: str) -> EthicsDecision:
    return EthicsDecision(rule_id, EthicsDecisionType.ALLOW, "allowed")


def _refuse(rule_id: str, reason: str) -> EthicsDecision:
    return EthicsDecision(rule_id, EthicsDecisionType.REFUSE, reason)


def _review(rule_id: str, reason: str) -> EthicsDecision:
    return EthicsDecision(rule_id, EthicsDecisionType.REQUIRE_HUMAN_REVIEW, reason)


def _provenance_required(context: EthicsContext) -> EthicsDecision:
    missing = [
        name
        for name, value in {
            "tenant_id": context.tenant_id,
            "actor_id": context.actor_id,
            "request_id": context.request_id,
            "provenance_parent_event_id": context.provenance_parent_event_id,
        }.items()
        if not value
    ]
    if missing:
        return _refuse("PROVENANCE_REQUIRED", f"missing provenance fields: {missing}")
    return _allow("PROVENANCE_REQUIRED")


def _specialist_must_be_registered(context: EthicsContext) -> EthicsDecision:
    if context.action_type == "specialist_call" and not context.specialist_registered:
        return _refuse(
            "SPECIALIST_MUST_BE_REGISTERED",
            f"unregistered specialist_id: {context.specialist_id}",
        )
    return _allow("SPECIALIST_MUST_BE_REGISTERED")


def _tenant_no_cross_access(context: EthicsContext) -> EthicsDecision:
    if (
        context.target_tenant_id
        and context.target_tenant_id != context.tenant_id
        and not context.cross_tenant_grant_id
    ):
        return _refuse(
            "TENANT_NO_CROSS_ACCESS",
            f"target tenant {context.target_tenant_id} differs from {context.tenant_id}",
        )
    return _allow("TENANT_NO_CROSS_ACCESS")


def _no_lora_write_without_learning_token(context: EthicsContext) -> EthicsDecision:
    if not context.writes_learning_memory:
        return _allow("NO_LORA_WRITE_WITHOUT_LEARNING_TOKEN")
    if context.learning_event_token is None:
        return _refuse(
            "NO_LORA_WRITE_WITHOUT_LEARNING_TOKEN",
            "learning-memory write missing LearningEventToken",
        )
    try:
        validate_learning_event_token(context.learning_event_token, None, context.tenant_id)
    except ValueError as exc:
        return _refuse("NO_LORA_WRITE_WITHOUT_LEARNING_TOKEN", str(exc))
    return _allow("NO_LORA_WRITE_WITHOUT_LEARNING_TOKEN")


def _no_cross_tenant_learning(context: EthicsContext) -> EthicsDecision:
    if (
        context.writes_learning_memory
        and context.target_tenant_id
        and context.target_tenant_id != context.tenant_id
        and not context.cross_tenant_grant_id
    ):
        return _refuse(
            "NO_CROSS_TENANT_LEARNING",
            "learning-memory write crosses tenant boundary",
        )
    return _allow("NO_CROSS_TENANT_LEARNING")


def _destructive_action_requires_approval(context: EthicsContext) -> EthicsDecision:
    if context.destructive and not context.governance_approval_id:
        return _review(
            "DESTRUCTIVE_ACTION_REQUIRES_APPROVAL",
            "destructive action missing governance_approval_id",
        )
    return _allow("DESTRUCTIVE_ACTION_REQUIRES_APPROVAL")


def _terraform_mutation_disabled(context: EthicsContext) -> EthicsDecision:
    operation = (context.operation or "").strip().lower()
    if operation.startswith("terraform "):
        operation = operation.removeprefix("terraform ").strip()
    if operation.startswith("terraform_"):
        operation = operation.removeprefix("terraform_").strip()
    command = operation.split()[0] if operation else ""
    if context.specialist_id == "terraform" and command in {
        "apply",
        "destroy",
        "force-unlock",
        "import",
        "state",
        "taint",
        "untaint",
        "workspace",
    }:
        return _refuse(
            "TERRAFORM_MUTATION_DISABLED",
            f"Terraform mutation command disabled in Phase 8: {command}",
        )
    return _allow("TERRAFORM_MUTATION_DISABLED")


def _risk_tag_refusal(context: EthicsContext) -> EthicsDecision:
    blocked = sorted(context.risk_tags & DEFAULT_REFUSE_RISK_TAGS)
    if blocked:
        return _refuse("RISK_TAG_REFUSAL", f"refused risk tags: {blocked}")
    return _allow("RISK_TAG_REFUSAL")


def _risk_tag_review(context: EthicsContext) -> EthicsDecision:
    review = sorted(context.risk_tags & DEFAULT_REVIEW_RISK_TAGS)
    if review and not context.governance_approval_id:
        return _review("RISK_TAG_REVIEW", f"review risk tags: {review}")
    return _allow("RISK_TAG_REVIEW")


def _decision_priority(decision: EthicsDecisionType) -> int:
    return {
        EthicsDecisionType.ALLOW: 0,
        EthicsDecisionType.REQUIRE_HUMAN_REVIEW: 1,
        EthicsDecisionType.REFUSE: 2,
    }[decision]


def _normalize_tag(tag: str) -> str:
    return str(tag).strip().lower()
