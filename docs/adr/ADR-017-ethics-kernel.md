# ADR-017: Deterministic EthicsKernel For Action Authorization

## Context

ADR-011 makes an immutable ethics layer a launch blocker before side-effecting
specialists. Gemma and future classifiers can help identify risk, but VECL-QB
also needs deterministic authorization for tool calls, artifacts, tenant access,
and learning updates.

## Decision

Implement `EthicsKernel` as a pure-rule action gate, not as an LLM judge. Rules
evaluate structured `EthicsContext` and return `ALLOW`, `REFUSE`, or
`REQUIRE_HUMAN_REVIEW`. The kernel reduces decisions conservatively:
`REFUSE > REQUIRE_HUMAN_REVIEW > ALLOW`.

Rules are versioned and hashed. Install/revoke is governance-gated by an
env-supplied admin token in Phase 6, with `ETHICS_RULE_INSTALLED` and
`ETHICS_RULE_REVOKED` events recording the change without storing the raw token.

The chain executor invokes the kernel before each chain step. Refusal emits
`CHAIN_STEP_REFUSED_BY_ETHICS`; review holds emit `CHAIN_STEP_AWAITING_REVIEW`.
Both halt the chain before the specialist runs and chain to `CHAIN_ABORTED`.

Phase 6 rules enforce VECL-native invariants: provenance required, registered
specialists only, no cross-tenant access, no LoRA/training-memory writes without
a valid `LearningEventToken`, no cross-tenant learning, destructive side effects
require scoped approval, and structured risk tags may refuse or require review.

## Consequences

- Ethics decisions are deterministic, unit-testable, and provenance-visible.
- The model can propose plans, but the system authorizes actions.
- Phase 6 enforces risk tags; it does not infer them from raw natural language.
- Phase 6 trusts planner-supplied risk tags. Later side-effecting specialists
  must derive risk tags from specialist id, operation, and typed arguments so
  risk classification is not solely model-controlled.
- The Phase 6b Gemma 4 31B eval validated all three outcomes: refusal, human
  review, and allow. It also validated the destructive-action approval override.
- The first attempted review fixture, opening a production firewall, was tagged
  by Gemma as `cyber_abuse` and refused. This empirically confirmed the
  conservative reduction order when refusal and review rules both apply.
- Episodic/provenance writes remain audit memory. LoRA/training writes require
  learning tokens per ADR-004.

## Rejected Alternatives

- LLM-as-judge for ethics decisions — too prompt-sensitive and unauditable.
- Full frontier-model safety policy clone — too broad for Phase 6; domain rule
  packs should arrive with specialists.
- Hard-ban all future side effects — safe, but incompatible with governed
  infrastructure workflows.
