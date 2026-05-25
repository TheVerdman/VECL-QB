# ADR-020: Terraform Plan-Only Specialist

## Context

Phase 8 introduces Terraform as the first real side-effect-capable specialist.
`terraform apply`, `destroy`, and state mutation can change external cloud or
data-center infrastructure, spend money, or delete services. The EthicsKernel
exists, and Phase 10a made plan artifacts durable and hash-verifiable.

## Decision

Implement Terraform as a plan-only specialist for Phase 8:

- `TerraformSpecialist` supports `validate` and `plan`.
- Plan output is stored as a JSON artifact in the content-addressed store.
- `apply`, `destroy`, `state`, `import`, workspace mutation, taint, untaint,
  and force-unlock are refused by the specialist before any Terraform binary is
  resolved.
- `default_ethics_rules()` also installs `TERRAFORM_MUTATION_DISABLED`, which
  refuses the same mutation commands for specialist id `terraform`, even when a
  governance approval id is present.
- A future governed-apply phase must replace this rule explicitly and bind any
  approval to a specific reviewed plan hash.

## Consequences

- Phase 8 can validate real Terraform planning without granting autonomous
  infrastructure mutation.
- Credentials are not enough to make mutation safe; approval must be scoped to
  the action and, later, the exact plan.
- The 31B eval should verify both positive routing to Terraform for plan-like
  requests and no specialist-call leak for mutation requests.

## Rejected Alternatives

- Allow `apply` with credentials in Phase 8 — too early without plan-hash scoped
  approval, rollback, and dual-control semantics.
- Rely only on prompt/risk tags — insufficient because mutation safety must not
  depend on the model self-labeling the action.
- Mock Terraform — rejected for Phase 8 validation; tests may skip locally when
  no binary is available, but the acceptance run uses the real CLI.
