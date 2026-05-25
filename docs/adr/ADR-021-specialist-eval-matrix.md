# ADR-021: Specialist Eval Matrix

## Context

ADR-015 requires every new specialist to get a real-resource router eval. Phase
8b showed a gap in the naive "out-of-domain" framing: proving a chess request
does not route to Terraform is weaker than proving it still routes to Stockfish
when both specialists are loaded. In production, specialists compete inside one
registry, so every new tool can accidentally shadow previously validated tools.

## Decision

Every new specialist eval must run against a representative registry, not only
against the new specialist in isolation.

Each eval fixture should include:

- New-specialist positives that should route to and execute the new specialist.
- Unsupported negatives that should route to no specialist.
- Safety or ethics cases when the specialist has sensitive or side-effecting
  behavior.
- Cross-specialist preservation cases that should still route to already
  supported specialists while the new specialist is present.

The acceptance check must distinguish "not routed to the new specialist" from
"routed to the correct existing specialist." A preservation case only passes
when the expected existing specialist is selected and, where relevant, executed.

## Consequences

- Specialist acceptance becomes a routing regression matrix instead of a set of
  isolated smoke tests.
- Phase 12 specialist waves can detect tool shadowing as the registry grows.
- ReleaseGate routing evaluators should track per-specialist false positives,
  false negatives, and cross-specialist steals.
- Fixtures will be slightly larger, but they better match the production system
  where Stockfish, SymPy, BLAST, TimesFM, Terraform, and later specialists are
  available together.

## Rejected Alternatives

- Treat all non-target prompts as generic negatives — this catches over-routing
  to the new specialist but misses degradation of existing capabilities.
- Test each specialist with a single-tool registry — too unlike production and
  unable to detect specialist shadowing.
- Rely only on final answer quality — insufficient because a wrong route can
  still produce plausible text while losing artifact provenance and tool
  correctness.
