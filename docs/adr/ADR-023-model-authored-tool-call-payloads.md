# ADR-023: Model-Authored Tool-Call Payloads

## Context

Phase 8c made model drivers interchangeable for routing and synthesis, but
route-only evaluation is not enough. A driver must also express the specialist's
input language: FEN plus depth for Stockfish, symbolic operation plus expression
for SymPy, and later bounded plan inputs for Terraform.

The UI surfaced this gap when a user requested Stockfish depth 12, but the
adapter fell back to the default depth 4. The model selected the right
specialist, yet the payload path lost the user's requested parameter.

## Decision

Add a single-step tool-call proposal layer. A `ModelDriver` may propose
`specialist_id`, `task_type`, and `input_payload` as structured JSON. VECL parses
that response, validates it against specialist-specific boundary rules, records
`LLM_TOOL_CALL_PROPOSED`, and then executes a one-step `ChainPlan` through
`ChainExecutor`.

The model authors payloads; VECL canonicalizes and gates them. Stockfish depth
is preserved up to a UI/runtime cap, SymPy operations are checked against the
supported operation set, and Terraform config paths remain runtime-controlled.
Terraform mutation commands remain rejected.

## Consequences

- We can separately measure proposal, execution, and final synthesis success.
- ChainExecutor remains the authority for specialist invocation, artifact
  events, ethics gating, and abort semantics.
- The path is realistic enough for frontier-driver A/B testing without
  introducing full LLM DAG planning.

## Rejected Alternatives

- Continue heuristic payload extraction in the UI — it hides model mistakes and
  loses explicit user parameters.
- Let models execute tools directly — bypasses ChainExecutor, EthicsKernel, and
  provenance.
- Implement full multi-step LLM planning now — useful later, but broader than
  the immediate payload-language validation gap.
