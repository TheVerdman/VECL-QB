# ADR-015: Phase Real-Resource Validation

## Context

Phase 3b proved that Gemma 4 31B can route chess requests to Stockfish. Phase 4 then added chain planning and execution, which needed a different validation path: Gemma routing plus `ChainPlanner`, `ChainExecutor`, specialist execution, artifact production, and provenance assertions in one run.

Default tests must remain deterministic, local, and cheap. But the platform thesis in ADR-011 depends on real model-mediated specialist use, so phase acceptance cannot rely only on mocked inference or rule-based routing.

## Decision

Each phase gets a dedicated opt-in real-resource validation that directly exercises the new surface introduced in that phase.

- If a phase claims Gemma-mediated behavior, the opt-in validation runs against Gemma on GPU.
- If a phase does not claim model behavior, it gets the relevant real-resource validation without forcing GPU use.
- Every new specialist gets a Phase-3b-style router eval: in-domain requests should route to it, and out-of-domain requests should not.
- When an A100 is provisioned, the phase's opt-in GPU checks should be batched into the same Vertex job: routing, chain execution, artifacts, provenance, and phase-specific real specialist tests.
- Opt-in tests stay explicitly gated by environment variables and are not run by default CI or `make all`.
- GPU validations must be bounded and non-looping; human approval is required before spending GPU budget.

## Consequences

- A routing-only test no longer closes a phase whose new work is chain execution, persistence, ethics gating, or training.
- Phase reports must distinguish default test status from opt-in real-resource status.
- Later ADR numbers shift: the vector DB choice originally expected as ADR-015 will use the next available number instead.
- Default contributors remain unblocked by GPU, cloud credentials, or large model downloads.

## Rejected Alternatives

- Run all opt-in tests in default CI — too expensive and too dependent on credentials and external services.
- Require GPU for every phase — inappropriate for phases such as persistence or local artifact integrity.
- Manual one-off smoke tests — useful during development, but not sufficient as acceptance evidence.
- Reuse Phase 3b routing evals for later chain phases — misses the behavior those phases actually add.
