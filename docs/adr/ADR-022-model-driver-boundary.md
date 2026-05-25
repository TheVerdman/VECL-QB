# ADR-022: Model Driver Boundary

## Context

Gemma 4 31B has been the reference driver for routing and final-answer
synthesis. VECL-QB should remain model-agnostic: the runtime owns specialist
cards, provenance, governance, artifact storage, and deterministic scoring,
while a model driver proposes routes and writes final summaries from verified
context.

Frontier API models do not expose weights, LoRA slots, gradients, or exact
rollback. They can still drive the VECL runtime, but they must not be treated as
bounded continual-learning substrates.

## Decision

Introduce a minimal `ModelDriver` boundary with two operations:

- `route(request, specialist_cards)` returns a parsed routing decision plus the
  raw provider response envelope.
- `synthesize(prompt)` returns a raw text response envelope with model id,
  provider, usage, latency, finish reason, and metadata.

Implement Gemma, OpenAI Responses API, and Anthropic Messages API drivers behind
this boundary. The API drivers use environment-provided keys and model ids, are
opt-in, and are not used by default tests.

The runtime still validates and executes all actions. Drivers generate text;
VECL parses, scores, gates, executes specialists, and records provenance.

## Consequences

- Existing Gemma evaluations can be migrated without changing chain execution.
- Frontier drivers can be compared against the same specialist cards and eval
  fixtures without becoming learning substrates.
- Raw model outputs remain visible for human review while deterministic parsers
  decide acceptance.
- Provider-specific API details stay localized to driver classes.

## Rejected Alternatives

- Bake OpenAI or Anthropic calls directly into eval scripts — repeats the Gemma
  coupling and makes ReleaseGate harder to unify.
- Add a full provider zoo now — broader than needed for Phase 8c.
- Let drivers execute chains directly — rejected because ChainExecutor,
  EthicsKernel, and artifact stores are the authoritative runtime.
