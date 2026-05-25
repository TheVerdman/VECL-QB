# ADR-018: Isolated Model Specialists

## Context

Phase 7b adds TimesFM, the first specialist that is itself a learned model.
ADR-009 keeps Gemma frozen and ADR-011 requires specialist artifacts with
provenance. A model specialist must not silently become a second learning
substrate or bypass the specialist registry.

## Decision

Treat model-backed tools as ordinary specialists with isolated runtime state.
Gemma routes to them, the specialist runs its own model, and VECL-QB records
claims, artifacts, trust anchors, and provenance exactly as with deterministic
tools.

TimesFM 2.5 is used zero-shot in Phase 7b. Fine-tuning examples in the upstream
TimesFM repository are deferred because they create mutable model state and
belong behind release-gated training governance.

Default tests use deterministic fake runners. Real model dependencies and
weights are opt-in and run through bounded Vertex evaluations. Heavy model
packages stay out of the base dependency set.

Model-specialist outputs must be LLM-consumable as well as machine-readable:
structured JSON/CSV artifacts plus concise claims that expose the forecast,
quantile bands, assumptions, and limitations.

## Consequences

- A model specialist can be stochastic or probabilistic without weakening
  provenance; the artifact stores the actual output used by downstream steps.
- Specialist-local weights are external tool state, not VECL learning memory.
- Fine-tuned specialist checkpoints require a later governance/release path.
- Phase-specific GPU tests must validate both routing and downstream use of the
  model specialist's artifact.

## Rejected Alternatives

- Import upstream agent skills as privileged behavior — bypasses VECL registry,
  provenance, ethics, and artifact contracts.
- Make TimesFM a LoRA memory substrate — out of scope and contrary to ADR-009.
- Add TimesFM to base dependencies — would make normal `make all` depend on a
  model stack and weight download path.
