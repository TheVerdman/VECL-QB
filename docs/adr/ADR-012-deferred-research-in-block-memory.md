# ADR-012: Deferred Research — In-Block Sparse Memory Layer

## Context

The original VECL 2.0 research design called for a `SparseMemoryLayer` integrated into one or more transformer blocks — a Titans-style addressable neural memory with N key/value slots, selected and updated via Sparse Memory Fine-tuning. The current substrate decision (ADR-009, refined by Phase 1) instead uses LoRA rows in `mlp.up_proj` of selected layers as VECL slots, leaving the Gemma forward graph structurally untouched.

The in-block memory path is the architecturally more ambitious choice. It would also be the right place for a future research collaborator to take a research swing without re-architecting from scratch. This ADR records the decision to defer, the rationale, and the compatibility constraints the platform must preserve for the eventual implementation.

## Decision

In-block `SparseMemoryLayer` is **deferred** from the initial build. The platform thesis (ADR-011) is met by three already-planned mechanisms:

1. **LoRA-row substrate** (Phase 1) — procedural learning of tool-use behavior.
2. **EpisodicStore** (Phase 5) — episodic recall of past experiences and tool-chain outcomes via a vector-indexed store with provenance.
3. **EWC-bounded continual learning** (ADR-013) — principled parameter drift constraint applied to the LoRA substrate.

These three together cover the thesis without architectural surgery. In-block addressable semantic memory would be a structural improvement on cross-domain transfer and consolidation, not a thesis-required component. The marginal value lives in subtle cross-chain transfer effects that are difficult to demonstrate without a controlled experiment of its own.

When this research bet is eventually taken — likely by future research collaborators with academic interest in continual-learning architectures — the platform must remain *substrate-compatible*. Specifically:

- The `EpisodicStore` data structures must be designable as a consolidation source for in-block memory. Episodic entries should carry the metadata needed for later semantic-slot consolidation (embeddings, source/authority, access counts, freshness).
- The provenance ledger must accommodate per-slot writes from a future `SparseMemoryLayer` without schema rewrite. New `EventType` values can be added; the chain-hash structure must remain compatible.
- The substrate registry must support a *second* substrate (in-block layer) alongside LoRA rows. The `LoRAMemorySubstrate` should not assume it is the only substrate; runtime code should accept a `Sequence[Substrate]` rather than a single instance.
- The EWC bound design in ADR-013 must remain applicable to non-LoRA parameters (Fisher computation per slot, drift constraint on whatever parameters the slot maps to).

## Consequences

- The current and near-term phases proceed on the LoRA substrate without architectural surgery on Gemma.
- The "research door" in the engineering-vs-research tradeoff stays open through explicit compatibility constraints, not through speculative scaffolding code.
- Future implementers can use this ADR as the design contract for compatibility expectations.
- The deferred work is documented and reviewable, not silently dropped.

## Rejected Alternatives

- **Build in-block memory now** — substantial substrate research with no published precedent specific to provenance-aware sparse updates inside transformer blocks; not load-bearing for the platform thesis; would block tractable progress on Phases 2–13.
- **Drop the future-work entry entirely** — loses the option to pick it up later cleanly, and risks architectural decisions in Phases 2–13 that would make in-block memory expensive to add.
- **Build a partial / scaffolded in-block layer now** — half-implementations rot. Either build it or document the deferral; no third path.
- **Treat the LoRA substrate as a permanent substitute** — would require updating ADR-011's "continual learning" claim to match the more limited reality. Better to keep the door open.
