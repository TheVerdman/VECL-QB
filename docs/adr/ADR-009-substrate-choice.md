# ADR-009: Gemma Dense Substrate With Sparse LoRA And KV Memory

## Context

VECL-QB needs a concrete model substrate before adding invasive memory-layer work. The substrate must support local development, cloud GPU validation, stable slot semantics, attribution, rollback, and a frozen-base discipline.

## Decision

Use the Gemma 4 E4B family for development and Gemma 4 31B Dense for production-scale validation. The default Phase 0 smoke/demo checkpoint is `google/gemma-4-E4B-it`; later substrate research may use the base `google/gemma-4-E4B` checkpoint when instruction tuning would distort training signals.

Base Gemma weights are frozen. Mutable learning surfaces are limited to sparse-row LoRA adapter rows and external KV memory. One LoRA row can become a stable VECL slot in later phases, while KV memory remains an episodic, non-gradient substrate.

Reject the Gemma 4 MoE path for this track. Expert routing complicates stable slot semantics, provenance attribution, rollback, and invasive memory-layer replacement because the active parameter path can vary by token and prompt. Dense layers give a simpler and more auditable substrate.

CUDA kernels remain out of scope for this phase. GPU validation is limited to proving that an approved A100 80GB or H100-class environment can load and run the selected Gemma checkpoint.

## Consequences

The first substrate implementation can focus on frozen-base PEFT/LoRA mechanics and external KV memory without changing base model weights. Production scaling can reuse the same conceptual slot policy on 31B Dense after the E4B path is proven.

The repo will include a manual Gemma smoke script for A100/H100 validation, but normal local tests and `make all` will not download model weights or require GPU hardware.

## Rejected Alternatives

- Use Gemma 4 26B A4B MoE as the primary substrate.
- Fine-tune or mutate base Gemma weights directly.
- Treat CUDA kernels as part of the initial substrate lock-in.
- Start directly on 31B before E4B substrate mechanics are proven.

## Revision (2026-05-18): Dev/Prod Boundary

The original framing implied that 31B Dense would arrive late in the phase plan — only after E4B substrate mechanics were proven across many phases. Practice has clarified that this boundary is wrong. The correct split is by *kind of work*, not by phase number:

- **E4B is used for model-agnostic platform mechanics**: plugin platform, registry, artifact storage with content hashing, EpisodicStore, EthicsKernel, sequential chain planner logic, release-gate evaluator infrastructure, persistence, and CI tests against a tiny shape-compatible local model.
- **31B Dense is the default model for any capability-dependent work** from Phase 3 onward: first specialist integration, chain planner end-to-end validation, real release-gate eval scores, tool-use training (Phase 11), Fisher diagonal computation, EWC drift bounds, and every eval metric that involves "did the agent actually do the thing."

Concrete effects:

- The opt-in Gemma test in `tests/substrate/test_lora_memory.py` should parameterize across both checkpoints. 31B becomes the default for capability-dependent tests; E4B remains for shape and plumbing tests.
- The originally-planned "Phase 13: Scale to 31B" disappears as a discrete migration phase and is replaced by "Production hardening on 31B" (performance tuning, batching, serving config, cost optimization).
- The dev/prod *checkpoint* split survives unchanged: `google/gemma-4-E4B-it` and `google/gemma-4-E4B` remain the development defaults; the corresponding 31B checkpoints (`google/gemma-4-31B` and its `-it` variant) become the production defaults from Phase 3 onward.

The MoE rejection in this ADR is unchanged. The LoRA-on-dense substrate decision remains the primary reason for staying on dense regardless of the dev/prod boundary update.
