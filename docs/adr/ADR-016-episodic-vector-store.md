# ADR-016: Episodic Vector Store

## Context

ADR-011 requires QB to remember tool-chain experiences with provenance, tenant isolation, and artifact links. Phase 5 adds this episodic layer before semantic consolidation. ADR-015 already covers real-resource validation, so the vector-store decision is recorded here.

## Decision

Use Qdrant embedded mode as the production-direction vector store for episodic memory. Store entries in one collection per tenant, and also include `tenant_id` in each payload for audit. Default unit tests use a deterministic in-process backend with the same collection-per-tenant semantics so `make all` does not require network access, model downloads, or optional services.

The embedding surface is a protocol. Phase 5 ships three implementations:

- `DeterministicEmbedder` for default tests.
- `SentenceTransformerEmbedder` for a practical local semantic baseline.
- `CausalLMHiddenStateEmbedder` for opt-in base-model-agnostic research with Gemma or another Hugging Face model.

## Consequences

- Tenant isolation is enforced at lookup scope, not by post-filtering mixed-tenant results.
- Heavy retrieval dependencies stay behind optional extras.
- Gemma embeddings can be evaluated without making them the correctness dependency for Phase 5.
- The chain executor can write episodic entries while preserving existing Phase 4 behavior when no store is configured.

## Phase 5 Eval Finding

The Vertex A100 eval against `google/gemma-4-31B-it` produced finite normalized
5376-dimensional vectors with Qdrant tenant isolation. Mean pooling over Gemma 4
hidden states failed the tiny retrieval probe by ranking a Terraform memory above
the chess memories for a chess query. Last-token pooling passed the same probe,
so `CausalLMHiddenStateEmbedder` uses last-token pooling until a broader
retrieval-quality eval says otherwise.

## Rejected Alternatives

- Chroma — workable, but Qdrant's explicit payload model and collection isolation are a better fit for tenant boundaries.
- One shared collection with metadata filters only — easier to query, weaker isolation discipline.
- Gemma-only embeddings — valuable research signal, but too unproven as the default retrieval substrate.
