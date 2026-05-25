from __future__ import annotations

import os

import pytest

from vecl.episodic.store import (
    CausalLMHiddenStateEmbedder,
    DeterministicEmbedder,
    EpisodicEntry,
    EpisodicStore,
    QdrantVectorBackend,
    SentenceTransformerEmbedder,
)


def test_qdrant_backend_round_trip_when_dependency_is_installed(tmp_path) -> None:
    pytest.importorskip("qdrant_client")
    embedder = DeterministicEmbedder(dimensions=16)
    store = EpisodicStore(
        backend=QdrantVectorBackend(tmp_path),
        embedder=embedder,
    )
    store.write(
        EpisodicEntry(
            "entry",
            "tenant",
            embedder.embed(["qdrant chess memory"])[0],
            "qdrant chess memory",
            {"success_status": "succeeded"},
        )
    )

    assert [
        entry.entry_id for entry in store.retrieve_text("chess memory", k=1, tenant_id="tenant")
    ] == ["entry"]


@pytest.mark.skipif(
    os.environ.get("VECL_RUN_EPISODIC_REAL_EMBEDDING_TESTS") != "1",
    reason="real embedding tests are opt-in",
)
def test_sentence_transformer_embedder_retrieves_easy_cluster() -> None:
    embedder = SentenceTransformerEmbedder.from_env()
    store = EpisodicStore(embedder=embedder)
    store.write_text(
        entry_id="chess",
        tenant_id="tenant",
        raw_text="Stockfish evaluated a tactical chess position with a forcing queen move.",
        metadata={"success_status": "succeeded"},
    )
    store.write_text(
        entry_id="terraform",
        tenant_id="tenant",
        raw_text="Terraform planned an infrastructure security group update.",
        metadata={"success_status": "succeeded"},
    )

    assert store.retrieve_text("best chess move", k=1, tenant_id="tenant")[0].entry_id == "chess"


@pytest.mark.skipif(
    os.environ.get("VECL_RUN_GEMMA_TESTS") != "1"
    or not os.environ.get("VECL_EPISODIC_EMBED_MODEL_ID")
    or not os.environ.get("HF_TOKEN"),
    reason="Gemma/base-model embedding tests require explicit GPU/model opt-in",
)
def test_causal_lm_hidden_state_embedder_returns_normalized_vectors() -> None:
    embedder = CausalLMHiddenStateEmbedder.from_env()
    vectors = embedder.embed(
        [
            "Stockfish found a forcing tactical line.",
            "Terraform generated a plan-only infrastructure artifact.",
        ]
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert all(abs(sum(value * value for value in vector) - 1.0) < 1e-3 for vector in vectors)
