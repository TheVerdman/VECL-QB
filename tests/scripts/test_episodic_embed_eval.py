from __future__ import annotations

from pathlib import Path

from scripts.episodic_embed_eval import EpisodicEvalConfig, _build_embedder, _build_store, _passes


def test_episodic_embed_eval_pass_contract() -> None:
    assert _passes(
        {
            "finite_vectors": True,
            "normalized_vectors": True,
            "embedding_dimensions": [64],
            "tenant_a_top_is_chess": True,
            "tenant_b_isolated": True,
        }
    )
    assert not _passes(
        {
            "finite_vectors": True,
            "normalized_vectors": True,
            "embedding_dimensions": [64],
            "tenant_a_top_is_chess": False,
            "tenant_b_isolated": True,
        }
    )


def test_episodic_embed_eval_local_store_can_retrieve(tmp_path: Path) -> None:
    config = EpisodicEvalConfig("deterministic", "local", tmp_path)
    embedder = _build_embedder("deterministic")
    store = _build_store(config, embedder)
    store.write_text(
        entry_id="entry",
        tenant_id="tenant",
        raw_text="Stockfish chess best move",
        metadata={"success_status": "succeeded"},
    )

    assert store.retrieve_text("chess best move", k=1, tenant_id="tenant")[0].entry_id == "entry"
