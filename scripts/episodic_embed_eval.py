#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecl.episodic.store import (  # noqa: E402
    CausalLMHiddenStateEmbedder,
    DeterministicEmbedder,
    EpisodicEmbedder,
    EpisodicStore,
    SentenceTransformerEmbedder,
)


@dataclass(frozen=True)
class EpisodicEvalConfig:
    embed_mode: str
    vector_backend: str
    root: Path

    @classmethod
    def from_env(cls) -> EpisodicEvalConfig:
        root = Path(
            os.environ.get(
                "VECL_EPISODIC_EVAL_ROOT",
                str(Path(tempfile.gettempdir()) / "vecl-qb-episodic-eval"),
            )
        )
        return cls(
            embed_mode=os.environ.get("VECL_EPISODIC_EMBED_MODE", "deterministic"),
            vector_backend=os.environ.get("VECL_EPISODIC_VECTOR_BACKEND", "local"),
            root=root,
        )


def main() -> int:
    config = EpisodicEvalConfig.from_env()
    if config.root.exists():
        shutil.rmtree(config.root)
    config.root.mkdir(parents=True)
    embedder = _build_embedder(config.embed_mode)
    store = _build_store(config, embedder)

    for entry in _entries():
        store.write_text(**entry)

    query_vector = embedder.embed(["Stockfish chess best move and principal variation"])[0]
    tenant_a_results = store.retrieve(query_vector, k=3, tenant_id="tenant-a")
    tenant_b_results = store.retrieve(query_vector, k=3, tenant_id="tenant-b")
    summary = _summarize(config, embedder, tenant_a_results, tenant_b_results)
    print("EPISODIC_EMBED_EVAL_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if _passes(summary) else 1


def _build_embedder(mode: str) -> EpisodicEmbedder:
    if mode == "deterministic":
        return DeterministicEmbedder(dimensions=64)
    if mode == "sentence_transformer":
        return SentenceTransformerEmbedder.from_env()
    if mode == "causal_lm":
        return CausalLMHiddenStateEmbedder.from_env()
    raise ValueError(f"unknown VECL_EPISODIC_EMBED_MODE: {mode}")


def _build_store(config: EpisodicEvalConfig, embedder: EpisodicEmbedder) -> EpisodicStore:
    if config.vector_backend == "local":
        return EpisodicStore.local(config.root / "local", embedder=embedder)
    if config.vector_backend == "qdrant":
        return EpisodicStore.qdrant(config.root / "qdrant", embedder=embedder)
    raise ValueError(f"unknown VECL_EPISODIC_VECTOR_BACKEND: {config.vector_backend}")


def _entries() -> list[dict[str, Any]]:
    return [
        {
            "entry_id": "tenant-a-chess-1",
            "tenant_id": "tenant-a",
            "raw_text": (
                "Stockfish chess analysis: best move e2e4, evaluation +0.3, "
                "principal variation e2e4 e7e5 g1f3."
            ),
            "metadata": _metadata("stockfish"),
            "chain_id": "chain-chess",
            "artifact_ids": ["artifact-chess-1"],
        },
        {
            "entry_id": "tenant-a-chess-2",
            "tenant_id": "tenant-a",
            "raw_text": (
                "Stockfish found a forcing chess tactic with best move d1h5 and "
                "principal variation d1h5 g7g6."
            ),
            "metadata": _metadata("stockfish"),
            "chain_id": "chain-chess",
            "artifact_ids": ["artifact-chess-2"],
        },
        {
            "entry_id": "tenant-a-terraform",
            "tenant_id": "tenant-a",
            "raw_text": "Terraform plan-only artifact for an infrastructure security group update.",
            "metadata": _metadata("terraform"),
            "chain_id": "chain-terraform",
            "artifact_ids": ["artifact-terraform"],
        },
        {
            "entry_id": "tenant-b-chess-private",
            "tenant_id": "tenant-b",
            "raw_text": "Tenant B private Stockfish chess best move analysis must not leak to tenant A.",
            "metadata": _metadata("stockfish"),
            "chain_id": "chain-private",
            "artifact_ids": ["artifact-private"],
        },
    ]


def _metadata(specialist_id: str) -> dict[str, Any]:
    return {
        "specialist_id": specialist_id,
        "source_id": specialist_id,
        "anchor_id": f"anchor-{specialist_id}",
        "timestamp": "2026-05-19T00:00:00+00:00",
        "access_count": 0,
        "freshness": 1.0,
        "success_status": "succeeded",
    }


def _summarize(
    config: EpisodicEvalConfig,
    embedder: EpisodicEmbedder,
    tenant_a_results: list[Any],
    tenant_b_results: list[Any],
) -> dict[str, Any]:
    probe_vectors = embedder.embed(["Stockfish chess", "Terraform infrastructure"])
    norms = [math.sqrt(sum(value * value for value in vector)) for vector in probe_vectors]
    dimensions = {len(vector) for vector in probe_vectors}
    return {
        "embed_mode": config.embed_mode,
        "vector_backend": config.vector_backend,
        "embedding_dimensions": sorted(dimensions),
        "finite_vectors": all(math.isfinite(value) for vector in probe_vectors for value in vector),
        "normalized_vectors": all(abs(norm - 1.0) < 1e-3 for norm in norms),
        "tenant_a_result_ids": [entry.entry_id for entry in tenant_a_results],
        "tenant_b_result_ids": [entry.entry_id for entry in tenant_b_results],
        "tenant_a_top_is_chess": bool(
            tenant_a_results and tenant_a_results[0].entry_id.startswith("tenant-a-chess")
        ),
        "tenant_b_isolated": all(entry.tenant_id == "tenant-b" for entry in tenant_b_results),
    }


def _passes(summary: dict[str, Any]) -> bool:
    return (
        summary["finite_vectors"]
        and summary["normalized_vectors"]
        and len(summary["embedding_dimensions"]) == 1
        and summary["tenant_a_top_is_chess"]
        and summary["tenant_b_isolated"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
