from __future__ import annotations

from pathlib import Path

import pytest

from vecl.episodic.store import (
    DeterministicEmbedder,
    EpisodicEntry,
    EpisodicStore,
    _qdrant_scroll_all,
)


def _entry(
    entry_id: str, tenant_id: str, text: str, success_status: str = "succeeded"
) -> EpisodicEntry:
    embedder = DeterministicEmbedder(dimensions=32)
    return EpisodicEntry(
        entry_id=entry_id,
        tenant_id=tenant_id,
        embedding=embedder.embed([text])[0],
        raw_text=text,
        metadata={
            "specialist_id": "stockfish",
            "source_id": "fixture",
            "anchor_id": "anchor-stockfish",
            "timestamp": "2026-05-19T00:00:00+00:00",
            "access_count": 0,
            "freshness": 1.0,
            "success_status": success_status,
        },
        chain_id="chain-a",
        artifact_ids=[f"artifact-{entry_id}"],
    )


def test_write_and_retrieve_are_tenant_scoped_at_collection_boundary() -> None:
    store = EpisodicStore(embedder=DeterministicEmbedder(dimensions=32))
    for index in range(50):
        store.write(_entry(f"a-{index:02d}", "tenant-a", f"chess tactic knight fork {index}"))
        store.write(_entry(f"b-{index:02d}", "tenant-b", f"terraform plan resource {index}"))

    tenant_a_results = store.retrieve_text("knight fork chess", k=10, tenant_id="tenant-a")
    tenant_b_results = store.retrieve_text("knight fork chess", k=10, tenant_id="tenant-b")

    assert tenant_a_results
    assert {entry.tenant_id for entry in tenant_a_results} == {"tenant-a"}
    assert {entry.tenant_id for entry in tenant_b_results} == {"tenant-b"}
    assert all("tenant-a" not in entry.entry_id for entry in tenant_b_results)


def test_retrieve_supports_exact_metadata_filters() -> None:
    store = EpisodicStore(embedder=DeterministicEmbedder(dimensions=32))
    store.write(_entry("ok", "tenant", "stockfish found best move", "succeeded"))
    store.write(_entry("failed", "tenant", "stockfish failed timeout", "failed"))

    results = store.retrieve_text(
        "stockfish",
        k=5,
        tenant_id="tenant",
        filters={"success_status": "succeeded"},
    )

    assert [entry.entry_id for entry in results] == ["ok"]


def test_sample_for_replay_is_deterministic() -> None:
    store = EpisodicStore(embedder=DeterministicEmbedder(dimensions=16))
    store.write(_entry("low", "tenant", "low value"))
    store.write(_entry("high-b", "tenant", "high value b"))
    store.write(_entry("high-a", "tenant", "high value a"))

    sampled = store.sample_for_replay(
        "tenant",
        lambda entry: 10.0 if entry.entry_id.startswith("high") else 1.0,
        2,
    )

    assert [entry.entry_id for entry in sampled] == ["high-a", "high-b"]


def test_mark_accessed_updates_access_metadata() -> None:
    store = EpisodicStore(embedder=DeterministicEmbedder(dimensions=16))
    store.write(_entry("entry", "tenant", "some useful experience"))

    updated = store.mark_accessed("entry")

    assert updated.metadata["access_count"] == 1
    assert "last_accessed_at" in updated.metadata
    assert store.mark_accessed("entry").metadata["access_count"] == 2


def test_local_backend_round_trips_entries_from_disk(tmp_path: Path) -> None:
    first = EpisodicStore.local(tmp_path, embedder=DeterministicEmbedder(dimensions=32))
    first.write(_entry("entry", "tenant", "persistent chain memory"))

    reopened = EpisodicStore.local(tmp_path, embedder=DeterministicEmbedder(dimensions=32))

    results = reopened.retrieve_text("persistent memory", k=1, tenant_id="tenant")
    assert [entry.entry_id for entry in results] == ["entry"]


def test_invalid_entries_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="embedding must be non-empty"):
        EpisodicEntry("entry", "tenant", [], "text", {})


def test_qdrant_scroll_all_paginates_until_exhausted() -> None:
    client = _PagedQdrantClient([_Point({"entry_id": f"entry-{index}"}) for index in range(5)])

    points = _qdrant_scroll_all(client, collection="tenant", page_size=2)

    assert [point.payload["entry_id"] for point in points] == [
        "entry-0",
        "entry-1",
        "entry-2",
        "entry-3",
        "entry-4",
    ]
    assert client.offsets == [None, 2, 4]


class _Point:
    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload


class _PagedQdrantClient:
    def __init__(self, points: list[_Point]) -> None:
        self.points = points
        self.offsets: list[int | None] = []

    def scroll(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: int | None,
        with_payload: bool,
    ) -> tuple[list[_Point], int | None]:
        assert collection_name == "tenant"
        assert with_payload
        self.offsets.append(offset)
        start = offset or 0
        end = min(start + limit, len(self.points))
        next_offset = end if end < len(self.points) else None
        return self.points[start:end], next_offset
