from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from vecl._compat import UTC
from vecl._gemma import resolve_gemma_model_class, resolve_torch_dtype

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_SENTENCE_TRANSFORMER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_QDRANT_SCROLL_PAGE_SIZE = 256


class EpisodicEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class EpisodicEntry:
    entry_id: str
    tenant_id: str
    embedding: list[float]
    raw_text: str
    metadata: dict[str, Any]
    chain_id: str | None = None
    artifact_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValueError("entry_id must be non-empty")
        if not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if not self.raw_text:
            raise ValueError("raw_text must be non-empty")
        _validate_embedding(self.embedding)
        object.__setattr__(self, "embedding", [float(value) for value in self.embedding])
        object.__setattr__(self, "artifact_ids", sorted(set(self.artifact_ids)))

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeterministicEmbedder:
    dimensions: int = 64

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_normalize(_token_hash_vector(text, self.dimensions)) for text in texts]


class SentenceTransformerEmbedder:
    def __init__(self, model_id: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL_ID) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.model = SentenceTransformer(model_id)

    @classmethod
    def from_env(cls) -> SentenceTransformerEmbedder:
        return cls(
            os.environ.get("VECL_EPISODIC_SENTENCE_MODEL_ID", DEFAULT_SENTENCE_TRANSFORMER_MODEL_ID)
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(value) for value in row] for row in vectors.tolist()]


class CausalLMHiddenStateEmbedder:
    def __init__(
        self,
        model_id: str,
        *,
        instruction: str = "Represent this chain step for retrieval:",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        hf_token: str | None = None,
    ) -> None:
        import torch

        self.model_id = model_id
        self.instruction = instruction
        dtype_value = resolve_torch_dtype(torch, dtype)
        kwargs: dict[str, Any] = {"device_map": device_map, "token": hf_token}
        if dtype_value is not None:
            kwargs["torch_dtype"] = dtype_value
        self.tokenizer: Any | None = None
        self.processor: Any | None = None
        self._input_mode = "tokenizer"
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
            self.model = AutoModel.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            from transformers import AutoProcessor

            # Gemma 4 multimodal checkpoints may not load through AutoModel in current
            # Transformers releases; keep this aligned with the generation smoke path.
            model_cls = resolve_gemma_model_class()
            if model_cls is None:
                raise RuntimeError("no compatible hidden-state model class is available") from exc
            self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
            self.model = model_cls.from_pretrained(model_id, **kwargs)
            self._input_mode = "processor"
        self.model.eval()

    @classmethod
    def from_env(cls) -> CausalLMHiddenStateEmbedder:
        model_id = os.environ.get("VECL_EPISODIC_EMBED_MODEL_ID")
        if not model_id:
            raise ValueError("VECL_EPISODIC_EMBED_MODEL_ID must be set")
        return cls(
            model_id,
            instruction=os.environ.get(
                "VECL_EPISODIC_EMBED_INSTRUCTION",
                "Represent this chain step for retrieval:",
            ),
            dtype=os.environ.get("VECL_EPISODIC_EMBED_DTYPE", "bfloat16"),
            device_map=os.environ.get("VECL_EPISODIC_EMBED_DEVICE_MAP", "auto"),
            hf_token=os.environ.get("HF_TOKEN"),
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        prompts = [f"{self.instruction}\n{text}" for text in texts]
        if self._input_mode == "processor":
            return [self._embed_processor_prompt(prompt, torch) for prompt in prompts]
        assert self.tokenizer is not None
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = self.model(**encoded)
        hidden = _last_hidden_state(outputs)
        pooled = _last_token_pool(hidden, encoded["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        return [[float(value) for value in row] for row in normalized.cpu().tolist()]

    def _embed_processor_prompt(self, prompt: str, torch_module: Any) -> list[float]:
        assert self.processor is not None
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        encoded = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch_module.inference_mode():
            outputs = self.model(**encoded, output_hidden_states=True, return_dict=True)
        hidden = _last_hidden_state(outputs)
        pooled = _last_token_pool(hidden, encoded["attention_mask"])
        normalized = torch_module.nn.functional.normalize(pooled.float(), p=2, dim=1)
        return [float(value) for value in normalized[0].cpu().tolist()]


class VectorBackend(Protocol):
    def write(self, entry: EpisodicEntry) -> None:
        raise NotImplementedError

    def retrieve(
        self,
        query_embedding: Sequence[float],
        *,
        k: int,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
    ) -> list[EpisodicEntry]:
        raise NotImplementedError

    def entries_for_tenant(self, tenant_id: str) -> list[EpisodicEntry]:
        raise NotImplementedError

    def get(self, entry_id: str) -> EpisodicEntry | None:
        raise NotImplementedError

    def replace(self, entry: EpisodicEntry) -> None:
        raise NotImplementedError


class LocalVectorBackend:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else None
        self._entries_by_tenant: dict[str, dict[str, EpisodicEntry]] = {}
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._load()

    def write(self, entry: EpisodicEntry) -> None:
        self._entries_by_tenant.setdefault(entry.tenant_id, {})[entry.entry_id] = entry
        self._persist_tenant(entry.tenant_id)

    def retrieve(
        self,
        query_embedding: Sequence[float],
        *,
        k: int,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
    ) -> list[EpisodicEntry]:
        _validate_embedding(query_embedding)
        entries = self.entries_for_tenant(tenant_id)
        if filters:
            entries = [entry for entry in entries if _matches_filters(entry, filters)]
        ranked = sorted(
            entries,
            key=lambda entry: (-_cosine(query_embedding, entry.embedding), entry.entry_id),
        )
        return ranked[:k]

    def entries_for_tenant(self, tenant_id: str) -> list[EpisodicEntry]:
        return list(self._entries_by_tenant.get(tenant_id, {}).values())

    def get(self, entry_id: str) -> EpisodicEntry | None:
        for entries in self._entries_by_tenant.values():
            if entry_id in entries:
                return entries[entry_id]
        return None

    def replace(self, entry: EpisodicEntry) -> None:
        if entry.entry_id not in self._entries_by_tenant.get(entry.tenant_id, {}):
            raise KeyError(entry.entry_id)
        self._entries_by_tenant[entry.tenant_id][entry.entry_id] = entry
        self._persist_tenant(entry.tenant_id)

    def _load(self) -> None:
        assert self.root is not None
        for path in self.root.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                if not line:
                    continue
                payload = json.loads(line)
                entry = EpisodicEntry(**payload)
                self._entries_by_tenant.setdefault(entry.tenant_id, {})[entry.entry_id] = entry

    def _persist_tenant(self, tenant_id: str) -> None:
        if self.root is None:
            return
        path = self.root / f"{_safe_tenant_id(tenant_id)}.jsonl"
        entries = sorted(
            self._entries_by_tenant.get(tenant_id, {}).values(), key=lambda e: e.entry_id
        )
        path.write_text(
            "\n".join(json.dumps(entry.to_payload(), sort_keys=True) for entry in entries)
        )


class QdrantVectorBackend:
    def __init__(self, path: Path | str, *, collection_prefix: str = "vecl_episodic") -> None:
        from qdrant_client import QdrantClient

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_prefix = collection_prefix
        self.client = QdrantClient(path=str(self.path))
        self._entries: dict[str, EpisodicEntry] = {}

    def write(self, entry: EpisodicEntry) -> None:
        from qdrant_client.http.models import Distance, PointStruct, VectorParams

        collection = self._collection(entry.tenant_id)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=VectorParams(size=len(entry.embedding), distance=Distance.COSINE),
            )
        point_id = _qdrant_point_id(entry.entry_id)
        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=entry.embedding,
                    payload=entry.to_payload(),
                )
            ],
        )
        self._entries[entry.entry_id] = entry

    def retrieve(
        self,
        query_embedding: Sequence[float],
        *,
        k: int,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
    ) -> list[EpisodicEntry]:
        _validate_embedding(query_embedding)
        collection = self._collection(tenant_id)
        if not self.client.collection_exists(collection):
            return []
        results = _qdrant_search(
            self.client,
            collection=collection,
            query_embedding=query_embedding,
            limit=max(k * 4, k),
        )
        entries = [
            EpisodicEntry(**dict(result.payload or {}))
            for result in results
            if result.payload is not None
        ]
        if filters:
            entries = [entry for entry in entries if _matches_filters(entry, filters)]
        return entries[:k]

    def entries_for_tenant(self, tenant_id: str) -> list[EpisodicEntry]:
        collection = self._collection(tenant_id)
        if not self.client.collection_exists(collection):
            return []
        points = _qdrant_scroll_all(
            self.client,
            collection=collection,
            page_size=_QDRANT_SCROLL_PAGE_SIZE,
        )
        return [
            EpisodicEntry(**dict(point.payload or {}))
            for point in points
            if point.payload is not None
        ]

    def get(self, entry_id: str) -> EpisodicEntry | None:
        if entry_id in self._entries:
            return self._entries[entry_id]
        for collection in self.client.get_collections().collections:
            points = self.client.retrieve(
                collection_name=collection.name,
                ids=[_qdrant_point_id(entry_id)],
                with_payload=True,
            )
            if points and points[0].payload:
                entry = EpisodicEntry(**dict(points[0].payload))
                self._entries[entry.entry_id] = entry
                return entry
        return None

    def replace(self, entry: EpisodicEntry) -> None:
        self.write(entry)

    def _collection(self, tenant_id: str) -> str:
        return f"{self.collection_prefix}_{_safe_tenant_id(tenant_id)}"


class EpisodicStore:
    def __init__(
        self,
        *,
        backend: VectorBackend | None = None,
        embedder: EpisodicEmbedder | None = None,
    ) -> None:
        self.backend = backend or LocalVectorBackend()
        self.embedder = embedder or DeterministicEmbedder()

    @classmethod
    def local(
        cls,
        root: Path | str | None = None,
        *,
        embedder: EpisodicEmbedder | None = None,
    ) -> EpisodicStore:
        return cls(backend=LocalVectorBackend(root), embedder=embedder)

    @classmethod
    def qdrant(
        cls,
        root: Path | str,
        *,
        embedder: EpisodicEmbedder | None = None,
        collection_prefix: str = "vecl_episodic",
    ) -> EpisodicStore:
        return cls(
            backend=QdrantVectorBackend(root, collection_prefix=collection_prefix),
            embedder=embedder,
        )

    def write(self, entry: EpisodicEntry) -> EpisodicEntry:
        self.backend.write(entry)
        return entry

    def write_text(
        self,
        *,
        entry_id: str,
        tenant_id: str,
        raw_text: str,
        metadata: dict[str, Any],
        chain_id: str | None = None,
        artifact_ids: Iterable[str] = (),
    ) -> EpisodicEntry:
        embedding = self.embedder.embed([raw_text])[0]
        return self.write(
            EpisodicEntry(
                entry_id=entry_id,
                tenant_id=tenant_id,
                embedding=embedding,
                raw_text=raw_text,
                metadata=dict(metadata),
                chain_id=chain_id,
                artifact_ids=list(artifact_ids),
            )
        )

    def retrieve(
        self,
        query_embedding: Sequence[float],
        *,
        k: int,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
    ) -> list[EpisodicEntry]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        return self.backend.retrieve(query_embedding, k=k, tenant_id=tenant_id, filters=filters)

    def retrieve_text(
        self,
        query_text: str,
        *,
        k: int,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
    ) -> list[EpisodicEntry]:
        return self.retrieve(
            self.embedder.embed([query_text])[0],
            k=k,
            tenant_id=tenant_id,
            filters=filters,
        )

    def sample_for_replay(
        self,
        tenant_id: str,
        importance_fn: Callable[[EpisodicEntry], float],
        n: int,
    ) -> list[EpisodicEntry]:
        if n <= 0:
            raise ValueError("n must be positive")
        entries = self.backend.entries_for_tenant(tenant_id)
        ranked = sorted(entries, key=lambda entry: (-importance_fn(entry), entry.entry_id))
        return ranked[:n]

    def mark_accessed(self, entry_id: str) -> EpisodicEntry:
        entry = self.backend.get(entry_id)
        if entry is None:
            raise KeyError(entry_id)
        metadata = dict(entry.metadata)
        metadata["access_count"] = int(metadata.get("access_count", 0)) + 1
        metadata["last_accessed_at"] = datetime.now(UTC).isoformat()
        updated = EpisodicEntry(
            entry_id=entry.entry_id,
            tenant_id=entry.tenant_id,
            embedding=list(entry.embedding),
            raw_text=entry.raw_text,
            metadata=metadata,
            chain_id=entry.chain_id,
            artifact_ids=list(entry.artifact_ids),
        )
        self.backend.replace(updated)
        return updated


def _token_hash_vector(text: str, dimensions: int) -> list[float]:
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    vector = [0.0] * dimensions
    for token in _TOKEN_RE.findall(text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0
    if not any(vector):
        vector[0] = 1.0
    return vector


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm == 0:
        raise ValueError("embedding norm must be non-zero")
    return [float(value) / norm for value in vector]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def _validate_embedding(embedding: Sequence[float]) -> None:
    if not embedding:
        raise ValueError("embedding must be non-empty")
    if not all(math.isfinite(float(value)) for value in embedding):
        raise ValueError("embedding values must be finite")


def _matches_filters(entry: EpisodicEntry, filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if key == "chain_id":
            actual = entry.chain_id
        elif key == "entry_id":
            actual = entry.entry_id
        else:
            actual = entry.metadata.get(key)
        if actual != expected:
            return False
    return True


def _safe_tenant_id(tenant_id: str) -> str:
    digest = hashlib.sha256(tenant_id.encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", tenant_id).strip("_") or "tenant"
    return f"{safe}_{digest}"


def _qdrant_point_id(entry_id: str) -> str:
    digest = hashlib.sha256(entry_id.encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _qdrant_search(
    client: Any,
    *,
    collection: str,
    query_embedding: Sequence[float],
    limit: int,
) -> list[Any]:
    if hasattr(client, "search"):
        return list(
            client.search(
                collection_name=collection,
                query_vector=list(query_embedding),
                limit=limit,
            )
        )
    response = client.query_points(
        collection_name=collection,
        query=list(query_embedding),
        limit=limit,
        with_payload=True,
    )
    return list(response.points)


def _last_hidden_state(outputs: Any) -> Any:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    raise RuntimeError("model output did not include hidden states")


def _last_token_pool(hidden: Any, attention_mask: Any) -> Any:
    import torch

    # Phase 5 GPU eval: mean pooling over Gemma 4 hidden states mixed chess and
    # terraform memories; last-token pooling passed the same retrieval probe.
    sequence_lengths = attention_mask.sum(dim=1).to(dtype=torch.long) - 1
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, sequence_lengths]


def _qdrant_scroll_all(client: Any, *, collection: str, page_size: int) -> list[Any]:
    offset: Any | None = None
    points: list[Any] = []
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=page_size,
            offset=offset,
            with_payload=True,
        )
        points.extend(batch)
        if offset is None:
            return points
