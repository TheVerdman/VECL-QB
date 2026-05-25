from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from vecl._compat import UTC
from vecl.provenance.events import EventType, ProvenanceEvent, stable_hash
from vecl.provenance.ledger import ProvenanceLedger


@dataclass(frozen=True)
class MemoryCheckpoint:
    checkpoint_id: str
    tenant_id: str
    memory_values_hash: str
    created_at: datetime
    parent_checkpoint_id: str | None = None


@dataclass(frozen=True)
class RollbackResult:
    tenant_id: str
    memory_values: np.ndarray
    exact: bool
    rollback_event_id: str
    excluded_event_ids: list[str]
    excluded_source_ids: list[str]


class CheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, MemoryCheckpoint] = {}
        self._values: dict[str, np.ndarray] = {}

    def create_checkpoint(
        self,
        memory_values: np.ndarray,
        tenant_id: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None = None,
    ) -> MemoryCheckpoint:
        values = np.asarray(memory_values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("memory_values must be finite")
        checkpoint = MemoryCheckpoint(
            checkpoint_id=checkpoint_id,
            tenant_id=tenant_id,
            memory_values_hash=stable_hash([float(value) for value in values]),
            created_at=datetime.now(UTC),
            parent_checkpoint_id=parent_checkpoint_id,
        )
        self._checkpoints[checkpoint_id] = checkpoint
        self._values[checkpoint_id] = values.copy()
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> np.ndarray:
        return self._values[checkpoint_id].copy()

    def get_checkpoint(self, checkpoint_id: str) -> MemoryCheckpoint:
        return self._checkpoints[checkpoint_id]

    def list_checkpoints(self, tenant_id: str) -> list[MemoryCheckpoint]:
        return [
            checkpoint
            for checkpoint in self._checkpoints.values()
            if checkpoint.tenant_id == tenant_id
        ]


class DiskCheckpointStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        memory_values: np.ndarray,
        tenant_id: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None = None,
    ) -> MemoryCheckpoint:
        values = np.asarray(memory_values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("memory_values must be finite")
        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        values_path = checkpoint_dir / "memory_values.npy"
        metadata_path = checkpoint_dir / "metadata.json"
        np.save(values_path, values)
        checkpoint = MemoryCheckpoint(
            checkpoint_id=checkpoint_id,
            tenant_id=tenant_id,
            memory_values_hash=stable_hash([float(value) for value in values]),
            created_at=datetime.now(UTC),
            parent_checkpoint_id=parent_checkpoint_id,
        )
        metadata = _checkpoint_to_metadata(checkpoint)
        metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2))
        (checkpoint_dir / "digest.sha256").write_text(
            _checkpoint_digest(values_path, metadata_path)
        )
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> np.ndarray:
        checkpoint_dir = self._checkpoint_dir(checkpoint_id)
        values_path = checkpoint_dir / "memory_values.npy"
        metadata_path = checkpoint_dir / "metadata.json"
        expected_digest = (checkpoint_dir / "digest.sha256").read_text().strip()
        actual_digest = _checkpoint_digest(values_path, metadata_path)
        if actual_digest != expected_digest:
            raise ValueError("checkpoint digest mismatch")
        values = np.load(values_path, allow_pickle=False)
        checkpoint = self.get_checkpoint(checkpoint_id)
        if stable_hash([float(value) for value in values]) != checkpoint.memory_values_hash:
            raise ValueError("checkpoint memory hash mismatch")
        return np.asarray(values, dtype=np.float64).copy()

    def get_checkpoint(self, checkpoint_id: str) -> MemoryCheckpoint:
        metadata_path = self._checkpoint_dir(checkpoint_id) / "metadata.json"
        return _checkpoint_from_metadata(json.loads(metadata_path.read_text()))

    def list_checkpoints(self, tenant_id: str) -> list[MemoryCheckpoint]:
        checkpoints = [
            self.get_checkpoint(path.name)
            for path in self.root.iterdir()
            if path.is_dir() and (path / "metadata.json").exists()
        ]
        return sorted(
            [checkpoint for checkpoint in checkpoints if checkpoint.tenant_id == tenant_id],
            key=lambda checkpoint: checkpoint.created_at,
        )

    def _checkpoint_dir(self, checkpoint_id: str) -> Path:
        if not checkpoint_id or "/" in checkpoint_id or "\\" in checkpoint_id:
            raise ValueError("checkpoint_id must be a non-empty path segment")
        return self.root / checkpoint_id


class RollbackService:
    def __init__(
        self, ledger: ProvenanceLedger, checkpoint_store: CheckpointStore, actor: str = "rollback"
    ) -> None:
        self.ledger = ledger
        self.checkpoint_store = checkpoint_store
        self.actor = actor

    def rollback_by_source(
        self,
        source_id: str,
        tenant_id: str,
        checkpoint_id: str | None = None,
        approximate: bool = False,
    ) -> RollbackResult:
        if approximate:
            raise ValueError("approximate rollback must use rollback_approximate")
        return self._exact_replay(
            tenant_id=tenant_id,
            checkpoint_id=checkpoint_id,
            excluded_source_ids=[source_id],
            excluded_event_ids=[],
        )

    def rollback_by_event(
        self,
        event_id: str,
        tenant_id: str,
        checkpoint_id: str | None = None,
        approximate: bool = False,
    ) -> RollbackResult:
        if approximate:
            raise ValueError("approximate rollback must use rollback_approximate")
        return self._exact_replay(
            tenant_id=tenant_id,
            checkpoint_id=checkpoint_id,
            excluded_source_ids=[],
            excluded_event_ids=[event_id],
        )

    def rollback_to_checkpoint(self, checkpoint_id: str) -> RollbackResult:
        checkpoint = self.checkpoint_store.get_checkpoint(checkpoint_id)
        values = self.checkpoint_store.restore_checkpoint(checkpoint_id)
        event = self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.ROLLBACK_PERFORMED,
                tenant_id=checkpoint.tenant_id,
                actor=self.actor,
                payload={
                    "checkpoint_id": checkpoint_id,
                    "exact": True,
                    "memory_hash": stable_hash([float(value) for value in values]),
                },
            )
        )
        return RollbackResult(
            tenant_id=checkpoint.tenant_id,
            memory_values=values,
            exact=True,
            rollback_event_id=event.event_id,
            excluded_event_ids=[],
            excluded_source_ids=[],
        )

    def rollback_approximate(self, tenant_id: str, reason: str) -> RollbackResult:
        event = self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.ROLLBACK_PERFORMED,
                tenant_id=tenant_id,
                actor=self.actor,
                payload={
                    "tenant_id": tenant_id,
                    "target_event_id": f"approximate:{reason}",
                    "reason": reason,
                    "approximate": True,
                },
            )
        )
        return RollbackResult(
            tenant_id=tenant_id,
            memory_values=np.array([], dtype=np.float64),
            exact=False,
            rollback_event_id=event.event_id,
            excluded_event_ids=[],
            excluded_source_ids=[],
        )

    def _exact_replay(
        self,
        tenant_id: str,
        checkpoint_id: str | None,
        excluded_source_ids: list[str],
        excluded_event_ids: list[str],
    ) -> RollbackResult:
        checkpoint = self._select_checkpoint(tenant_id, checkpoint_id)
        values = self.checkpoint_store.restore_checkpoint(checkpoint.checkpoint_id)
        excluded_event_set = set(excluded_event_ids)
        excluded_source_set = set(excluded_source_ids)
        for event in self.ledger.query_by_tenant(tenant_id):
            if event.event_type != EventType.SPARSE_UPDATE_APPLIED:
                continue
            payload_sources = set(event.payload.get("source_ids", []))
            if (
                event.event_id in excluded_event_set
                or event.payload.get("learning_event_id") in excluded_event_set
            ):
                continue
            if (
                payload_sources & excluded_source_set
                or event.payload.get("source_id") in excluded_source_set
            ):
                continue
            for record in event.payload.get("delta_records", []):
                values[int(record["slot_id"])] += float(record["delta"])
        rollback_event = self.ledger.append(
            ProvenanceEvent(
                event_type=EventType.ROLLBACK_PERFORMED,
                tenant_id=tenant_id,
                actor=self.actor,
                payload={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "source_id": excluded_source_ids[0] if excluded_source_ids else None,
                    "target_event_id": excluded_event_ids[0] if excluded_event_ids else None,
                    "exact": True,
                    "memory_hash": stable_hash([float(value) for value in values]),
                },
            )
        )
        return RollbackResult(
            tenant_id=tenant_id,
            memory_values=values,
            exact=True,
            rollback_event_id=rollback_event.event_id,
            excluded_event_ids=excluded_event_ids,
            excluded_source_ids=excluded_source_ids,
        )

    def _select_checkpoint(self, tenant_id: str, checkpoint_id: str | None) -> MemoryCheckpoint:
        if checkpoint_id is not None:
            checkpoint = self.checkpoint_store.get_checkpoint(checkpoint_id)
            if checkpoint.tenant_id != tenant_id:
                raise PermissionError("checkpoint tenant mismatch")
            return checkpoint
        checkpoints = sorted(
            self.checkpoint_store.list_checkpoints(tenant_id),
            key=lambda checkpoint: checkpoint.created_at,
        )
        if not checkpoints:
            raise ValueError(f"no checkpoint for tenant {tenant_id}")
        return checkpoints[0]


def _checkpoint_to_metadata(checkpoint: MemoryCheckpoint) -> dict[str, object]:
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "tenant_id": checkpoint.tenant_id,
        "memory_values_hash": checkpoint.memory_values_hash,
        "created_at": checkpoint.created_at.isoformat(),
        "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
    }


def _checkpoint_from_metadata(payload: dict[str, object]) -> MemoryCheckpoint:
    return MemoryCheckpoint(
        checkpoint_id=str(payload["checkpoint_id"]),
        tenant_id=str(payload["tenant_id"]),
        memory_values_hash=str(payload["memory_values_hash"]),
        created_at=datetime.fromisoformat(str(payload["created_at"])),
        parent_checkpoint_id=(
            str(payload["parent_checkpoint_id"])
            if payload.get("parent_checkpoint_id") is not None
            else None
        ),
    )


def _checkpoint_digest(values_path: Path, metadata_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(values_path.read_bytes())
    digest.update(metadata_path.read_bytes())
    return digest.hexdigest()
