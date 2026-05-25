from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vecl.provenance.events import stable_hash
from vecl.substrate.lora_memory import LoRARawGradient

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LoRAFisherEstimate:
    slot_fisher: FloatArray
    sample_count: int
    dataset_hash: str
    loss_sum: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _validated_vector(self.slot_fisher, name="slot_fisher", allow_zero=True)
        if self.sample_count <= 0:
            raise ValueError("sample_count must be > 0")
        if not self.dataset_hash:
            raise ValueError("dataset_hash must be non-empty")
        if not np.isfinite(self.loss_sum):
            raise ValueError("loss_sum must be finite")
        object.__setattr__(self, "slot_fisher", values)

    def to_payload(self) -> dict[str, Any]:
        return {
            "slot_fisher": [float(value) for value in self.slot_fisher],
            "sample_count": self.sample_count,
            "dataset_hash": self.dataset_hash,
            "loss_sum": self.loss_sum,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LoRADriftReport:
    drift_value: float
    drift_threshold: float
    drift_within_bound: bool
    approved_snapshot_hash: str
    candidate_snapshot_hash: str
    slot_count: int
    per_slot_drift: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = _validated_vector(self.per_slot_drift, name="per_slot_drift", allow_zero=True)
        if not np.isfinite(self.drift_value) or self.drift_value < 0:
            raise ValueError("drift_value must be finite and non-negative")
        if not np.isfinite(self.drift_threshold) or self.drift_threshold < 0:
            raise ValueError("drift_threshold must be finite and non-negative")
        if self.slot_count != values.shape[0]:
            raise ValueError("slot_count must match per_slot_drift length")
        if not self.approved_snapshot_hash or not self.candidate_snapshot_hash:
            raise ValueError("snapshot hashes must be non-empty")
        object.__setattr__(self, "per_slot_drift", values)

    def to_payload(self) -> dict[str, Any]:
        return {
            "drift_value": self.drift_value,
            "drift_threshold": self.drift_threshold,
            "drift_within_bound": self.drift_within_bound,
            "approved_snapshot_hash": self.approved_snapshot_hash,
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
            "slot_count": self.slot_count,
            "per_slot_drift": [float(value) for value in self.per_slot_drift],
            "metadata": dict(self.metadata),
        }


def accumulate_lora_fisher(
    gradient_batches: list[dict[int, LoRARawGradient]] | tuple[dict[int, LoRARawGradient], ...],
    *,
    slot_count: int,
    dataset_items: list[Any] | tuple[Any, ...],
    loss_sum: float,
    metadata: dict[str, Any] | None = None,
) -> LoRAFisherEstimate:
    if slot_count <= 0:
        raise ValueError("slot_count must be > 0")
    if not gradient_batches:
        raise ValueError("at least one gradient batch is required")
    if len(dataset_items) != len(gradient_batches):
        raise ValueError("dataset_items length must match gradient_batches length")
    if not np.isfinite(loss_sum):
        raise ValueError("loss_sum must be finite")

    fisher = np.zeros(slot_count, dtype=np.float64)
    for batch in gradient_batches:
        seen: set[int] = set()
        for slot_id, raw_gradient in batch.items():
            if slot_id != raw_gradient.slot_id:
                raise ValueError("raw gradient mapping key must match gradient slot_id")
            if slot_id < 0 or slot_id >= slot_count:
                raise ValueError(f"slot_id out of range: {slot_id}")
            if slot_id in seen:
                raise ValueError(f"duplicate slot gradient: {slot_id}")
            seen.add(slot_id)
            grad_a = _gradient_array(raw_gradient.lora_a_row)
            grad_b = _gradient_array(raw_gradient.lora_b_column)
            fisher[slot_id] += float(np.sum(np.square(grad_a)) + np.sum(np.square(grad_b)))
    fisher /= float(len(gradient_batches))
    return LoRAFisherEstimate(
        slot_fisher=fisher,
        sample_count=len(gradient_batches),
        dataset_hash=stable_hash(list(dataset_items)),
        loss_sum=float(loss_sum),
        metadata=metadata or {},
    )


def validate_fisher_diagonal(values: Any, *, slot_count: int | None = None) -> FloatArray:
    vector = _validated_vector(values, name="fisher_diagonal", allow_zero=True)
    if slot_count is not None and vector.shape != (slot_count,):
        raise ValueError(f"fisher_diagonal must have shape ({slot_count},)")
    return vector


def load_snapshot_fisher(path: str | Path) -> LoRAFisherEstimate | None:
    with np.load(path, allow_pickle=False) as snapshot:
        if "fisher_diagonal" not in snapshot:
            return None
        fisher = validate_fisher_diagonal(snapshot["fisher_diagonal"])
        metadata = {}
        if "fisher_metadata_json" in snapshot:
            metadata = json.loads(str(snapshot["fisher_metadata_json"].item()))
        if metadata.get("fisher_status") == "not_accumulated":
            return None
        return LoRAFisherEstimate(
            slot_fisher=fisher,
            sample_count=int(metadata.get("sample_count", 1)),
            dataset_hash=str(metadata.get("dataset_hash") or _snapshot_hash(path)),
            loss_sum=float(metadata.get("loss_sum", 0.0)),
            metadata=metadata,
        )


def compute_lora_snapshot_drift(
    approved_snapshot: str | Path,
    candidate_snapshot: str | Path,
    *,
    fisher_diagonal: Any | None = None,
    drift_threshold: float,
) -> LoRADriftReport:
    approved = _load_lora_snapshot(approved_snapshot)
    candidate = _load_lora_snapshot(candidate_snapshot)
    if approved["metadata"] != candidate["metadata"]:
        raise ValueError("LoRA snapshots have incompatible metadata")

    slot_metadata = approved["metadata"]["slot_metadata"]
    slot_count = int(approved["metadata"]["slot_count"])
    fisher = (
        validate_fisher_diagonal(fisher_diagonal, slot_count=slot_count)
        if fisher_diagonal is not None
        else _required_snapshot_fisher(approved_snapshot, slot_count)
    )
    per_slot_raw = np.zeros(slot_count, dtype=np.float64)
    for item in slot_metadata:
        slot_id = int(item["slot_id"])
        module_name = str(item["module_name"])
        rank_index = int(item["rank_index"])
        approved_a = approved["modules"][module_name]["lora_a"][rank_index, :]
        candidate_a = candidate["modules"][module_name]["lora_a"][rank_index, :]
        approved_b = approved["modules"][module_name]["lora_b"][:, rank_index]
        candidate_b = candidate["modules"][module_name]["lora_b"][:, rank_index]
        a_delta = candidate_a - approved_a
        b_delta = candidate_b - approved_b
        per_slot_raw[slot_id] = float(np.sum(np.square(a_delta)) + np.sum(np.square(b_delta)))

    per_slot_drift = fisher * per_slot_raw
    drift_value = float(np.sum(per_slot_drift))
    threshold = float(drift_threshold)
    return LoRADriftReport(
        drift_value=drift_value,
        drift_threshold=threshold,
        drift_within_bound=drift_value <= threshold,
        approved_snapshot_hash=_snapshot_hash(approved_snapshot),
        candidate_snapshot_hash=_snapshot_hash(candidate_snapshot),
        slot_count=slot_count,
        per_slot_drift=per_slot_drift,
        metadata={
            "unweighted_slot_delta_sum": float(np.sum(per_slot_raw)),
            "positive_fisher_slots": int(np.count_nonzero(fisher > 0)),
            "top_slots": _top_slot_payload(per_slot_drift, slot_metadata),
        },
    )


def _required_snapshot_fisher(path: str | Path, slot_count: int) -> FloatArray:
    estimate = load_snapshot_fisher(path)
    if estimate is None:
        raise ValueError("approved snapshot does not contain fisher_diagonal")
    return validate_fisher_diagonal(estimate.slot_fisher, slot_count=slot_count)


def _load_lora_snapshot(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as snapshot:
        metadata = json.loads(str(snapshot["metadata_json"].item()))
        modules: dict[str, dict[str, FloatArray]] = {}
        for index in range(len(metadata["target_modules"])):
            module_name = str(snapshot[f"module_{index}_name"].item())
            modules[module_name] = {
                "lora_a": np.asarray(snapshot[f"module_{index}_lora_a"], dtype=np.float64),
                "lora_b": np.asarray(snapshot[f"module_{index}_lora_b"], dtype=np.float64),
            }
    return {"metadata": metadata, "modules": modules}


def _validated_vector(values: Any, *, name: str, allow_zero: bool) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    if np.any(vector < 0):
        raise ValueError(f"{name} must be non-negative")
    if not allow_zero and not np.any(vector > 0):
        raise ValueError(f"{name} must contain at least one positive value")
    return vector.copy()


def _gradient_array(value: Any) -> FloatArray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("gradient tensors must be finite")
    return array


def _snapshot_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _top_slot_payload(
    per_slot_drift: FloatArray, slot_metadata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    top_indices = np.argsort(per_slot_drift)[::-1][: min(10, len(per_slot_drift))]
    payload = []
    metadata_by_slot = {int(item["slot_id"]): item for item in slot_metadata}
    for slot_id in top_indices:
        value = float(per_slot_drift[int(slot_id)])
        if value <= 0:
            continue
        metadata = metadata_by_slot[int(slot_id)]
        payload.append(
            {
                "slot_id": int(slot_id),
                "drift": value,
                "module_name": metadata["module_name"],
                "rank_index": int(metadata["rank_index"]),
            }
        )
    return payload
