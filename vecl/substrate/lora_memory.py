from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vecl.sparse.types import SparseUpdateResult

FloatArray = NDArray[np.float64]

_LAYER_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class LoRASlotMetadata:
    slot_id: int
    adapter_name: str
    module_name: str
    layer_index: int | None
    rank_index: int
    lora_a_shape: tuple[int, int]
    lora_b_shape: tuple[int, int]


@dataclass(frozen=True)
class LoRARawGradient:
    slot_id: int
    lora_a_row: Any
    lora_b_column: Any


@dataclass(frozen=True)
class LoRATensorDeltaRecord:
    slot_id: int
    module_name: str
    rank_index: int
    authority: float
    lora_a_delta_norm: float
    lora_b_delta_norm: float
    total_delta_norm: float
    before_tensor_hash: str
    after_tensor_hash: str


@dataclass(frozen=True)
class _LoRASlotBinding:
    metadata: LoRASlotMetadata
    lora_module: Any


class LoRAMemorySubstrate:
    """Expose PEFT LoRA rank components as VECL sparse memory slots.

    The sparse oracle ranks per-slot summaries. The substrate then applies the
    real tensor update to the selected LoRA A row and paired LoRA B column.
    """

    def __init__(self, model: Any, adapter_name: str, slots: Sequence[_LoRASlotBinding]) -> None:
        if not slots:
            raise ValueError("LoRAMemorySubstrate requires at least one LoRA slot")
        self.model = model
        self.adapter_name = adapter_name
        self._slots = list(slots)
        self._slot_by_id = {slot.metadata.slot_id: slot for slot in self._slots}
        if len(self._slot_by_id) != len(self._slots):
            raise ValueError("slot ids must be unique")

    @classmethod
    def attach(
        cls,
        model: Any,
        *,
        rank: int = 4,
        adapter_name: str = "vecl_sparse",
        target_module_suffix: str = "mlp.up_proj",
        last_n_layers: int = 8,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
    ) -> LoRAMemorySubstrate:
        if rank <= 0:
            raise ValueError("rank must be > 0")
        if last_n_layers <= 0:
            raise ValueError("last_n_layers must be > 0")

        target_modules = _discover_target_modules(model, target_module_suffix, last_n_layers)
        if not target_modules:
            raise ValueError(f"no target modules found for suffix {target_module_suffix!r}")

        peft = _import_peft()
        for parameter in model.parameters():
            parameter.requires_grad = False

        config = peft.LoraConfig(
            r=rank,
            lora_alpha=lora_alpha if lora_alpha is not None else rank,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        peft_model = peft.get_peft_model(model, config, adapter_name=adapter_name)
        _freeze_base_enable_lora(peft_model, adapter_name)
        return cls.from_peft_model(peft_model, adapter_name=adapter_name)

    @classmethod
    def from_peft_model(
        cls,
        model: Any,
        *,
        adapter_name: str = "vecl_sparse",
        target_module_suffix: str = "mlp.up_proj",
    ) -> LoRAMemorySubstrate:
        slots: list[_LoRASlotBinding] = []
        modules = _iter_lora_modules(model, adapter_name, target_module_suffix)
        for module_name, layer_index, lora_module in modules:
            lora_a = _lora_a_weight(lora_module, adapter_name)
            lora_b = _lora_b_weight(lora_module, adapter_name)
            a_shape = tuple(int(value) for value in lora_a.shape)
            b_shape = tuple(int(value) for value in lora_b.shape)
            if len(a_shape) != 2 or len(b_shape) != 2:
                raise ValueError(f"LoRA tensors for {module_name} must be two-dimensional")
            if a_shape[0] != b_shape[1]:
                raise ValueError(
                    f"LoRA rank mismatch for {module_name}: A shape {a_shape}, B shape {b_shape}"
                )
            for rank_index in range(a_shape[0]):
                slot_id = len(slots)
                metadata = LoRASlotMetadata(
                    slot_id=slot_id,
                    adapter_name=adapter_name,
                    module_name=module_name,
                    layer_index=layer_index,
                    rank_index=rank_index,
                    lora_a_shape=a_shape,
                    lora_b_shape=b_shape,
                )
                slots.append(_LoRASlotBinding(metadata=metadata, lora_module=lora_module))
        return cls(model=model, adapter_name=adapter_name, slots=slots)

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    @property
    def slot_metadata(self) -> list[LoRASlotMetadata]:
        return [slot.metadata for slot in self._slots]

    def flatten(self) -> FloatArray:
        values = np.array(
            [_component_norm(*self._slot_tensors(slot)) for slot in self._slots],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("LoRA slot summaries must not contain NaN or Inf")
        return values

    def gradient_summaries(
        self, raw_gradients: Mapping[int, LoRARawGradient] | Iterable[LoRARawGradient]
    ) -> FloatArray:
        gradients = _normalize_raw_gradients(raw_gradients)
        values = np.zeros(self.slot_count, dtype=np.float64)
        for slot_id, raw_gradient in gradients.items():
            slot = self._slot_by_id.get(slot_id)
            if slot is None:
                raise ValueError(f"unknown LoRA slot id {slot_id}")
            grad_a, grad_b = self._validated_gradient_tensors(slot, raw_gradient)
            values[slot_id] = _component_norm(grad_a, grad_b)
        if not np.all(np.isfinite(values)):
            raise ValueError("LoRA gradient summaries must not contain NaN or Inf")
        return values

    def raw_gradients_from_current_grads(self) -> dict[int, LoRARawGradient]:
        gradients: dict[int, LoRARawGradient] = {}
        for slot in self._slots:
            metadata = slot.metadata
            lora_a = _lora_a_weight(slot.lora_module, metadata.adapter_name)
            lora_b = _lora_b_weight(slot.lora_module, metadata.adapter_name)
            if lora_a.grad is None:
                raise ValueError(f"missing LoRA A gradient for slot {metadata.slot_id}")
            if lora_b.grad is None:
                raise ValueError(f"missing LoRA B gradient for slot {metadata.slot_id}")
            gradients[metadata.slot_id] = LoRARawGradient(
                slot_id=metadata.slot_id,
                lora_a_row=lora_a.grad[metadata.rank_index, :].detach().clone(),
                lora_b_column=lora_b.grad[:, metadata.rank_index].detach().clone(),
            )
        return gradients

    def apply_update(
        self,
        result: SparseUpdateResult,
        raw_gradients: Mapping[int, LoRARawGradient] | Iterable[LoRARawGradient],
        *,
        learning_rate: float,
    ) -> list[LoRATensorDeltaRecord]:
        if learning_rate < 0:
            raise ValueError("learning_rate must be >= 0")

        torch = _import_torch()
        gradients = _normalize_raw_gradients(raw_gradients)
        records_by_slot = {record.slot_id: record for record in result.delta_records}
        delta_records: list[LoRATensorDeltaRecord] = []

        for slot_id in result.selected_slots:
            slot = self._slot_by_id.get(slot_id)
            if slot is None:
                raise ValueError(f"selected unknown LoRA slot id {slot_id}")
            oracle_record = records_by_slot.get(slot_id)
            if oracle_record is None:
                raise ValueError(f"selected slot {slot_id} has no oracle DeltaRecord")
            raw_gradient = gradients.get(slot_id)
            if raw_gradient is None:
                raise ValueError(f"missing raw gradient for selected slot {slot_id}")

            grad_a, grad_b = self._validated_gradient_tensors(slot, raw_gradient)
            a_row, b_column = self._slot_tensors(slot)
            before_hash = _component_hash(a_row, b_column)
            scale = float(learning_rate * oracle_record.authority)
            update_a = grad_a.to(device=a_row.device, dtype=a_row.dtype) * scale
            update_b = grad_b.to(device=b_column.device, dtype=b_column.dtype) * scale

            with torch.no_grad():
                a_row.sub_(update_a)
                b_column.sub_(update_b)

            after_a_row, after_b_column = self._slot_tensors(slot)
            after_hash = _component_hash(after_a_row, after_b_column)
            a_delta_norm = _tensor_norm(update_a)
            b_delta_norm = _tensor_norm(update_b)
            metadata = slot.metadata
            delta_records.append(
                LoRATensorDeltaRecord(
                    slot_id=slot_id,
                    module_name=metadata.module_name,
                    rank_index=metadata.rank_index,
                    authority=float(oracle_record.authority),
                    lora_a_delta_norm=a_delta_norm,
                    lora_b_delta_norm=b_delta_norm,
                    total_delta_norm=float(np.sqrt(a_delta_norm**2 + b_delta_norm**2)),
                    before_tensor_hash=before_hash,
                    after_tensor_hash=after_hash,
                )
            )

        return delta_records

    def snapshot(
        self,
        path: str | Path,
        *,
        fisher_diagonal: Any | None = None,
        fisher_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        modules = self._unique_modules()
        metadata = self._snapshot_metadata()
        arrays: dict[str, NDArray[Any]] = {"metadata_json": np.array(json.dumps(metadata))}
        if fisher_diagonal is None:
            fisher = np.zeros(self.slot_count, dtype=np.float64)
            fisher_payload = {"fisher_status": "not_accumulated", "sample_count": 0}
        else:
            fisher = np.asarray(fisher_diagonal, dtype=np.float64)
            if fisher.shape != (self.slot_count,):
                raise ValueError(f"fisher_diagonal must have shape ({self.slot_count},)")
            if not np.all(np.isfinite(fisher)) or np.any(fisher < 0):
                raise ValueError("fisher_diagonal must be finite and non-negative")
            fisher_payload = dict(fisher_metadata or {})
            fisher_payload.setdefault("fisher_status", "accumulated")
        arrays["fisher_diagonal"] = fisher
        arrays["fisher_metadata_json"] = np.array(json.dumps(fisher_payload, sort_keys=True))
        for index, (module_name, lora_module) in enumerate(modules.items()):
            arrays[f"module_{index}_name"] = np.array(module_name)
            arrays[f"module_{index}_lora_a"] = _tensor_to_numpy(
                _lora_a_weight(lora_module, self.adapter_name)
            )
            arrays[f"module_{index}_lora_b"] = _tensor_to_numpy(
                _lora_b_weight(lora_module, self.adapter_name)
            )
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]

    def restore(self, path: str | Path) -> None:
        path = Path(path)
        with np.load(path, allow_pickle=False) as snapshot:
            metadata = json.loads(str(snapshot["metadata_json"].item()))
            current_metadata = self._snapshot_metadata()
            if metadata != current_metadata:
                raise ValueError("LoRA snapshot metadata is incompatible with this substrate")

            modules = self._unique_modules()
            pending: list[tuple[Any, NDArray[Any], NDArray[Any]]] = []
            for index, (module_name, lora_module) in enumerate(modules.items()):
                saved_name = str(snapshot[f"module_{index}_name"].item())
                if saved_name != module_name:
                    raise ValueError("LoRA snapshot module order is incompatible")
                lora_a = np.asarray(snapshot[f"module_{index}_lora_a"])
                lora_b = np.asarray(snapshot[f"module_{index}_lora_b"])
                current_a = _lora_a_weight(lora_module, self.adapter_name)
                current_b = _lora_b_weight(lora_module, self.adapter_name)
                if tuple(lora_a.shape) != tuple(current_a.shape):
                    raise ValueError(f"LoRA A shape mismatch for {module_name}")
                if tuple(lora_b.shape) != tuple(current_b.shape):
                    raise ValueError(f"LoRA B shape mismatch for {module_name}")
                if not np.all(np.isfinite(lora_a)) or not np.all(np.isfinite(lora_b)):
                    raise ValueError(f"LoRA snapshot tensors for {module_name} must be finite")
                pending.append((lora_module, lora_a, lora_b))

        torch = _import_torch()
        with torch.no_grad():
            for lora_module, lora_a, lora_b in pending:
                current_a = _lora_a_weight(lora_module, self.adapter_name)
                current_b = _lora_b_weight(lora_module, self.adapter_name)
                current_a.copy_(
                    torch.as_tensor(lora_a, device=current_a.device, dtype=current_a.dtype)
                )
                current_b.copy_(
                    torch.as_tensor(lora_b, device=current_b.device, dtype=current_b.dtype)
                )

    def fisher_weighted_drift_penalty(
        self,
        approved_snapshot: str | Path,
        *,
        fisher_diagonal: Any | None = None,
    ) -> Any:
        """Return differentiable Fisher-weighted LoRA drift from an approved snapshot."""

        path = Path(approved_snapshot)
        with np.load(path, allow_pickle=False) as snapshot:
            metadata = json.loads(str(snapshot["metadata_json"].item()))
            current_metadata = self._snapshot_metadata()
            if metadata != current_metadata:
                raise ValueError("LoRA snapshot metadata is incompatible with this substrate")
            fisher = (
                _validated_fisher_diagonal(fisher_diagonal, self.slot_count)
                if fisher_diagonal is not None
                else _snapshot_fisher_diagonal(snapshot, self.slot_count)
            )
            approved_modules: dict[str, tuple[NDArray[Any], NDArray[Any]]] = {}
            modules = self._unique_modules()
            for index, module_name in enumerate(modules):
                saved_name = str(snapshot[f"module_{index}_name"].item())
                if saved_name != module_name:
                    raise ValueError("LoRA snapshot module order is incompatible")
                approved_modules[module_name] = (
                    np.asarray(snapshot[f"module_{index}_lora_a"]),
                    np.asarray(snapshot[f"module_{index}_lora_b"]),
                )

        torch = _import_torch()
        penalty = None
        for slot in self._slots:
            metadata = slot.metadata
            a_row, b_column = self._slot_tensors(slot)
            approved_a, approved_b = approved_modules[metadata.module_name]
            approved_a_row = torch.as_tensor(
                approved_a[metadata.rank_index, :], device=a_row.device, dtype=a_row.dtype
            )
            approved_b_column = torch.as_tensor(
                approved_b[:, metadata.rank_index], device=b_column.device, dtype=b_column.dtype
            )
            weight = torch.as_tensor(
                float(fisher[metadata.slot_id]), device=a_row.device, dtype=torch.float32
            )
            slot_penalty = weight * (
                torch.sum(torch.square(a_row.float() - approved_a_row.float()))
                + torch.sum(torch.square(b_column.float() - approved_b_column.float()))
            )
            penalty = slot_penalty if penalty is None else penalty + slot_penalty
        if penalty is None:  # pragma: no cover - constructor rejects empty slot lists.
            return torch.zeros((), dtype=torch.float32)
        return penalty

    def _slot_tensors(self, slot: _LoRASlotBinding) -> tuple[Any, Any]:
        metadata = slot.metadata
        lora_a = _lora_a_weight(slot.lora_module, metadata.adapter_name)
        lora_b = _lora_b_weight(slot.lora_module, metadata.adapter_name)
        return lora_a[metadata.rank_index, :], lora_b[:, metadata.rank_index]

    def _validated_gradient_tensors(
        self, slot: _LoRASlotBinding, raw_gradient: LoRARawGradient
    ) -> tuple[Any, Any]:
        if raw_gradient.slot_id != slot.metadata.slot_id:
            raise ValueError(
                f"raw gradient slot id {raw_gradient.slot_id} does not match slot "
                f"{slot.metadata.slot_id}"
            )
        torch = _import_torch()
        expected_a, expected_b = self._slot_tensors(slot)
        grad_a = torch.as_tensor(raw_gradient.lora_a_row)
        grad_b = torch.as_tensor(raw_gradient.lora_b_column)
        if tuple(grad_a.shape) != tuple(expected_a.shape):
            raise ValueError(
                f"LoRA A gradient shape mismatch for slot {slot.metadata.slot_id}: "
                f"expected {tuple(expected_a.shape)}, got {tuple(grad_a.shape)}"
            )
        if tuple(grad_b.shape) != tuple(expected_b.shape):
            raise ValueError(
                f"LoRA B gradient shape mismatch for slot {slot.metadata.slot_id}: "
                f"expected {tuple(expected_b.shape)}, got {tuple(grad_b.shape)}"
            )
        if not bool(torch.isfinite(grad_a).all()) or not bool(torch.isfinite(grad_b).all()):
            raise ValueError(f"raw gradients for slot {slot.metadata.slot_id} must be finite")
        return grad_a, grad_b

    def _unique_modules(self) -> dict[str, Any]:
        modules: dict[str, Any] = {}
        for slot in self._slots:
            modules.setdefault(slot.metadata.module_name, slot.lora_module)
        return modules

    def _snapshot_metadata(self) -> dict[str, Any]:
        module_names = list(self._unique_modules())
        ranks = sorted({slot.metadata.lora_a_shape[0] for slot in self._slots})
        slot_metadata = []
        for metadata in self.slot_metadata:
            item = asdict(metadata)
            item["lora_a_shape"] = list(metadata.lora_a_shape)
            item["lora_b_shape"] = list(metadata.lora_b_shape)
            slot_metadata.append(item)
        return {
            "model_class_name": self.model.__class__.__qualname__,
            "adapter_name": self.adapter_name,
            "rank": ranks[0] if len(ranks) == 1 else ranks,
            "target_modules": module_names,
            "layer_indices": [slot.metadata.layer_index for slot in self._slots],
            "slot_count": self.slot_count,
            "slot_metadata": slot_metadata,
        }


def _import_peft() -> Any:
    try:
        import peft
    except ImportError as exc:  # pragma: no cover - exercised when optional deps absent.
        raise ImportError(
            "LoRAMemorySubstrate requires the substrate extra: pip install -e '.[substrate]'"
        ) from exc
    return peft


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when optional deps absent.
        raise ImportError(
            "LoRAMemorySubstrate requires the substrate extra: pip install -e '.[substrate]'"
        ) from exc
    return torch


def _discover_target_modules(
    model: Any, target_module_suffix: str, last_n_layers: int
) -> list[str]:
    suffix = target_module_suffix.removeprefix(".")
    candidates: list[tuple[int | None, str]] = []
    for name, _module in model.named_modules():
        if name.endswith(suffix):
            candidates.append((_layer_index(name), name))
    if not candidates:
        return []
    indexed_layers = sorted({layer for layer, _name in candidates if layer is not None})
    if indexed_layers:
        selected_layers = set(indexed_layers[-last_n_layers:])
        candidates = [
            (layer, name)
            for layer, name in candidates
            if layer is not None and layer in selected_layers
        ]
    return [
        name
        for _layer, name in sorted(candidates, key=lambda item: (_sort_layer(item[0]), item[1]))
    ]


def _iter_lora_modules(
    model: Any, adapter_name: str, target_module_suffix: str
) -> list[tuple[str, int | None, Any]]:
    suffix = target_module_suffix.removeprefix(".")
    modules: list[tuple[str, int | None, Any]] = []
    for name, module in model.named_modules():
        if not name.endswith(suffix):
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        lora_a = module.lora_A
        lora_b = module.lora_B
        if adapter_name not in lora_a or adapter_name not in lora_b:
            continue
        canonical_name = _canonical_module_name(name)
        modules.append((canonical_name, _layer_index(canonical_name), module))
    modules.sort(key=lambda item: (_sort_layer(item[1]), item[0]))
    return modules


def _freeze_base_enable_lora(model: Any, adapter_name: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            f"lora_A.{adapter_name}" in name or f"lora_B.{adapter_name}" in name
        )


def _lora_a_weight(lora_module: Any, adapter_name: str) -> Any:
    return lora_module.lora_A[adapter_name].weight


def _lora_b_weight(lora_module: Any, adapter_name: str) -> Any:
    return lora_module.lora_B[adapter_name].weight


def _layer_index(module_name: str) -> int | None:
    match = _LAYER_RE.search(module_name)
    return int(match.group(1)) if match else None


def _canonical_module_name(module_name: str) -> str:
    while module_name.startswith("base_model.model."):
        module_name = module_name.removeprefix("base_model.model.")
    return module_name


def _sort_layer(layer_index: int | None) -> int:
    return layer_index if layer_index is not None else 10**9


def _normalize_raw_gradients(
    raw_gradients: Mapping[int, LoRARawGradient] | Iterable[LoRARawGradient],
) -> dict[int, LoRARawGradient]:
    if isinstance(raw_gradients, Mapping):
        gradients = dict(raw_gradients)
    else:
        gradients = {gradient.slot_id: gradient for gradient in raw_gradients}
    for slot_id, raw_gradient in gradients.items():
        if slot_id != raw_gradient.slot_id:
            raise ValueError(f"raw gradient mapping key {slot_id} does not match gradient slot id")
    return gradients


def _component_norm(lora_a_row: Any, lora_b_column: Any) -> float:
    torch = _import_torch()
    a_norm = torch.linalg.vector_norm(lora_a_row.detach().double())
    b_norm = torch.linalg.vector_norm(lora_b_column.detach().double())
    return float(torch.sqrt(a_norm.square() + b_norm.square()).cpu().item())


def _tensor_norm(tensor: Any) -> float:
    torch = _import_torch()
    return float(torch.linalg.vector_norm(tensor.detach().double()).cpu().item())


def _component_hash(lora_a_row: Any, lora_b_column: Any) -> str:
    digest = hashlib.sha256()
    for tensor in (lora_a_row, lora_b_column):
        array = _tensor_to_numpy(tensor)
        digest.update(json.dumps({"shape": array.shape, "dtype": str(array.dtype)}).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tensor_to_numpy(tensor: Any) -> NDArray[Any]:
    return tensor.detach().cpu().contiguous().numpy()


def _snapshot_fisher_diagonal(snapshot: Any, slot_count: int) -> FloatArray:
    if "fisher_diagonal" not in snapshot:
        raise ValueError("approved snapshot does not contain fisher_diagonal")
    if "fisher_metadata_json" in snapshot:
        metadata = json.loads(str(snapshot["fisher_metadata_json"].item()))
        if metadata.get("fisher_status") == "not_accumulated":
            raise ValueError("approved snapshot fisher_diagonal is not accumulated")
    return _validated_fisher_diagonal(snapshot["fisher_diagonal"], slot_count)


def _validated_fisher_diagonal(values: Any, slot_count: int) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (slot_count,):
        raise ValueError(f"fisher_diagonal must have shape ({slot_count},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("fisher_diagonal must be finite")
    if np.any(vector < 0):
        raise ValueError("fisher_diagonal must be non-negative")
    return vector.copy()
