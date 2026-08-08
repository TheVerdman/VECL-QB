from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_finite_vector(name: str, value: Iterable[float] | NDArray[np.floating]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN or Inf")
    return array.copy()


@dataclass(frozen=True)
class SparseMemoryInputs:
    memory_values: FloatArray
    gradients: FloatArray
    activation: FloatArray
    rarity: FloatArray
    authority: FloatArray
    learning_rate: float
    min_authority: float
    min_score: float
    max_slots: int
    quarantined_slots: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        arrays = {
            "memory_values": _as_finite_vector("memory_values", self.memory_values),
            "gradients": _as_finite_vector("gradients", self.gradients),
            "activation": _as_finite_vector("activation", self.activation),
            "rarity": _as_finite_vector("rarity", self.rarity),
            "authority": _as_finite_vector("authority", self.authority),
        }
        lengths = {name: len(array) for name, array in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"all numeric arrays must have the same length: {lengths}")
        if not np.all(
            np.isfinite(
                np.asarray(
                    [self.learning_rate, self.min_authority, self.min_score], dtype=np.float64
                )
            )
        ):
            raise ValueError("learning_rate and thresholds must be finite")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be >= 0")
        if isinstance(self.max_slots, bool) or not isinstance(self.max_slots, int):
            raise ValueError("max_slots must be an integer")
        if self.max_slots < 0:
            raise ValueError("max_slots must be >= 0")
        if self.min_authority < 0:
            raise ValueError("min_authority must be >= 0")
        if self.min_score < 0:
            raise ValueError("min_score must be >= 0")
        slot_count = next(iter(lengths.values()), 0)
        quarantined = set(self.quarantined_slots)
        if any(isinstance(slot, bool) or not isinstance(slot, int) for slot in quarantined):
            raise ValueError("quarantined slots must be integers")
        invalid = sorted(slot for slot in quarantined if slot < 0 or slot >= slot_count)
        if invalid:
            raise ValueError(f"quarantined slots out of range: {invalid}")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "quarantined_slots", quarantined)


@dataclass(frozen=True)
class DeltaRecord:
    event_id: str
    slot_id: int
    old_value: float
    new_value: float
    delta: float
    score: float
    authority: float
    provenance_root: str

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if self.slot_id < 0:
            raise ValueError("slot_id must be >= 0")
        if not self.provenance_root:
            raise ValueError("provenance_root must be non-empty")
        values = [self.old_value, self.new_value, self.delta, self.score, self.authority]
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError("DeltaRecord numeric fields must not contain NaN or Inf")


@dataclass(frozen=True)
class SparseUpdateResult:
    new_memory_values: FloatArray
    scores: FloatArray
    eligible_slots: list[int]
    selected_slots: list[int]
    delta_records: list[DeltaRecord]

    def __post_init__(self) -> None:
        new_memory_values = _as_finite_vector("new_memory_values", self.new_memory_values)
        scores = _as_finite_vector("scores", self.scores)
        if len(new_memory_values) != len(scores):
            raise ValueError("new_memory_values and scores must have the same length")
        for name, slots in {
            "eligible_slots": self.eligible_slots,
            "selected_slots": self.selected_slots,
        }.items():
            if any(slot < 0 or slot >= len(scores) for slot in slots):
                raise ValueError(f"{name} contains an out-of-range slot")
        object.__setattr__(self, "new_memory_values", new_memory_values)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "eligible_slots", list(self.eligible_slots))
        object.__setattr__(self, "selected_slots", list(self.selected_slots))
        object.__setattr__(self, "delta_records", list(self.delta_records))
