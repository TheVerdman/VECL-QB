from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from vecl.provenance.events import EventType

FloatArray = NDArray[np.float64]


def activation_from_gradient_summaries(gradient_summaries: Any) -> FloatArray:
    """Normalize gradient magnitude for v0 gradient-greedy slot selection.

    Rarity and authority tilt this ranking; they are not independent activation signals yet.
    """
    values = _finite_vector(gradient_summaries, "gradient_summaries")
    max_value = float(np.max(values)) if values.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return values / max_value


def slot_selection_counts_from_ledger(
    ledger: Any, *, slot_count: int, tenant_id: str | None = None
) -> FloatArray:
    if slot_count <= 0:
        raise ValueError("slot_count must be > 0")
    counts = np.zeros(slot_count, dtype=np.float64)
    events = ledger.find_by_type(EventType.SPARSE_UPDATE_APPLIED)
    for event in events:
        if tenant_id is not None and event.tenant_id != tenant_id:
            continue
        for slot_id in event.payload.get("selected_slots", []):
            if isinstance(slot_id, int) and 0 <= slot_id < slot_count:
                counts[slot_id] += 1.0
    return counts


def rarity_from_selection_counts(counts: Any) -> FloatArray:
    values = _finite_vector(counts, "selection_counts")
    if np.any(values < 0):
        raise ValueError("selection_counts must be non-negative")
    return 1.0 / np.sqrt(1.0 + values)


def _finite_vector(values: Any, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector
