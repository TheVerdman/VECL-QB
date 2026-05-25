from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def deterministic_top_k(
    scores: Iterable[float] | np.ndarray, eligible_slots: Iterable[int], k: int
) -> list[int]:
    """Return deterministic TopK by score descending and slot id ascending."""

    if k < 0:
        raise ValueError("k must be >= 0")
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must not contain NaN or Inf")
    if k == 0:
        return []
    unique_slots = sorted(set(eligible_slots))
    if not unique_slots:
        return []
    invalid = [slot for slot in unique_slots if slot < 0 or slot >= len(score_array)]
    if invalid:
        raise ValueError(f"eligible_slots contains out-of-range slots: {invalid}")
    ranked = sorted(unique_slots, key=lambda slot: (-float(score_array[slot]), slot))
    return ranked[:k]
