from __future__ import annotations

from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs, SparseUpdateResult


def apply_sparse_update(
    inputs: SparseMemoryInputs, event_id: str, provenance_root: str
) -> SparseUpdateResult:
    """Production-facing wrapper for the CPU oracle in v0."""

    return sparse_update_oracle(inputs, event_id, provenance_root)
