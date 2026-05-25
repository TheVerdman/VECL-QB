"""Sparse memory update primitives."""

from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.topk import deterministic_top_k
from vecl.sparse.types import DeltaRecord, SparseMemoryInputs, SparseUpdateResult

__all__ = [
    "DeltaRecord",
    "SparseMemoryInputs",
    "SparseUpdateResult",
    "deterministic_top_k",
    "sparse_update_oracle",
]
