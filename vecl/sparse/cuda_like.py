from __future__ import annotations

from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs, SparseUpdateResult


def sparse_update_cuda_like(
    inputs: SparseMemoryInputs,
    event_id: str = "cuda-like-event",
    provenance_root: str = "cuda-like-root",
) -> SparseUpdateResult:
    """Non-accelerated API scaffold that delegates to the CPU oracle.

    This function is not an independent implementation and its tests are not
    differential accelerator validation. Future CUDA kernels must remain
    data-plane-only and be independently tested against the oracle.
    """

    return sparse_update_oracle(inputs, event_id, provenance_root)
