from __future__ import annotations

from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs, SparseUpdateResult


def sparse_update_cuda_like(
    inputs: SparseMemoryInputs,
    event_id: str = "cuda-like-event",
    provenance_root: str = "cuda-like-root",
) -> SparseUpdateResult:
    """Optional accelerator API boundary.

    v0 deliberately delegates to the CPU oracle. Future CUDA kernels must remain
    data-plane-only and be differentially tested against this exact result.
    """

    return sparse_update_oracle(inputs, event_id, provenance_root)
