import numpy as np

from vecl.sparse.cuda_like import sparse_update_cuda_like
from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs


def test_cuda_like_matches_cpu_oracle_exact_boundary() -> None:
    inputs = SparseMemoryInputs(
        memory_values=np.array([1.0, 2.0, 3.0]),
        gradients=np.array([1.0, 0.5, 2.0]),
        activation=np.array([1.0, 1.0, 1.0]),
        rarity=np.array([1.0, 2.0, 1.0]),
        authority=np.array([1.0, 0.5, 0.0]),
        learning_rate=0.1,
        min_authority=0.1,
        min_score=0.1,
        max_slots=2,
        quarantined_slots=set(),
    )
    cpu = sparse_update_oracle(inputs, "evt", "root")
    cuda_like = sparse_update_cuda_like(inputs, "evt", "root")
    assert cuda_like.selected_slots == cpu.selected_slots
    assert cuda_like.eligible_slots == cpu.eligible_slots
    assert np.allclose(cuda_like.new_memory_values, cpu.new_memory_values)
