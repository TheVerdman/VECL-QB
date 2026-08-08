from unittest.mock import patch

import numpy as np

from vecl.sparse.cuda_like import sparse_update_cuda_like
from vecl.sparse.oracle import sparse_update_oracle
from vecl.sparse.types import SparseMemoryInputs


def test_cuda_like_boundary_is_explicit_cpu_oracle_scaffold() -> None:
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
    expected = sparse_update_oracle(inputs, "evt", "root")

    with patch("vecl.sparse.cuda_like.sparse_update_oracle", return_value=expected) as oracle:
        actual = sparse_update_cuda_like(inputs, "evt", "root")

    oracle.assert_called_once_with(inputs, "evt", "root")
    assert actual is expected
