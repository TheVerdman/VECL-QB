import numpy as np

from vecl.sparse import SparseMemoryInputs, sparse_update_oracle


def test_smoke_sparse_oracle_discovers() -> None:
    inputs = SparseMemoryInputs(
        memory_values=np.array([1.0, 2.0]),
        gradients=np.array([0.5, 1.0]),
        activation=np.array([1.0, 1.0]),
        rarity=np.array([1.0, 1.0]),
        authority=np.array([1.0, 0.0]),
        learning_rate=0.1,
        min_authority=0.1,
        min_score=0.1,
        max_slots=1,
        quarantined_slots=set(),
    )
    result = sparse_update_oracle(inputs, "evt", "root")
    assert result.selected_slots == [0]
    assert result.new_memory_values[0] == 0.95
