from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vecl.qb.claim_graph import ClaimGraph
from vecl.sparse.types import SparseMemoryInputs, SparseUpdateResult


@dataclass
class MetricsCollector:
    values: dict[str, float] = field(default_factory=dict)

    def emit(self, name: str, value: float) -> None:
        self.values[name] = float(value)

    def as_dict(self) -> dict[str, float]:
        return dict(sorted(self.values.items()))


def sparse_update_metrics(
    inputs: SparseMemoryInputs, result: SparseUpdateResult
) -> dict[str, float]:
    changed = np.where(~np.isclose(inputs.memory_values, result.new_memory_values))[0]
    return {
        "sparse_update_slots_changed": float(len(changed)),
        "sparse_update_norm": float(
            np.linalg.norm(result.new_memory_values - inputs.memory_values)
        ),
    }


def adversarial_metrics(
    adversarial_source_id: str,
    replay_source_ids: list[str],
    selected_source_ids: list[str],
    memory_before: np.ndarray,
    memory_after: np.ndarray,
    trust_before: float,
    trust_after: float,
) -> dict[str, float]:
    replay_fraction = (
        replay_source_ids.count(adversarial_source_id) / len(replay_source_ids)
        if replay_source_ids
        else 0.0
    )
    selected_fraction = (
        selected_source_ids.count(adversarial_source_id) / len(selected_source_ids)
        if selected_source_ids
        else 0.0
    )
    return {
        "adversarial_source_replay_fraction": float(replay_fraction),
        "adversarial_source_selected_fraction": float(selected_fraction),
        "memory_drift_norm": float(np.linalg.norm(memory_after - memory_before)),
        "trust_drift_by_source": float(trust_after - trust_before),
    }


def claim_graph_metrics(claim_graph: ClaimGraph, verifier_statuses: list[str]) -> dict[str, float]:
    total = len(verifier_statuses) or 1
    return {
        "claim_graph_unsupported_claim_count": float(len(claim_graph.unsupported_claims())),
        "verifier_pass_rate": verifier_statuses.count("PASSED") / total,
        "conflict_review_rate": verifier_statuses.count("NEEDS_REVIEW") / total,
    }


def rollback_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    return {"rollback_error_norm": float(np.linalg.norm(actual - expected))}


def tenant_crossing_metric(attempt_count: int) -> dict[str, float]:
    return {"tenant_crossing_attempt_count": float(attempt_count)}
