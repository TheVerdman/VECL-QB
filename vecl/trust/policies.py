from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrustPolicy:
    policy_version: str
    min_authority: float
    max_total_drift: float
    require_governance_for_zero_root: bool = True

    def __post_init__(self) -> None:
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty")
        if self.min_authority < 0:
            raise ValueError("min_authority must be >= 0")
        if self.max_total_drift < 0:
            raise ValueError("max_total_drift must be >= 0")
