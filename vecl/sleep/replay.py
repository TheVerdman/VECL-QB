from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ReplayCandidate:
    evidence_id: str
    source_id: str
    tenant_id: str
    activation: float
    rarity: float
    authority: float
    claim_ids: list[str]
    last_accessed_at: datetime
    quarantine_status: bool = False


@dataclass(frozen=True)
class ReplayPolicy:
    max_batch_size: int
    max_fraction_per_source: float
    min_authority: float
    exclude_quarantined: bool = True
    require_independent_sources: bool = False
    tenant_id: str | None = None


@dataclass(frozen=True)
class ReplayBatch:
    batch_id: str
    tenant_id: str
    candidates: list[ReplayCandidate]


class ReplayBatchBuilder:
    def build_batch(self, candidates: list[ReplayCandidate], policy: ReplayPolicy) -> ReplayBatch:
        if policy.max_batch_size < 0:
            raise ValueError("max_batch_size must be >= 0")
        if not 0 < policy.max_fraction_per_source <= 1:
            raise ValueError("max_fraction_per_source must be in (0, 1]")
        tenants = {candidate.tenant_id for candidate in candidates}
        tenant_id = policy.tenant_id
        if tenant_id is None:
            if len(tenants) > 1:
                raise ValueError("replay batch requires explicit tenant_id for mixed candidates")
            tenant_id = next(iter(tenants), "unknown")
        filtered = [
            candidate
            for candidate in candidates
            if candidate.tenant_id == tenant_id
            and candidate.authority >= policy.min_authority
            and (not policy.exclude_quarantined or not candidate.quarantine_status)
        ]
        ranked = sorted(
            filtered,
            key=lambda candidate: (
                -(candidate.activation * candidate.rarity * candidate.authority),
                candidate.source_id,
                candidate.evidence_id,
            ),
        )
        max_per_source = max(1, int(policy.max_batch_size * policy.max_fraction_per_source))
        per_source: dict[str, int] = defaultdict(int)
        selected: list[ReplayCandidate] = []
        for candidate in ranked:
            if len(selected) >= policy.max_batch_size:
                break
            if per_source[candidate.source_id] >= max_per_source:
                continue
            if policy.require_independent_sources and per_source[candidate.source_id] > 0:
                continue
            selected.append(candidate)
            per_source[candidate.source_id] += 1
        return ReplayBatch(
            batch_id=f"replay:{tenant_id}:{','.join(candidate.evidence_id for candidate in selected)}",
            tenant_id=tenant_id,
            candidates=selected,
        )
