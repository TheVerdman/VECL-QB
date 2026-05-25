from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from vecl._compat import UTC


@dataclass(frozen=True)
class SpecialistRequest:
    request_id: str
    tenant_id: str
    task_type: str
    input_payload: dict[str, Any]
    required_output_schema: dict[str, Any]
    provenance_context: dict[str, Any]


@dataclass(frozen=True)
class SpecialistClaim:
    claim_id: str
    specialist_id: str
    claim_text: str
    claim_type: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    high_risk: bool = False

    def __post_init__(self) -> None:
        if (
            not self.claim_id
            or not self.specialist_id
            or not self.claim_text
            or not self.claim_type
        ):
            raise ValueError("claim identifiers and text must be non-empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if not self.evidence_ids and not self.assumptions:
            raise ValueError("every claim must have evidence or an explicit assumption")


@dataclass(frozen=True)
class SpecialistResponse:
    request_id: str
    specialist_id: str
    claims: list[SpecialistClaim]
    tenant_id: str
    refusal_or_error: str | None = None
    cost_metadata: dict[str, Any] | None = None


class Specialist(ABC):
    specialist_id: str

    @abstractmethod
    def can_handle(self, request: SpecialistRequest) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        raise NotImplementedError


class MockSpecialist(Specialist):
    def __init__(
        self,
        specialist_id: str,
        task_types: set[str],
        claims: list[SpecialistClaim] | None = None,
    ) -> None:
        self.specialist_id = specialist_id
        self.task_types = set(task_types)
        self._claims = claims or []

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type in self.task_types

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        claims = [
            SpecialistClaim(
                claim_id=claim.claim_id,
                specialist_id=self.specialist_id,
                claim_text=claim.claim_text,
                claim_type=claim.claim_type,
                confidence=claim.confidence,
                evidence_ids=list(claim.evidence_ids),
                source_ids=list(claim.source_ids),
                artifact_ids=list(claim.artifact_ids),
                assumptions=list(claim.assumptions),
                limitations=list(claim.limitations),
                created_at=claim.created_at,
                high_risk=claim.high_risk,
            )
            for claim in self._claims
        ]
        return SpecialistResponse(
            request_id=request.request_id,
            specialist_id=self.specialist_id,
            claims=claims,
            tenant_id=request.tenant_id,
        )
