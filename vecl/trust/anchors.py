from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vecl._compat import UTC, StrEnum


class TrustRootKind(StrEnum):
    REGULATORY_CREDENTIAL = "REGULATORY_CREDENTIAL"
    GOVERNANCE_APPROVAL = "GOVERNANCE_APPROVAL"
    VERIFIED_OPERATIONAL_RECORD = "VERIFIED_OPERATIONAL_RECORD"
    HUMAN_EXPERT = "HUMAN_EXPERT"
    SYSTEM_POLICY = "SYSTEM_POLICY"
    TEST_FIXTURE = "TEST_FIXTURE"


@dataclass(frozen=True)
class TrustAnchor:
    anchor_id: str
    source_id: str
    root_kind: TrustRootKind
    trust_value: float
    issued_by: str
    issued_at: datetime
    expires_at: datetime
    credential_hash: str
    revocation_status: str = "active"

    def __post_init__(self) -> None:
        if (
            not self.anchor_id
            or not self.source_id
            or not self.issued_by
            or not self.credential_hash
        ):
            raise ValueError("trust anchor identifiers must be non-empty")
        if not 0 <= self.trust_value <= 1:
            raise ValueError("trust_value must be in [0, 1]")

    def is_active(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(UTC)
        return self.revocation_status != "revoked" and self.expires_at >= now


@dataclass
class TrustAnchorRegistry:
    anchors: dict[str, TrustAnchor] = field(default_factory=dict)

    def add_anchor(self, anchor: TrustAnchor) -> None:
        self.anchors[anchor.anchor_id] = anchor

    def revoke_anchor(self, anchor_id: str) -> None:
        anchor = self.anchors[anchor_id]
        self.anchors[anchor_id] = TrustAnchor(
            anchor_id=anchor.anchor_id,
            source_id=anchor.source_id,
            root_kind=anchor.root_kind,
            trust_value=anchor.trust_value,
            issued_by=anchor.issued_by,
            issued_at=anchor.issued_at,
            expires_at=anchor.expires_at,
            credential_hash=anchor.credential_hash,
            revocation_status="revoked",
        )

    def active_anchors_for_source(
        self, source_id: str, now: datetime | None = None
    ) -> list[TrustAnchor]:
        return [
            anchor
            for anchor in self.anchors.values()
            if anchor.source_id == source_id and anchor.is_active(now)
        ]

    def get_root_trust(self, source_id: str, now: datetime | None = None) -> float:
        active = self.active_anchors_for_source(source_id, now)
        if not active:
            return 0.0
        return min(1.0, max(anchor.trust_value for anchor in active))

    def explain_root_trust(self, source_id: str, now: datetime | None = None) -> dict[str, object]:
        active = self.active_anchors_for_source(source_id, now)
        return {
            "source_id": source_id,
            "root_trust": self.get_root_trust(source_id, now),
            "rule": "capped max",
            "anchors": [anchor.anchor_id for anchor in active],
        }
