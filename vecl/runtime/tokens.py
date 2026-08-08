from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import isfinite
from uuid import uuid4

from vecl._compat import UTC


@dataclass(frozen=True)
class LearningEventToken:
    event_id: str
    batch_id: str
    tenant_id: str
    source_set_hash: str
    provenance_root_hash: str
    min_authority: float
    min_score: float
    max_slots: int
    created_at: datetime
    expires_at: datetime
    policy_version: str
    trust_policy_version: str


def create_learning_event_token(
    *,
    batch_id: str,
    tenant_id: str,
    source_set_hash: str,
    provenance_root_hash: str,
    min_authority: float,
    min_score: float,
    max_slots: int,
    policy_version: str,
    trust_policy_version: str,
    ttl_seconds: int = 300,
    now: datetime | None = None,
    event_id: str | None = None,
) -> LearningEventToken:
    now = now or datetime.now(UTC)
    return LearningEventToken(
        event_id=event_id or f"learn-{uuid4()}",
        batch_id=batch_id,
        tenant_id=tenant_id,
        source_set_hash=source_set_hash,
        provenance_root_hash=provenance_root_hash,
        min_authority=min_authority,
        min_score=min_score,
        max_slots=max_slots,
        created_at=now,
        expires_at=now + timedelta(seconds=ttl_seconds),
        policy_version=policy_version,
        trust_policy_version=trust_policy_version,
    )


def validate_learning_event_token(
    token: LearningEventToken, now: datetime | None, expected_tenant_id: str
) -> bool:
    now = now or datetime.now(UTC)
    if now.tzinfo is None or token.created_at.tzinfo is None or token.expires_at.tzinfo is None:
        raise ValueError("token timestamps must be timezone-aware")
    if token.created_at > now:
        raise ValueError("learning event token was created in the future")
    if token.expires_at <= now:
        raise ValueError("learning event token is expired")
    if token.expires_at <= token.created_at:
        raise ValueError("learning event token expiration must follow creation")
    if token.tenant_id != expected_tenant_id:
        raise ValueError("learning event token tenant mismatch")
    if not isfinite(token.min_authority) or not isfinite(token.min_score):
        raise ValueError("token thresholds must be finite")
    if token.min_authority < 0:
        raise ValueError("min_authority must be >= 0")
    if token.min_score < 0:
        raise ValueError("min_score must be >= 0")
    if isinstance(token.max_slots, bool) or not isinstance(token.max_slots, int):
        raise ValueError("max_slots must be an integer")
    if token.max_slots < 0:
        raise ValueError("max_slots must be >= 0")
    if not token.source_set_hash or not token.provenance_root_hash:
        raise ValueError("token hashes must be non-empty")
    if not token.policy_version or not token.trust_policy_version:
        raise ValueError("policy versions must be non-empty")
    if not token.event_id or not token.batch_id or not token.tenant_id:
        raise ValueError("token identifiers must be non-empty")
    return True


def token_to_audit_dict(token: LearningEventToken) -> dict[str, object]:
    audit = asdict(token)
    audit["created_at"] = token.created_at.isoformat()
    audit["expires_at"] = token.expires_at.isoformat()
    return audit
