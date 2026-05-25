from datetime import UTC, datetime, timedelta

import pytest

from vecl.runtime.tokens import (
    LearningEventToken,
    create_learning_event_token,
    validate_learning_event_token,
)


def _token(**overrides: object) -> LearningEventToken:
    now = datetime.now(UTC)
    token = create_learning_event_token(
        batch_id="batch",
        tenant_id="tenant-a",
        source_set_hash="sources",
        provenance_root_hash="root",
        min_authority=0.1,
        min_score=0.2,
        max_slots=3,
        policy_version="p1",
        trust_policy_version="t1",
        now=now,
    )
    data = token.__dict__ | overrides
    return LearningEventToken(**data)


def test_expired_token_rejected() -> None:
    token = _token(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    with pytest.raises(ValueError, match="expired"):
        validate_learning_event_token(token, datetime.now(UTC), "tenant-a")


def test_tenant_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="tenant"):
        validate_learning_event_token(_token(), datetime.now(UTC), "tenant-b")


def test_missing_hashes_rejected() -> None:
    with pytest.raises(ValueError, match="hashes"):
        validate_learning_event_token(_token(source_set_hash=""), datetime.now(UTC), "tenant-a")


def test_invalid_thresholds_rejected() -> None:
    with pytest.raises(ValueError, match="min_authority"):
        validate_learning_event_token(_token(min_authority=-0.1), datetime.now(UTC), "tenant-a")
    with pytest.raises(ValueError, match="max_slots"):
        validate_learning_event_token(_token(max_slots=-1), datetime.now(UTC), "tenant-a")
