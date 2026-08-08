from datetime import datetime, timedelta
from math import nan

import pytest

from vecl._compat import UTC
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


def test_token_is_rejected_at_exact_expiration() -> None:
    token = _token()
    now = token.expires_at
    with pytest.raises(ValueError, match="expired"):
        validate_learning_event_token(token, now, "tenant-a")


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
    with pytest.raises(ValueError, match="finite"):
        validate_learning_event_token(_token(min_score=nan), datetime.now(UTC), "tenant-a")


def test_malformed_token_timestamps_fail_closed() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_learning_event_token(_token(created_at=now.replace(tzinfo=None)), now, "tenant-a")
    with pytest.raises(ValueError, match="future"):
        validate_learning_event_token(
            _token(created_at=now + timedelta(seconds=1)), now, "tenant-a"
        )
