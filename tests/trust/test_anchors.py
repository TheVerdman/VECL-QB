from datetime import UTC, datetime, timedelta

from vecl.trust.anchors import TrustAnchor, TrustAnchorRegistry, TrustRootKind


def _anchor(anchor_id: str, trust: float, expires_delta: int = 1) -> TrustAnchor:
    now = datetime.now(UTC)
    return TrustAnchor(
        anchor_id=anchor_id,
        source_id="source",
        root_kind=TrustRootKind.TEST_FIXTURE,
        trust_value=trust,
        issued_by="tester",
        issued_at=now - timedelta(days=1),
        expires_at=now + timedelta(days=expires_delta),
        credential_hash=f"hash-{anchor_id}",
    )


def test_expired_credential_not_counted() -> None:
    registry = TrustAnchorRegistry()
    registry.add_anchor(_anchor("a", 0.8, expires_delta=-1))
    assert registry.get_root_trust("source") == 0.0


def test_revoked_credential_not_counted() -> None:
    registry = TrustAnchorRegistry()
    registry.add_anchor(_anchor("a", 0.8))
    registry.revoke_anchor("a")
    assert registry.get_root_trust("source") == 0.0


def test_no_anchor_means_zero_trust() -> None:
    assert TrustAnchorRegistry().get_root_trust("missing") == 0.0


def test_multiple_anchors_capped_max() -> None:
    registry = TrustAnchorRegistry()
    registry.add_anchor(_anchor("a", 0.6))
    registry.add_anchor(_anchor("b", 0.9))
    assert registry.get_root_trust("source") == 0.9
