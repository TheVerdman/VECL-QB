import pytest

from vecl.qb.router import QBRouter, SpecialistCard
from vecl.qb.specialist import MockSpecialist, SpecialistClaim, SpecialistRequest


def _request(task_type: str = "rate") -> SpecialistRequest:
    return SpecialistRequest("req", "tenant", task_type, {}, {}, {})


def test_every_claim_has_evidence_or_assumption() -> None:
    with pytest.raises(ValueError, match="evidence"):
        SpecialistClaim("c", "s", "unsupported", "rate", 0.8)


def test_tenant_id_carried_through_mock_specialist() -> None:
    specialist = MockSpecialist(
        "s",
        {"rate"},
        [SpecialistClaim("c", "s", "rate is fair", "rate", 0.8, evidence_ids=["e"])],
    )
    response = specialist.run(_request())
    assert response.tenant_id == "tenant"


def test_route_to_correct_specialist() -> None:
    router = QBRouter()
    specialist = MockSpecialist("s", {"rate"})
    router.register_specialist(
        SpecialistCard("s", {"rate"}, {}, 1.0, 1.0, "v1", effective_trust=0.5), specialist
    )
    assert router.route(_request("rate")) == [specialist]


def test_deterministic_tie_breaking() -> None:
    router = QBRouter()
    b = MockSpecialist("b", {"rate"})
    a = MockSpecialist("a", {"rate"})
    router.register_specialist(
        SpecialistCard("b", {"rate"}, {}, 1.0, 1.0, "v1", effective_trust=0.5), b
    )
    router.register_specialist(
        SpecialistCard("a", {"rate"}, {}, 1.0, 1.0, "v1", effective_trust=0.5), a
    )
    assert [specialist.specialist_id for specialist in router.route(_request("rate"))] == ["a", "b"]


def test_unhandled_task() -> None:
    assert QBRouter().route(_request("unknown")) == []


def test_max_specialists() -> None:
    router = QBRouter(max_specialists=1)
    for sid in ["a", "b"]:
        router.register_specialist(
            SpecialistCard(sid, {"rate"}, {}, 1.0, 1.0, "v1", effective_trust=0.5),
            MockSpecialist(sid, {"rate"}),
        )
    assert len(router.route(_request("rate"))) == 1
