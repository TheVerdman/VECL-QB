from __future__ import annotations

from vecl.qb.router import QBRouter
from vecl.qb.specialist import SpecialistRequest
from vecl.specialists.registry import SpecialistRegistry


def test_registry_loads_yaml_and_instantiates_specialists(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config_path = tmp_path / "specialists.yaml"
    config_path.write_text(
        """
specialists:
  - id: library-fixture
    class_path: tests.specialists.fixtures.sample_specialists:EchoLibrarySpecialist
    task_types: [echo]
    trust_anchor_id: trust-library
    cost_hint: 0.1
    latency_hint: 0.2
    version: v1
"""
    )

    registry = SpecialistRegistry.load([config_path])

    definition = registry.definitions["library-fixture"]
    specialist = registry.specialist("library-fixture")
    assert definition.task_types == {"echo"}
    assert definition.trust_anchor_id == "trust-library"
    assert specialist.specialist_id == "library-fixture"


def test_registry_registers_cards_with_router(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config_path = tmp_path / "specialists.yaml"
    config_path.write_text(
        """
specialists:
  - id: library-fixture
    class_path: tests.specialists.fixtures.sample_specialists:EchoLibrarySpecialist
    task_types: [echo]
    trust_anchor_id: trust-library
    cost_hint: 0.1
    latency_hint: 0.2
    version: v1
"""
    )
    registry = SpecialistRegistry.load([config_path])
    router = QBRouter()

    registry.register_with_router(router)

    request = SpecialistRequest("req", "tenant", "echo", {"text": "hi"}, {}, {})
    routed = router.route(request)
    assert [specialist.specialist_id for specialist in routed] == ["library-fixture"]
