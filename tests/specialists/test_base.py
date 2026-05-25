from __future__ import annotations

from tests.specialists.fixtures.sample_specialists import (
    EchoLibrarySpecialist,
    EchoServiceSpecialist,
    EchoSubprocessSpecialist,
)
from vecl.qb.specialist import SpecialistRequest


def _request(task_type: str = "echo", text: str = "hello") -> SpecialistRequest:
    return SpecialistRequest(
        request_id="req",
        tenant_id="tenant",
        task_type=task_type,
        input_payload={"text": text},
        required_output_schema={},
        provenance_context={},
    )


def test_subprocess_specialist_runs_cli_and_parses_files() -> None:
    specialist = EchoSubprocessSpecialist("subprocess-fixture", {"echo"}, version="v1")

    response = specialist.run(_request(text="subprocess"))

    assert response.refusal_or_error is None
    assert response.tenant_id == "tenant"
    assert response.claims[0].claim_text == "echoed subprocess"
    assert response.claims[0].evidence_ids == ["subprocess"]


def test_library_specialist_calls_imported_callable() -> None:
    specialist = EchoLibrarySpecialist("library-fixture", {"echo"}, version="v1")

    response = specialist.run(_request(text="library"))

    assert response.refusal_or_error is None
    assert response.claims[0].specialist_id == "library-fixture"
    assert response.claims[0].claim_text == "library library"


def test_service_specialist_connects_queries_and_disconnects() -> None:
    specialist = EchoServiceSpecialist("service-fixture", {"echo"}, version="v1")

    response = specialist.run(_request(text="service"))

    assert response.refusal_or_error is None
    assert specialist.connected
    assert specialist.disconnected
    assert response.claims[0].claim_text == "service service"


def test_base_classes_return_refusal_for_unsupported_task() -> None:
    specialist = EchoLibrarySpecialist("library-fixture", {"echo"}, version="v1")

    response = specialist.run(_request(task_type="other"))

    assert response.claims == []
    assert response.refusal_or_error == "unsupported task_type: other"
