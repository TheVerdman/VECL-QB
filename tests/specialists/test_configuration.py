from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vecl.qb.specialist import Specialist, SpecialistRequest
from vecl.specialists.artifacts import ArtifactRecord, ContentAddressedStore
from vecl.specialists.blast_specialist import BLASTSpecialist
from vecl.specialists.openroad_specialist import OpenROADSpecialist
from vecl.specialists.stockfish import StockfishSpecialist
from vecl.specialists.sympy_specialist import SymPySpecialist
from vecl.specialists.terraform_specialist import TerraformSpecialist
from vecl.specialists.timesfm_specialist import DeterministicTimesFMRunner, TimesFMSpecialist
from vecl.specialists.yosys_specialist import YosysSpecialist


@pytest.mark.parametrize(
    ("specialist_id", "task_type"),
    [
        ("stockfish", "chess_eval"),
        ("sympy", "symbolic_math"),
        ("blast", "sequence_alignment"),
        ("terraform", "infrastructure_plan"),
        ("timesfm", "demand_forecast"),
        ("yosys", "hardware_synthesis"),
        ("openroad", "physical_design"),
    ],
)
def test_all_registered_specialists_emit_configuration_artifacts(
    tmp_path: Path, specialist_id: str, task_type: str
) -> None:
    store = ContentAddressedStore(tmp_path / "artifacts")
    specialist = _specialist(specialist_id, store, tmp_path)
    request = SpecialistRequest(
        "req",
        "tenant",
        task_type,
        {
            "operation": "configure",
            "configuration_template": True,
            "query": f"Give me a {specialist_id} configuration.",
        },
        {},
        {"parent_event_id": "evt-parent"},
    )

    response = specialist.run(request)

    assert response.refusal_or_error is None
    assert response.claims[0].claim_type == "specialist_config"
    record = ArtifactRecord.from_payload(response.cost_metadata["artifact_records"][0])  # type: ignore[index]
    payload = json.loads(store.restore(record).decode())
    assert payload["specialist_id"] == specialist_id
    assert payload["operation"] == "configure"
    assert payload["contract"]["configure"]["template"]


def _specialist(specialist_id: str, store: ContentAddressedStore, tmp_path: Path) -> Specialist:
    if specialist_id == "stockfish":
        return StockfishSpecialist(
            binary=_fake_executable(tmp_path, "stockfish"), artifact_store=store
        )
    if specialist_id == "sympy":
        return SymPySpecialist(artifact_store=store)
    if specialist_id == "blast":
        return BLASTSpecialist(binary=_fake_executable(tmp_path, "blastn"), artifact_store=store)
    if specialist_id == "terraform":
        return TerraformSpecialist(binary=str(tmp_path / "terraform"), artifact_store=store)
    if specialist_id == "timesfm":
        return TimesFMSpecialist(runner=DeterministicTimesFMRunner(), artifact_store=store)
    if specialist_id == "yosys":
        return YosysSpecialist(binary=_fake_executable(tmp_path, "yosys"), artifact_store=store)
    if specialist_id == "openroad":
        return OpenROADSpecialist(
            binary=_fake_executable(tmp_path, "openroad"), artifact_store=store
        )
    raise AssertionError(f"unknown specialist_id: {specialist_id}")


def _fake_executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(path, 0o755)
    return path
