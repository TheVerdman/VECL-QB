from __future__ import annotations

import json
from pathlib import Path

import pytest

from vecl.qb.specialist import SpecialistRequest
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.terraform_specialist import (
    TERRAFORM_SOURCE_ID,
    TerraformSpecialist,
    create_terraform_trust_anchor,
    resolve_terraform_binary,
)


def _terraform_binary() -> str:
    try:
        return resolve_terraform_binary()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _request(payload: dict[str, object]) -> SpecialistRequest:
    return SpecialistRequest(
        "req-terraform",
        "tenant-infra",
        "infrastructure_plan",
        payload,
        {},
        {"parent_event_id": "evt-parent"},
    )


def _tiny_terraform_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "terraform"
    config_dir.mkdir()
    (config_dir / "main.tf").write_text(
        "\n".join(
            [
                'terraform { required_version = ">= 1.4.0" }',
                "",
                'variable "sku_name" {',
                "  type    = string",
                '  default = "sku-a"',
                "}",
                "",
                'resource "terraform_data" "inventory" {',
                "  input = {",
                "    sku_name = var.sku_name",
                "  }",
                "}",
                "",
                'output "sku_name" {',
                "  value = terraform_data.inventory.output.sku_name",
                "}",
                "",
            ]
        )
    )
    return config_dir


def test_real_terraform_plan_writes_json_artifact(tmp_path: Path) -> None:
    binary = _terraform_binary()
    config_dir = _tiny_terraform_config(tmp_path)
    store = ContentAddressedStore(tmp_path / "artifacts")
    specialist = TerraformSpecialist(binary=binary, artifact_store=store)

    response = specialist.run(
        _request(
            {
                "operation": "plan",
                "config_dir": str(config_dir),
                "variables": {"sku_name": "sku-b"},
            }
        )
    )

    assert response.refusal_or_error is None
    claim = response.claims[0]
    assert claim.claim_type == "terraform_plan"
    assert claim.source_ids == [TERRAFORM_SOURCE_ID]
    payload = json.loads(claim.claim_text.removeprefix("terraform_result="))
    assert payload["change_summary"]["create"] == 1
    assert payload["resource_changes"][0]["address"] == "terraform_data.inventory"
    record = response.cost_metadata["artifact_records"][0]  # type: ignore[index]
    artifact = json.loads(store.restore(record).decode())
    assert artifact["resource_changes"][0]["address"] == "terraform_data.inventory"


def test_real_terraform_validate_writes_json_artifact(tmp_path: Path) -> None:
    binary = _terraform_binary()
    config_dir = _tiny_terraform_config(tmp_path)
    store = ContentAddressedStore(tmp_path / "artifacts")
    specialist = TerraformSpecialist(binary=binary, artifact_store=store)

    response = specialist.run(_request({"operation": "validate", "config_dir": str(config_dir)}))

    assert response.refusal_or_error is None
    claim = response.claims[0]
    assert claim.claim_type == "terraform_validation"
    payload = json.loads(claim.claim_text.removeprefix("terraform_result="))
    assert payload["valid"] is True
    record = response.cost_metadata["artifact_records"][0]  # type: ignore[index]
    artifact = json.loads(store.restore(record).decode())
    assert artifact["valid"] is True


@pytest.mark.parametrize("operation", ["apply", "destroy", "state list", "terraform_apply"])
def test_terraform_mutating_operations_are_refused_without_binary(
    tmp_path: Path, operation: str
) -> None:
    specialist = TerraformSpecialist(binary=tmp_path / "missing-terraform")
    expected_operation = operation.replace("terraform_", "").split()[0]

    response = specialist.run(_request({"operation": operation}))

    assert response.claims == []
    assert response.refusal_or_error == (
        f"Terraform operation is disabled in Phase 8: {expected_operation}"
    )
    assert response.cost_metadata == {
        "operation": expected_operation,
        "phase": "plan-only",
    }


def test_terraform_reports_missing_config_directory(tmp_path: Path) -> None:
    specialist = TerraformSpecialist(binary=tmp_path / "missing-terraform")

    response = specialist.run(
        _request({"operation": "plan", "config_dir": str(tmp_path / "missing-config")})
    )

    assert response.claims == []
    assert response.refusal_or_error == "config_dir must be an existing directory"


def test_terraform_trust_anchor_binds_binary_and_version(tmp_path: Path) -> None:
    binary = tmp_path / "terraform"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)

    anchor = create_terraform_trust_anchor(binary, "Terraform v1.15.3")

    assert anchor.source_id == TERRAFORM_SOURCE_ID
    assert anchor.root_kind.value == "VERIFIED_OPERATIONAL_RECORD"
    assert anchor.trust_value == 0.95
    assert anchor.credential_hash
