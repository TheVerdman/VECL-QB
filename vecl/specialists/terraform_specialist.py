from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from vecl._compat import UTC
from vecl.provenance.events import stable_hash
from vecl.qb.specialist import SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.specialists.artifacts import ArtifactRecord, ContentAddressedStore
from vecl.specialists.base import SubprocessSpecialist
from vecl.specialists.configuration import maybe_specialist_config_response
from vecl.trust.anchors import TrustAnchor, TrustRootKind

TERRAFORM_SOURCE_ID = "terraform-cli"
TERRAFORM_ALLOWED_OPERATIONS = frozenset({"plan", "validate"})
TERRAFORM_DISABLED_OPERATIONS = frozenset(
    {
        "apply",
        "destroy",
        "force-unlock",
        "import",
        "state",
        "taint",
        "untaint",
        "workspace",
    }
)


@dataclass(frozen=True)
class TerraformChangeSummary:
    create: int = 0
    update: int = 0
    delete: int = 0
    replace: int = 0
    read: int = 0
    no_op: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "create": self.create,
            "update": self.update,
            "delete": self.delete,
            "replace": self.replace,
            "read": self.read,
            "no_op": self.no_op,
        }


class TerraformSpecialist(SubprocessSpecialist):
    def __init__(
        self,
        specialist_id: str = "terraform",
        task_types: set[str] | tuple[str, ...] = ("infrastructure_plan",),
        *,
        version: str = "cli",
        binary: str | Path | None = None,
        artifact_store: ContentAddressedStore | Path | str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(specialist_id, task_types, version=f"terraform-{version}")
        self.binary = str(binary or os.environ.get("TERRAFORM_BINARY") or "terraform")
        self.timeout_seconds = timeout_seconds
        if isinstance(artifact_store, ContentAddressedStore):
            self.artifact_store = artifact_store
        else:
            root = artifact_store or os.environ.get("VECL_ARTIFACT_STORE")
            self.artifact_store = ContentAddressedStore(
                root or Path(os.environ.get("TMPDIR", "/tmp")) / "vecl-qb-artifacts"
            )

    def args_from(self, request: SpecialistRequest) -> tuple[str, ...]:
        raise NotImplementedError("TerraformSpecialist runs multiple Terraform commands")

    def parse_output(
        self, stdout: str, stderr: str, files: dict[str, bytes]
    ) -> list[SpecialistClaim]:
        raise NotImplementedError("TerraformSpecialist parses request-scoped JSON output")

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        if not self.can_handle(request):
            return SpecialistResponse(
                request.request_id,
                self.specialist_id,
                [],
                request.tenant_id,
                refusal_or_error=f"unsupported task_type: {request.task_type}",
            )
        config_response = maybe_specialist_config_response(
            specialist_id=self.specialist_id,
            version=self.version,
            source_id=TERRAFORM_SOURCE_ID,
            request=request,
            artifact_store=self.artifact_store,
        )
        if config_response is not None:
            return config_response
        operation = _operation_from_payload(request.input_payload)
        if (
            operation in TERRAFORM_DISABLED_OPERATIONS
            or operation not in TERRAFORM_ALLOWED_OPERATIONS
        ):
            return SpecialistResponse(
                request.request_id,
                self.specialist_id,
                [],
                request.tenant_id,
                refusal_or_error=f"Terraform operation is disabled in Phase 8: {operation}",
                cost_metadata={"operation": operation, "phase": "plan-only"},
            )
        try:
            config_dir = _required_directory(request.input_payload, "config_dir")
            variables = _variables_from_payload(request.input_payload)
            if operation == "validate":
                payload, stderr = self._run_validate(config_dir)
                claim_type = "terraform_validation"
            else:
                payload, stderr = self._run_plan(config_dir, variables)
                claim_type = "terraform_plan"
            record = self._write_artifact(request, operation, config_dir, variables, payload)
            claim = self._claim_from_payload(
                request=request,
                operation=operation,
                payload=payload,
                record=record,
                claim_type=claim_type,
            )
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            return SpecialistResponse(
                request.request_id,
                self.specialist_id,
                [],
                request.tenant_id,
                refusal_or_error=str(exc),
            )
        return SpecialistResponse(
            request.request_id,
            self.specialist_id,
            [claim],
            request.tenant_id,
            cost_metadata={
                "artifact_records": [record.to_payload()],
                "operation": operation,
                "binary_path": resolve_terraform_binary(self.binary),
                "stderr": stderr,
            },
        )

    def _run_validate(self, config_dir: Path) -> tuple[dict[str, Any], str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = _copy_config_dir(config_dir, Path(tmpdir))
            self._terraform(working_dir, "init", "-backend=false", "-input=false", "-no-color")
            completed = self._terraform(working_dir, "validate", "-json", check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or _terraform_validate_error(completed.stdout)
                or f"terraform validate exited with status {completed.returncode}"
            )
        return json.loads(completed.stdout), completed.stderr.strip()

    def _run_plan(
        self, config_dir: Path, variables: dict[str, object]
    ) -> tuple[dict[str, Any], str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = _copy_config_dir(config_dir, Path(tmpdir))
            self._terraform(working_dir, "init", "-backend=false", "-input=false", "-no-color")
            plan_path = working_dir / "tfplan.bin"
            plan_args = [
                "plan",
                "-input=false",
                "-no-color",
                "-lock=false",
                "-out",
                str(plan_path),
                *_var_args(variables),
            ]
            self._terraform(working_dir, *plan_args)
            show = self._terraform(working_dir, "show", "-json", str(plan_path))
        return json.loads(show.stdout), show.stderr.strip()

    def _terraform(
        self,
        cwd: Path,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [resolve_terraform_binary(self.binary), *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if check and completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"terraform {' '.join(args)} exited with status {completed.returncode}"
            )
        return completed

    def _write_artifact(
        self,
        request: SpecialistRequest,
        operation: str,
        config_dir: Path,
        variables: dict[str, object],
        payload: dict[str, Any],
    ) -> ArtifactRecord:
        parent_event_id = str(request.provenance_context.get("parent_event_id") or "standalone")
        return self.artifact_store.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            producer_specialist_id=self.specialist_id,
            producer_version=self.version,
            input_hash=stable_hash(
                {
                    "operation": operation,
                    "config_dir": str(config_dir),
                    "variables": variables,
                }
            ),
            output_format="json",
            parent_event_id=parent_event_id,
        )

    def _claim_from_payload(
        self,
        *,
        request: SpecialistRequest,
        operation: str,
        payload: dict[str, Any],
        record: ArtifactRecord,
        claim_type: str,
    ) -> SpecialistClaim:
        summary = _change_summary(payload).to_payload() if operation == "plan" else {}
        claim_payload = {
            "operation": operation,
            "terraform_version": payload.get("terraform_version"),
            "format_version": payload.get("format_version"),
            "valid": payload.get("valid"),
            "change_summary": summary,
            "resource_changes": _resource_change_summaries(payload),
        }
        return SpecialistClaim(
            claim_id="terraform-" + stable_hash(claim_payload)[:16],
            specialist_id=self.specialist_id,
            claim_text="terraform_result="
            + json.dumps(claim_payload, sort_keys=True, separators=(",", ":")),
            claim_type=claim_type,
            confidence=0.92,
            evidence_ids=[stable_hash(request.input_payload)],
            source_ids=[TERRAFORM_SOURCE_ID],
            artifact_ids=[record.artifact_id],
            limitations=[
                "plan-only Phase 8 specialist; apply and destroy are disabled",
                "plan output is advisory until reviewed against the target environment",
            ],
        )


def resolve_terraform_binary(binary: str | Path = "terraform") -> str:
    if str(binary) == "terraform" and os.environ.get("TERRAFORM_BINARY"):
        binary = str(os.environ["TERRAFORM_BINARY"])
    resolved = shutil.which(str(binary))
    if resolved:
        return resolved
    path = Path(binary)
    if path.exists() and os.access(path, os.X_OK):
        return str(path)
    raise FileNotFoundError("terraform binary not found; install Terraform or set TERRAFORM_BINARY")


def create_terraform_trust_anchor(
    binary_path: str | Path,
    version_name: str,
    *,
    issued_at: datetime | None = None,
) -> TrustAnchor:
    issued = issued_at or datetime.now(UTC)
    credential_hash = stable_hash(
        {"binary_path": str(Path(binary_path)), "version_name": version_name}
    )
    return TrustAnchor(
        anchor_id=f"terraform-{credential_hash[:12]}",
        source_id=TERRAFORM_SOURCE_ID,
        root_kind=TrustRootKind.VERIFIED_OPERATIONAL_RECORD,
        trust_value=0.95,
        issued_by="vecl-qb",
        issued_at=issued,
        expires_at=issued + timedelta(days=365),
        credential_hash=credential_hash,
    )


def _operation_from_payload(payload: dict[str, Any]) -> str:
    raw = str(payload.get("operation") or payload.get("command") or "plan").strip().lower()
    if raw.startswith("terraform "):
        raw = raw.removeprefix("terraform ").strip()
    if raw.startswith("terraform_"):
        raw = raw.removeprefix("terraform_").strip()
    return raw.split()[0] if raw else "plan"


def _required_directory(payload: dict[str, Any], key: str) -> Path:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{key} is required")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{key} must be an existing directory")
    return path


def _variables_from_payload(payload: dict[str, Any]) -> dict[str, object]:
    raw = payload.get("variables") or {}
    if not isinstance(raw, dict):
        raise ValueError("variables must be an object")
    return {str(key): value for key, value in sorted(raw.items())}


def _copy_config_dir(source: Path, temp_root: Path) -> Path:
    target = temp_root / "terraform-config"
    ignore = shutil.ignore_patterns(".terraform", ".terraform.lock.hcl", "*.tfstate", "*.tfstate.*")
    shutil.copytree(source, target, ignore=ignore)
    return target


def _var_args(variables: dict[str, object]) -> list[str]:
    args: list[str] = []
    for key, value in variables.items():
        args.extend(["-var", f"{key}={_hcl_var_value(value)}"])
    return args


def _hcl_var_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list | dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _change_summary(payload: dict[str, Any]) -> TerraformChangeSummary:
    counts = {
        "create": 0,
        "update": 0,
        "delete": 0,
        "replace": 0,
        "read": 0,
        "no_op": 0,
    }
    for item in payload.get("resource_changes", []):
        actions = list(item.get("change", {}).get("actions", []))
        if actions == ["no-op"]:
            counts["no_op"] += 1
        elif actions == ["read"]:
            counts["read"] += 1
        elif actions == ["create"]:
            counts["create"] += 1
        elif actions == ["update"]:
            counts["update"] += 1
        elif actions == ["delete"]:
            counts["delete"] += 1
        elif "delete" in actions and "create" in actions:
            counts["replace"] += 1
    return TerraformChangeSummary(**counts)


def _resource_change_summaries(payload: dict[str, Any]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for item in payload.get("resource_changes", []):
        summaries.append(
            {
                "address": item.get("address"),
                "type": item.get("type"),
                "name": item.get("name"),
                "actions": list(item.get("change", {}).get("actions", [])),
            }
        )
    return summaries


def _terraform_validate_error(stdout: str) -> str:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip()
    diagnostics = payload.get("diagnostics", [])
    if diagnostics:
        return str(diagnostics[0].get("summary") or diagnostics[0].get("detail") or "")
    return ""
