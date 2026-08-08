from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from vecl._compat import UTC
from vecl._paths import environment_directory
from vecl.provenance.events import stable_hash
from vecl.qb.specialist import SpecialistClaim, SpecialistRequest, SpecialistResponse
from vecl.specialists.artifacts import ArtifactRecord, ContentAddressedStore
from vecl.specialists.base import SubprocessSpecialist
from vecl.specialists.configuration import maybe_specialist_config_response
from vecl.trust.anchors import TrustAnchor, TrustRootKind

OPENROAD_SOURCE_ID = "openroad-cli"
OPENROAD_ALLOWED_OPERATIONS = frozenset({"analyze", "floorplan", "place_route"})


class OpenROADSpecialist(SubprocessSpecialist):
    def __init__(
        self,
        specialist_id: str = "openroad",
        task_types: set[str] | tuple[str, ...] = ("physical_design", "eda_flow"),
        *,
        version: str = "cli",
        binary: str | Path | None = None,
        artifact_store: ContentAddressedStore | Path | str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(specialist_id, task_types, version=f"openroad-{version}")
        self.binary = str(binary or os.environ.get("OPENROAD_BINARY") or "openroad")
        self.timeout_seconds = timeout_seconds
        if isinstance(artifact_store, ContentAddressedStore):
            self.artifact_store = artifact_store
        else:
            root = artifact_store or os.environ.get("VECL_ARTIFACT_STORE")
            self.artifact_store = ContentAddressedStore(
                root
                or environment_directory("VECL_ARTIFACT_STORE", prefix="vecl-openroad-artifacts-")
            )

    def args_from(self, request: SpecialistRequest) -> tuple[str, ...]:
        raise NotImplementedError("OpenROADSpecialist writes a request-scoped Tcl script")

    def parse_output(
        self, stdout: str, stderr: str, files: dict[str, bytes]
    ) -> list[SpecialistClaim]:
        raise NotImplementedError("OpenROADSpecialist parses request-scoped reports")

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
            source_id=OPENROAD_SOURCE_ID,
            request=request,
            artifact_store=self.artifact_store,
        )
        if config_response is not None:
            return config_response
        try:
            operation = _operation_from_payload(request.input_payload)
            if operation not in OPENROAD_ALLOWED_OPERATIONS:
                raise ValueError(f"unsupported OpenROAD operation: {operation}")
            top_module = _required_text(request.input_payload, "top_module")
            netlist_text, netlist_origin = self._netlist_from_payload(request.input_payload)
            reports, script_text, log_text, input_hash = self._run_openroad(
                payload=request.input_payload,
                operation=operation,
                top_module=top_module,
                netlist_text=netlist_text,
                netlist_origin=netlist_origin,
            )
            records = self._write_artifacts(
                request=request,
                reports=reports,
                script_text=script_text,
                log_text=log_text,
                input_hash=input_hash,
            )
            claim = self._claim_from_artifacts(
                request=request,
                operation=operation,
                top_module=top_module,
                netlist_origin=netlist_origin,
                records=records,
                reports=reports,
                input_hash=input_hash,
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
                "artifact_records": [record.to_payload() for record in records],
                "operation": operation,
                "binary_path": resolve_openroad_binary(self.binary),
                "top_module": top_module,
                "netlist_origin": netlist_origin,
            },
        )

    def _netlist_from_payload(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if payload.get("netlist_text"):
            return str(payload["netlist_text"]), {"kind": "inline"}
        if payload.get("netlist_path"):
            path = Path(str(payload["netlist_path"])).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"netlist_path not found: {path}")
            return path.read_text(encoding="utf-8"), {"kind": "path", "path": str(path)}
        upstream = payload.get("upstream")
        if isinstance(upstream, dict):
            preferred_step = str(payload.get("netlist_from") or "")
            for step_id, summary in sorted(upstream.items()):
                if preferred_step and step_id != preferred_step:
                    continue
                if not isinstance(summary, dict):
                    continue
                for record_payload in summary.get("artifact_records", []):
                    if not isinstance(record_payload, dict):
                        continue
                    if str(record_payload.get("output_format")) not in {"v", "sv", "verilog"}:
                        continue
                    content = self.artifact_store.restore(record_payload).decode()
                    return (
                        content,
                        {
                            "kind": "upstream_artifact",
                            "step_id": step_id,
                            "artifact_id": str(record_payload.get("artifact_id")),
                            "output_hash": str(record_payload.get("output_hash")),
                        },
                    )
        raise ValueError(
            "openroad requires netlist_text, netlist_path, or upstream netlist artifact"
        )

    def _run_openroad(
        self,
        *,
        payload: dict[str, Any],
        operation: str,
        top_module: str,
        netlist_text: str,
        netlist_origin: dict[str, Any],
    ) -> tuple[dict[str, str], str, str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            netlist_path = workdir / "input_netlist.v"
            netlist_path.write_text(netlist_text, encoding="utf-8")
            script_payload = _materialize_technology_payload(payload, workdir)
            script_text = _openroad_script(script_payload, operation, top_module, netlist_path)
            script_path = workdir / "flow.tcl"
            script_path.write_text(script_text, encoding="utf-8")
            completed = subprocess.run(
                [resolve_openroad_binary(self.binary), "-exit", str(script_path)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"openroad exited with status {completed.returncode}"
                )
            reports = {
                name: path.read_text(encoding="utf-8")
                for name, path in {
                    "area": workdir / "area.rpt",
                    "checks": workdir / "checks.rpt",
                    "design": workdir / "design.rpt",
                    "def": workdir / "design.def",
                }.items()
                if path.exists() and path.read_text(encoding="utf-8").strip()
            }
            log_text = completed.stdout + (
                "\nSTDERR:\n" + completed.stderr if completed.stderr else ""
            )
            input_hash = stable_hash(
                {
                    "operation": operation,
                    "top_module": top_module,
                    "netlist_origin": netlist_origin,
                    "netlist_hash": stable_hash(netlist_text),
                    "script": script_text,
                }
            )
            return reports, script_text, log_text or "# openroad completed\n", input_hash

    def _write_artifacts(
        self,
        *,
        request: SpecialistRequest,
        reports: dict[str, str],
        script_text: str,
        log_text: str,
        input_hash: str,
    ) -> list[ArtifactRecord]:
        parent_event_id = str(request.provenance_context.get("parent_event_id") or "standalone")
        records = [
            self.artifact_store.write_text(
                log_text,
                producer_specialist_id=self.specialist_id,
                producer_version=self.version,
                input_hash=input_hash,
                output_format="txt",
                parent_event_id=parent_event_id,
            ),
            self.artifact_store.write_text(
                script_text,
                producer_specialist_id=self.specialist_id,
                producer_version=self.version,
                input_hash=input_hash,
                output_format="tcl",
                parent_event_id=parent_event_id,
            ),
        ]
        for name, text in sorted(reports.items()):
            records.append(
                self.artifact_store.write_text(
                    text,
                    producer_specialist_id=self.specialist_id,
                    producer_version=self.version,
                    input_hash=input_hash,
                    output_format="def" if name == "def" else "rpt",
                    parent_event_id=parent_event_id,
                )
            )
        return records

    def _claim_from_artifacts(
        self,
        *,
        request: SpecialistRequest,
        operation: str,
        top_module: str,
        netlist_origin: dict[str, Any],
        records: list[ArtifactRecord],
        reports: dict[str, str],
        input_hash: str,
    ) -> SpecialistClaim:
        payload = {
            "operation": operation,
            "top_module": top_module,
            "netlist_origin": netlist_origin,
            "report_names": sorted(reports),
            "artifact_ids": [record.artifact_id for record in records],
            "warnings": _warning_lines("\n".join(reports.values())),
        }
        return SpecialistClaim(
            claim_id="openroad-" + stable_hash(payload)[:16],
            specialist_id=self.specialist_id,
            claim_text="openroad_physical_design="
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            claim_type="physical_design",
            confidence=0.88,
            evidence_ids=[input_hash, stable_hash(request.input_payload)],
            source_ids=[OPENROAD_SOURCE_ID],
            artifact_ids=[record.artifact_id for record in records],
            limitations=[
                "physical-design quality depends on supplied technology LEF/liberty/SDC context"
            ],
        )


def resolve_openroad_binary(binary: str | Path = "openroad") -> str:
    if str(binary) == "openroad" and os.environ.get("OPENROAD_BINARY"):
        binary = str(os.environ["OPENROAD_BINARY"])
    resolved = shutil.which(str(binary))
    if resolved:
        return resolved
    path = Path(binary)
    if path.exists() and os.access(path, os.X_OK):
        return str(path)
    raise FileNotFoundError("openroad binary not found; install OpenROAD or set OPENROAD_BINARY")


def create_openroad_trust_anchor(
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
        anchor_id=f"openroad-{credential_hash[:12]}",
        source_id=OPENROAD_SOURCE_ID,
        root_kind=TrustRootKind.VERIFIED_OPERATIONAL_RECORD,
        trust_value=0.9,
        issued_by="vecl-qb",
        issued_at=issued,
        expires_at=issued + timedelta(days=365),
        credential_hash=credential_hash,
    )


def _operation_from_payload(payload: dict[str, Any]) -> str:
    raw = str(payload.get("operation") or payload.get("command") or "analyze")
    raw = raw.strip().lower()
    if raw.startswith("openroad "):
        raw = raw.removeprefix("openroad ").strip()
    return raw.split()[0] if raw else "analyze"


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{key} is required")
    return str(value).strip()


def _openroad_script(
    payload: dict[str, Any], operation: str, top_module: str, netlist_path: Path
) -> str:
    lines: list[str] = []
    for path in _path_list(payload.get("liberty_files")):
        lines.append(f"read_liberty {_quote_tcl_path(path)}")
    for path in _path_list(payload.get("lef_files")):
        lines.append(f"read_lef {_quote_tcl_path(path)}")
    lines.extend(
        [
            f"read_verilog {_quote_tcl_path(netlist_path)}",
            f"link_design {top_module}",
        ]
    )
    if payload.get("sdc_file"):
        lines.append(f"read_sdc {_quote_tcl_path(Path(str(payload['sdc_file'])).expanduser())}")
    if operation in {"floorplan", "place_route"}:
        die_area = str(payload.get("die_area") or "0 0 100 100")
        core_area = str(payload.get("core_area") or "10 10 90 90")
        lines.extend(
            [
                f"initialize_floorplan -die_area {{{die_area}}} -core_area {{{core_area}}}",
                "write_def design.def",
            ]
        )
    if operation == "place_route":
        lines.extend(
            [
                "place_pins -hor_layer met3 -ver_layer met2",
                "global_placement",
                "detailed_placement",
                "global_route",
                "write_def design.def",
            ]
        )
    lines.extend(
        [
            "report_design_area > area.rpt",
            "report_checks > checks.rpt",
            "report_design_area > design.rpt",
            "",
        ]
    )
    return "\n".join(lines)


def _path_list(value: object) -> list[Path]:
    if value is None:
        return []
    raw_values = [value] if isinstance(value, str) else value
    if not isinstance(raw_values, list | tuple):
        raise ValueError("technology file fields must be strings or lists")
    paths: list[Path] = []
    for item in raw_values:
        path = Path(str(item)).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"technology file not found: {path}")
        paths.append(path)
    return paths


def _materialize_technology_payload(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    copied = dict(payload)
    tech_dir = workdir / "technology"
    for key, prefix in (("liberty_files", "liberty"), ("lef_files", "lef")):
        paths = _path_list(payload.get(key))
        if not paths:
            copied.pop(key, None)
            continue
        tech_dir.mkdir(exist_ok=True)
        local_paths: list[str] = []
        for index, path in enumerate(paths):
            target = tech_dir / f"{prefix}_{index}{path.suffix or '.txt'}"
            shutil.copyfile(path, target)
            local_paths.append(str(target))
        copied[key] = local_paths
    if payload.get("sdc_file"):
        paths = _path_list(payload.get("sdc_file"))
        if paths:
            tech_dir.mkdir(exist_ok=True)
            target = tech_dir / f"constraints{paths[0].suffix or '.sdc'}"
            shutil.copyfile(paths[0], target)
            copied["sdc_file"] = str(target)
    return copied


def _quote_tcl_path(path: Path) -> str:
    return "{" + str(path) + "}"


def _warning_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if "warning" in line.lower() or "error" in line.lower()
    ][:20]
