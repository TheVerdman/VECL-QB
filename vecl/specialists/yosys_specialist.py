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

YOSYS_SOURCE_ID = "yosys-cli"
YOSYS_ALLOWED_OPERATIONS = frozenset({"synthesize", "synth", "compile"})


class YosysSpecialist(SubprocessSpecialist):
    def __init__(
        self,
        specialist_id: str = "yosys",
        task_types: set[str] | tuple[str, ...] = ("hardware_synthesis", "eda_flow"),
        *,
        version: str = "cli",
        binary: str | Path | None = None,
        artifact_store: ContentAddressedStore | Path | str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(specialist_id, task_types, version=f"yosys-{version}")
        self.binary = str(binary or os.environ.get("YOSYS_BINARY") or "yosys")
        self.timeout_seconds = timeout_seconds
        if isinstance(artifact_store, ContentAddressedStore):
            self.artifact_store = artifact_store
        else:
            root = artifact_store or os.environ.get("VECL_ARTIFACT_STORE")
            self.artifact_store = ContentAddressedStore(
                root or environment_directory("VECL_ARTIFACT_STORE", prefix="vecl-yosys-artifacts-")
            )

    def args_from(self, request: SpecialistRequest) -> tuple[str, ...]:
        raise NotImplementedError("YosysSpecialist writes a request-scoped script")

    def parse_output(
        self, stdout: str, stderr: str, files: dict[str, bytes]
    ) -> list[SpecialistClaim]:
        raise NotImplementedError("YosysSpecialist parses request-scoped artifacts")

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
            source_id=YOSYS_SOURCE_ID,
            request=request,
            artifact_store=self.artifact_store,
        )
        if config_response is not None:
            return config_response
        try:
            operation = _operation_from_payload(request.input_payload)
            if operation not in YOSYS_ALLOWED_OPERATIONS:
                raise ValueError(f"unsupported Yosys operation: {operation}")
            top_module = _required_text(request.input_payload, "top_module")
            netlist_text, design_json, log_text, script_text, input_hash = self._run_yosys(
                request.input_payload, top_module
            )
            records = self._write_artifacts(
                request=request,
                top_module=top_module,
                netlist_text=netlist_text,
                design_json=design_json,
                log_text=log_text,
                script_text=script_text,
                input_hash=input_hash,
            )
            claim = self._claim_from_artifacts(
                request=request,
                top_module=top_module,
                log_text=log_text,
                records=records,
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
                "binary_path": resolve_yosys_binary(self.binary),
                "top_module": top_module,
            },
        )

    def _run_yosys(
        self, payload: dict[str, Any], top_module: str
    ) -> tuple[str, dict[str, Any], str, str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            rtl_paths = _materialize_rtl_sources(payload, workdir)
            script_text = _yosys_script(rtl_paths, top_module)
            script_path = workdir / "synth.ys"
            script_path.write_text(script_text, encoding="utf-8")
            completed = subprocess.run(
                [resolve_yosys_binary(self.binary), "-s", str(script_path)],
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
                    or f"yosys exited with status {completed.returncode}"
                )
            netlist_path = workdir / "synthesized.v"
            json_path = workdir / "design.json"
            if not netlist_path.exists():
                raise RuntimeError("yosys did not produce synthesized.v")
            design_json: dict[str, Any] = {}
            if json_path.exists() and json_path.read_text(encoding="utf-8").strip():
                design_json = json.loads(json_path.read_text(encoding="utf-8"))
            log_text = completed.stdout + (
                "\nSTDERR:\n" + completed.stderr if completed.stderr else ""
            )
            input_hash = stable_hash(
                {
                    "top_module": top_module,
                    "rtl_hashes": [
                        stable_hash(path.read_text(encoding="utf-8")) for path in rtl_paths
                    ],
                    "script": script_text,
                }
            )
            return (
                netlist_path.read_text(encoding="utf-8"),
                design_json,
                log_text or "# yosys completed\n",
                script_text,
                input_hash,
            )

    def _write_artifacts(
        self,
        *,
        request: SpecialistRequest,
        top_module: str,
        netlist_text: str,
        design_json: dict[str, Any],
        log_text: str,
        script_text: str,
        input_hash: str,
    ) -> list[ArtifactRecord]:
        parent_event_id = str(request.provenance_context.get("parent_event_id") or "standalone")
        records = [
            self.artifact_store.write_text(
                netlist_text,
                producer_specialist_id=self.specialist_id,
                producer_version=self.version,
                input_hash=input_hash,
                output_format="v",
                parent_event_id=parent_event_id,
            ),
            self.artifact_store.write_text(
                json.dumps(design_json, sort_keys=True, indent=2) if design_json else "{}\n",
                producer_specialist_id=self.specialist_id,
                producer_version=self.version,
                input_hash=input_hash,
                output_format="json",
                parent_event_id=parent_event_id,
            ),
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
                output_format="ys",
                parent_event_id=parent_event_id,
            ),
        ]
        return records

    def _claim_from_artifacts(
        self,
        *,
        request: SpecialistRequest,
        top_module: str,
        log_text: str,
        records: list[ArtifactRecord],
        input_hash: str,
    ) -> SpecialistClaim:
        payload = {
            "top_module": top_module,
            "netlist_artifact_id": records[0].artifact_id,
            "json_artifact_id": records[1].artifact_id,
            "log_artifact_id": records[2].artifact_id,
            "warnings": _warning_lines(log_text),
            "stat_excerpt": _stat_excerpt(log_text),
        }
        return SpecialistClaim(
            claim_id="yosys-" + stable_hash(payload)[:16],
            specialist_id=self.specialist_id,
            claim_text="yosys_synthesis="
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            claim_type="hardware_synthesis",
            confidence=0.9,
            evidence_ids=[input_hash, stable_hash(request.input_payload)],
            source_ids=[YOSYS_SOURCE_ID],
            artifact_ids=[record.artifact_id for record in records],
            limitations=[
                "generic Yosys synthesis; physical timing requires downstream technology context"
            ],
        )


def resolve_yosys_binary(binary: str | Path = "yosys") -> str:
    if str(binary) == "yosys" and os.environ.get("YOSYS_BINARY"):
        binary = str(os.environ["YOSYS_BINARY"])
    resolved = shutil.which(str(binary))
    if resolved:
        return resolved
    path = Path(binary)
    if path.exists() and os.access(path, os.X_OK):
        return str(path)
    raise FileNotFoundError("yosys binary not found; install Yosys or set YOSYS_BINARY")


def create_yosys_trust_anchor(
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
        anchor_id=f"yosys-{credential_hash[:12]}",
        source_id=YOSYS_SOURCE_ID,
        root_kind=TrustRootKind.VERIFIED_OPERATIONAL_RECORD,
        trust_value=0.9,
        issued_by="vecl-qb",
        issued_at=issued,
        expires_at=issued + timedelta(days=365),
        credential_hash=credential_hash,
    )


def _operation_from_payload(payload: dict[str, Any]) -> str:
    raw = str(payload.get("operation") or payload.get("command") or "synthesize")
    raw = raw.strip().lower()
    if raw.startswith("yosys "):
        raw = raw.removeprefix("yosys ").strip()
    return raw.split()[0] if raw else "synthesize"


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{key} is required")
    return str(value).strip()


def _materialize_rtl_sources(payload: dict[str, Any], workdir: Path) -> list[Path]:
    paths: list[Path] = []
    if payload.get("verilog_text"):
        path = workdir / str(payload.get("filename") or "input.v")
        path.write_text(str(payload["verilog_text"]), encoding="utf-8")
        paths.append(path)
    sources = payload.get("verilog_sources")
    if isinstance(sources, dict):
        for index, (name, text) in enumerate(sorted(sources.items())):
            safe_name = _safe_rtl_filename(str(name), fallback=f"source_{index}.v")
            path = workdir / safe_name
            path.write_text(str(text), encoding="utf-8")
            paths.append(path)
    rtl_files = payload.get("rtl_files") or payload.get("source_files") or []
    if isinstance(rtl_files, str):
        rtl_files = [rtl_files]
    if isinstance(rtl_files, list | tuple):
        for item in rtl_files:
            source = Path(str(item)).expanduser().resolve()
            if not source.is_file():
                raise ValueError(f"RTL source file not found: {source}")
            target = workdir / _safe_rtl_filename(source.name, fallback=f"rtl_{len(paths)}.v")
            shutil.copyfile(source, target)
            paths.append(target)
    if not paths:
        raise ValueError("yosys requires verilog_text, verilog_sources, or rtl_files")
    return paths


def _safe_rtl_filename(name: str, *, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    if not cleaned or cleaned in {".", ".."}:
        cleaned = fallback
    if not cleaned.endswith((".v", ".sv")):
        cleaned += ".v"
    return cleaned


def _yosys_script(rtl_paths: list[Path], top_module: str) -> str:
    read_lines = "\n".join(f"read_verilog {_quote_yosys_path(path)}" for path in rtl_paths)
    return "\n".join(
        [
            read_lines,
            f"hierarchy -check -top {top_module}",
            "proc",
            "opt",
            "fsm",
            "opt",
            "memory",
            "opt",
            "techmap",
            "opt",
            "stat",
            "write_json design.json",
            "write_verilog synthesized.v",
            "",
        ]
    )


def _quote_yosys_path(path: Path) -> str:
    # Yosys command scripts do not use Tcl brace quoting; brace-wrapped paths are
    # interpreted literally by real Yosys even though the fake fixture accepts them.
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _warning_lines(log_text: str) -> list[str]:
    return [
        line.strip()
        for line in log_text.splitlines()
        if "warning" in line.lower() or "error" in line.lower()
    ][:20]


def _stat_excerpt(log_text: str, max_lines: int = 40) -> list[str]:
    lines = [line.rstrip() for line in log_text.splitlines()]
    selected: list[str] = []
    capture = False
    for line in lines:
        if "Printing statistics" in line or line.strip().startswith("=== "):
            capture = True
        if capture:
            selected.append(line)
        if len(selected) >= max_lines:
            break
    return selected
