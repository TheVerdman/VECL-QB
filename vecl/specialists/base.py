from __future__ import annotations

import importlib
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from vecl.qb.specialist import Specialist, SpecialistClaim, SpecialistRequest, SpecialistResponse

SpecialistRunOutput = SpecialistResponse | list[SpecialistClaim]


class _TaskTypedSpecialist(Specialist):
    def __init__(
        self,
        specialist_id: str,
        task_types: Sequence[str] | set[str],
        *,
        version: str = "v0",
        **_: Any,
    ) -> None:
        if not specialist_id:
            raise ValueError("specialist_id must be non-empty")
        if not task_types:
            raise ValueError("task_types must be non-empty")
        self.specialist_id = specialist_id
        self.task_types = set(task_types)
        self.version = version

    def can_handle(self, request: SpecialistRequest) -> bool:
        return request.task_type in self.task_types

    def _response_from_output(
        self, request: SpecialistRequest, output: SpecialistRunOutput
    ) -> SpecialistResponse:
        if isinstance(output, SpecialistResponse):
            return output
        return SpecialistResponse(
            request_id=request.request_id,
            specialist_id=self.specialist_id,
            claims=output,
            tenant_id=request.tenant_id,
        )


class SubprocessSpecialist(_TaskTypedSpecialist):
    binary: str = ""

    def args_from(self, request: SpecialistRequest) -> Sequence[str]:
        raise NotImplementedError

    def parse_output(
        self, stdout: str, stderr: str, files: dict[str, bytes]
    ) -> SpecialistRunOutput:
        raise NotImplementedError

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        if not self.can_handle(request):
            return SpecialistResponse(
                request_id=request.request_id,
                specialist_id=self.specialist_id,
                claims=[],
                tenant_id=request.tenant_id,
                refusal_or_error=f"unsupported task_type: {request.task_type}",
            )
        if not self.binary:
            raise ValueError("SubprocessSpecialist.binary must be non-empty")
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [self.binary, *self.args_from(request)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
            )
            files = _collect_files(Path(tmpdir))
        if completed.returncode != 0:
            return SpecialistResponse(
                request_id=request.request_id,
                specialist_id=self.specialist_id,
                claims=[],
                tenant_id=request.tenant_id,
                refusal_or_error=completed.stderr.strip()
                or f"{self.binary} exited with status {completed.returncode}",
            )
        return self._response_from_output(
            request, self.parse_output(completed.stdout, completed.stderr, files)
        )


class LibrarySpecialist(_TaskTypedSpecialist):
    import_path: str = ""

    def call(self, request: SpecialistRequest) -> SpecialistRunOutput:
        if not self.import_path:
            raise ValueError("LibrarySpecialist.import_path must be non-empty")
        return _load_callable(self.import_path)(request)

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        if not self.can_handle(request):
            return SpecialistResponse(
                request_id=request.request_id,
                specialist_id=self.specialist_id,
                claims=[],
                tenant_id=request.tenant_id,
                refusal_or_error=f"unsupported task_type: {request.task_type}",
            )
        return self._response_from_output(request, self.call(request))


class ServiceSpecialist(_TaskTypedSpecialist):
    def connect(self) -> Any:
        raise NotImplementedError

    def query(self, request: SpecialistRequest) -> SpecialistRunOutput:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        if not self.can_handle(request):
            return SpecialistResponse(
                request_id=request.request_id,
                specialist_id=self.specialist_id,
                claims=[],
                tenant_id=request.tenant_id,
                refusal_or_error=f"unsupported task_type: {request.task_type}",
            )
        self.connect()
        try:
            return self._response_from_output(request, self.query(request))
        finally:
            self.disconnect()


def _load_callable(import_path: str) -> Callable[[SpecialistRequest], SpecialistRunOutput]:
    module_name, separator, attr_name = import_path.partition(":")
    if not separator:
        module_name, separator, attr_name = import_path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"invalid import path: {import_path}")
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)
    if not callable(target):
        raise TypeError(f"import path is not callable: {import_path}")
    return cast(Callable[[SpecialistRequest], SpecialistRunOutput], target)


def _collect_files(directory: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(directory))] = path.read_bytes()
    return files
