from __future__ import annotations

import json
import sys

from vecl.qb.specialist import SpecialistClaim, SpecialistRequest
from vecl.specialists.base import LibrarySpecialist, ServiceSpecialist, SubprocessSpecialist


class EchoSubprocessSpecialist(SubprocessSpecialist):
    binary = sys.executable

    def args_from(self, request: SpecialistRequest) -> list[str]:
        text = request.input_payload.get("text", "")
        code = (
            "import json, pathlib, sys; "
            "text=sys.argv[1]; "
            "pathlib.Path('echo.json').write_text(json.dumps({'echo': text})); "
            "print(text)"
        )
        return ["-c", code, str(text)]

    def parse_output(
        self, stdout: str, stderr: str, files: dict[str, bytes]
    ) -> list[SpecialistClaim]:
        del stderr
        payload = json.loads(files["echo.json"].decode())
        return [
            SpecialistClaim(
                claim_id="claim-subprocess",
                specialist_id=self.specialist_id,
                claim_text=f"echoed {stdout.strip()}",
                claim_type="echo",
                confidence=1.0,
                evidence_ids=[payload["echo"]],
            )
        ]


class EchoLibrarySpecialist(LibrarySpecialist):
    import_path = "tests.specialists.fixtures.sample_specialists:library_echo"


def library_echo(request: SpecialistRequest) -> list[SpecialistClaim]:
    text = str(request.input_payload.get("text", ""))
    return [
        SpecialistClaim(
            claim_id="claim-library",
            specialist_id="library-fixture",
            claim_text=f"library {text}",
            claim_type="echo",
            confidence=0.9,
            evidence_ids=[text],
        )
    ]


class EchoServiceSpecialist(ServiceSpecialist):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.connected = False
        self.disconnected = False

    def connect(self) -> object:
        self.connected = True
        return self

    def query(self, request: SpecialistRequest) -> list[SpecialistClaim]:
        text = str(request.input_payload.get("text", ""))
        return [
            SpecialistClaim(
                claim_id="claim-service",
                specialist_id=self.specialist_id,
                claim_text=f"service {text}",
                claim_type="echo",
                confidence=0.8,
                evidence_ids=[text],
            )
        ]

    def disconnect(self) -> None:
        self.disconnected = True
