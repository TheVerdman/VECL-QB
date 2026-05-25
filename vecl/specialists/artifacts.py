from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vecl._compat import UTC

_FORMAT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    producer_specialist_id: str
    producer_version: str
    input_hash: str
    output_hash: str
    output_path: str
    output_format: str
    created_at: datetime
    parent_event_id: str

    def __post_init__(self) -> None:
        required = [
            self.artifact_id,
            self.producer_specialist_id,
            self.producer_version,
            self.input_hash,
            self.output_hash,
            self.output_path,
            self.output_format,
            self.parent_event_id,
        ]
        if any(not value for value in required):
            raise ValueError("ArtifactRecord fields must be non-empty")

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ArtifactRecord:
        return cls(
            artifact_id=str(payload["artifact_id"]),
            producer_specialist_id=str(payload["producer_specialist_id"]),
            producer_version=str(payload["producer_version"]),
            input_hash=str(payload["input_hash"]),
            output_hash=str(payload["output_hash"]),
            output_path=str(payload["output_path"]),
            output_format=str(payload["output_format"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            parent_event_id=str(payload["parent_event_id"]),
        )


@dataclass
class ContentAddressedStore:
    root: Path | str
    _root_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root_path = Path(self.root)
        root_path.mkdir(parents=True, exist_ok=True)
        self._root_path = root_path

    def write_bytes(
        self,
        content: bytes,
        *,
        producer_specialist_id: str,
        producer_version: str,
        input_hash: str,
        output_format: str,
        parent_event_id: str,
    ) -> ArtifactRecord:
        if not content:
            raise ValueError("artifact content must be non-empty")
        output_hash = hashlib.sha256(content).hexdigest()
        safe_format = _safe_format(output_format)
        output_path = self._root_path / f"{output_hash}.{safe_format}"
        if not output_path.exists():
            output_path.write_bytes(content)
        return ArtifactRecord(
            artifact_id=f"artifact-{output_hash}",
            producer_specialist_id=producer_specialist_id,
            producer_version=producer_version,
            input_hash=input_hash,
            output_hash=output_hash,
            output_path=str(output_path),
            output_format=safe_format,
            created_at=datetime.now(UTC),
            parent_event_id=parent_event_id,
        )

    def write_text(
        self,
        text: str,
        *,
        producer_specialist_id: str,
        producer_version: str,
        input_hash: str,
        output_format: str,
        parent_event_id: str,
    ) -> ArtifactRecord:
        return self.write_bytes(
            text.encode(),
            producer_specialist_id=producer_specialist_id,
            producer_version=producer_version,
            input_hash=input_hash,
            output_format=output_format,
            parent_event_id=parent_event_id,
        )

    def read_bytes(self, record: ArtifactRecord) -> bytes:
        content = Path(record.output_path).read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != record.output_hash:
            raise ValueError("artifact content hash mismatch")
        return content

    def read_text(self, record: ArtifactRecord) -> str:
        return self.read_bytes(record).decode()

    def restore(self, record_or_payload: ArtifactRecord | dict[str, Any]) -> bytes:
        record = (
            record_or_payload
            if isinstance(record_or_payload, ArtifactRecord)
            else ArtifactRecord.from_payload(record_or_payload)
        )
        return self.read_bytes(record)


def _safe_format(output_format: str) -> str:
    if not output_format:
        raise ValueError("output_format must be non-empty")
    return _FORMAT_RE.sub("_", output_format).strip("._") or "artifact"
