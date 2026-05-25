from __future__ import annotations

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

BLAST_SOURCE_ID = "blast-v2.17"
DEFAULT_OUTFMT = (
    "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore stitle"
)


@dataclass(frozen=True)
class BlastHit:
    qseqid: str
    sseqid: str
    pident: float
    length: int
    mismatch: int
    gapopen: int
    qstart: int
    qend: int
    sstart: int
    send: int
    evalue: float
    bitscore: float
    stitle: str

    def to_payload(self) -> dict[str, object]:
        return {
            "qseqid": self.qseqid,
            "sseqid": self.sseqid,
            "pident": self.pident,
            "length": self.length,
            "mismatch": self.mismatch,
            "gapopen": self.gapopen,
            "qstart": self.qstart,
            "qend": self.qend,
            "sstart": self.sstart,
            "send": self.send,
            "evalue": self.evalue,
            "bitscore": self.bitscore,
            "stitle": self.stitle,
        }


class BLASTSpecialist(SubprocessSpecialist):
    def __init__(
        self,
        specialist_id: str = "blast",
        task_types: set[str] | tuple[str, ...] = ("sequence_alignment",),
        *,
        version: str = "2.17",
        binary: str | Path | None = None,
        artifact_store: ContentAddressedStore | Path | str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(specialist_id, task_types, version=f"blast-{version}")
        self.binary = str(resolve_blast_binary("blastn", binary))
        self.timeout_seconds = timeout_seconds
        if isinstance(artifact_store, ContentAddressedStore):
            self.artifact_store = artifact_store
        else:
            root = artifact_store or os.environ.get("VECL_ARTIFACT_STORE")
            self.artifact_store = ContentAddressedStore(
                root or Path(os.environ.get("TMPDIR", "/tmp")) / "vecl-qb-artifacts"
            )

    def args_from(self, request: SpecialistRequest) -> tuple[str, ...]:
        raise NotImplementedError("BLASTSpecialist writes query FASTA before execution")

    def parse_output(
        self, stdout: str, stderr: str, files: dict[str, bytes]
    ) -> list[SpecialistClaim]:
        raise NotImplementedError("BLASTSpecialist parses a request-scoped TSV output")

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        if not self.can_handle(request):
            return SpecialistResponse(
                request.request_id,
                self.specialist_id,
                [],
                request.tenant_id,
                refusal_or_error=f"unsupported task_type: {request.task_type}",
            )
        try:
            config_response = maybe_specialist_config_response(
                specialist_id=self.specialist_id,
                version=self.version,
                source_id=BLAST_SOURCE_ID,
                request=request,
                artifact_store=self.artifact_store,
            )
            if config_response is not None:
                return config_response
            database = _required_text(request.input_payload, "database")
            query_fasta = _query_fasta(request.input_payload)
            max_target_seqs = _positive_int(request.input_payload.get("max_target_seqs", 5))
            task = str(request.input_payload.get("task") or _default_task(query_fasta))
            evalue = str(request.input_payload.get("evalue") or "10")
            hits, raw_output, stderr = self._run_blast(
                database=database,
                query_fasta=query_fasta,
                max_target_seqs=max_target_seqs,
                task=task,
                evalue=evalue,
            )
            record = self._write_artifact(request, database, query_fasta, raw_output)
            claim = self._claim_from_hits(request, database, query_fasta, hits, record)
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
                "hit_count": len(hits),
                "binary_path": self.binary,
                "stderr": stderr,
            },
        )

    def _run_blast(
        self,
        *,
        database: str,
        query_fasta: str,
        max_target_seqs: int,
        task: str,
        evalue: str,
    ) -> tuple[list[BlastHit], str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            query_path = Path(tmpdir) / "query.fasta"
            output_path = Path(tmpdir) / "blast.tsv"
            query_path.write_text(query_fasta)
            completed = subprocess.run(
                [
                    self.binary,
                    "-query",
                    str(query_path),
                    "-db",
                    database,
                    "-out",
                    str(output_path),
                    "-outfmt",
                    DEFAULT_OUTFMT,
                    "-max_target_seqs",
                    str(max_target_seqs),
                    "-evalue",
                    evalue,
                    "-task",
                    task,
                    "-dust",
                    "no",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    completed.stderr.strip() or f"blastn exited with status {completed.returncode}"
                )
            raw_output = output_path.read_text() if output_path.exists() else ""
        return _parse_hits(raw_output), raw_output, completed.stderr.strip()

    def _write_artifact(
        self,
        request: SpecialistRequest,
        database: str,
        query_fasta: str,
        raw_output: str,
    ) -> ArtifactRecord:
        parent_event_id = str(request.provenance_context.get("parent_event_id") or "standalone")
        content = raw_output if raw_output.strip() else "# no BLAST hits\n"
        return self.artifact_store.write_text(
            content,
            producer_specialist_id=self.specialist_id,
            producer_version=self.version,
            input_hash=stable_hash({"database": database, "query_fasta": query_fasta}),
            output_format="tsv",
            parent_event_id=parent_event_id,
        )

    def _claim_from_hits(
        self,
        request: SpecialistRequest,
        database: str,
        query_fasta: str,
        hits: list[BlastHit],
        record: ArtifactRecord,
    ) -> SpecialistClaim:
        top_hits = [hit.to_payload() for hit in hits]
        payload = {
            "database": database,
            "query_count": query_fasta.count(">"),
            "top_hits": top_hits,
        }
        claim_id = "blast-" + stable_hash(payload)[:16]
        return SpecialistClaim(
            claim_id=claim_id,
            specialist_id=self.specialist_id,
            claim_text="blast_hits=" + _json_dumps(payload),
            claim_type="sequence_alignment",
            confidence=0.9,
            evidence_ids=[stable_hash({"query_fasta": query_fasta, "database": database})],
            source_ids=[BLAST_SOURCE_ID],
            artifact_ids=[record.artifact_id],
        )


def resolve_blast_binary(name: str = "blastn", binary: str | Path | None = None) -> Path:
    raw_candidate = binary or os.environ.get(f"{name.upper()}_BINARY")
    if raw_candidate:
        candidate = Path(raw_candidate)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
        resolved = shutil.which(str(candidate))
        if resolved:
            return Path(resolved)
    resolved = shutil.which(name)
    if resolved:
        return Path(resolved)
    raise FileNotFoundError(
        f"{name} binary not found; install NCBI BLAST+ or set {name.upper()}_BINARY"
    )


def create_blast_trust_anchor(
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
        anchor_id=f"blast-v2.17-{credential_hash[:12]}",
        source_id=BLAST_SOURCE_ID,
        root_kind=TrustRootKind.VERIFIED_OPERATIONAL_RECORD,
        trust_value=0.9,
        issued_by="vecl-qb",
        issued_at=issued,
        expires_at=issued + timedelta(days=365),
        credential_hash=credential_hash,
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{key} is required")
    return str(value).strip()


def _positive_int(value: Any) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be positive")
    return parsed


def _query_fasta(payload: dict[str, Any]) -> str:
    if payload.get("query_fasta"):
        return str(payload["query_fasta"]).strip() + "\n"
    if payload.get("query_sequence"):
        return ">query1\n" + _wrap_sequence(str(payload["query_sequence"])) + "\n"
    sequences = payload.get("query_sequences")
    if isinstance(sequences, dict):
        return "".join(
            f">{name}\n{_wrap_sequence(str(sequence))}\n"
            for name, sequence in sorted(sequences.items())
        )
    if isinstance(sequences, list):
        return "".join(
            f">query{index}\n{_wrap_sequence(str(sequence))}\n"
            for index, sequence in enumerate(sequences, start=1)
        )
    raise ValueError("query_sequence, query_sequences, or query_fasta is required")


def _wrap_sequence(sequence: str, width: int = 80) -> str:
    cleaned = "".join(sequence.split()).upper()
    if not cleaned:
        raise ValueError("query sequence must be non-empty")
    return "\n".join(cleaned[index : index + width] for index in range(0, len(cleaned), width))


def _default_task(query_fasta: str) -> str:
    sequence = "".join(
        line.strip() for line in query_fasta.splitlines() if not line.startswith(">")
    )
    return "blastn-short" if len(sequence) < 50 else "blastn"


def _parse_hits(raw_output: str) -> list[BlastHit]:
    hits: list[BlastHit] = []
    for line in raw_output.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 12:
            raise ValueError(f"invalid BLAST outfmt 6 row: {line}")
        if len(fields) == 12:
            fields.append("")
        hits.append(
            BlastHit(
                qseqid=fields[0],
                sseqid=fields[1],
                pident=float(fields[2]),
                length=int(fields[3]),
                mismatch=int(fields[4]),
                gapopen=int(fields[5]),
                qstart=int(fields[6]),
                qend=int(fields[7]),
                sstart=int(fields[8]),
                send=int(fields[9]),
                evalue=float(fields[10]),
                bitscore=float(fields[11]),
                stitle=fields[12],
            )
        )
    return hits


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))
