from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vecl.qb.specialist import SpecialistRequest
from vecl.specialists.artifacts import ContentAddressedStore
from vecl.specialists.blast_specialist import (
    BLAST_SOURCE_ID,
    BLASTSpecialist,
    create_blast_trust_anchor,
    resolve_blast_binary,
)

SUBJECT_SEQUENCE = "ATGCGTACGTAGCTAGCTAGCTAG"


def _request(payload: dict[str, object]) -> SpecialistRequest:
    return SpecialistRequest(
        "req-blast",
        "tenant-bio",
        "sequence_alignment",
        payload,
        {},
        {"parent_event_id": "evt-parent"},
    )


def _tiny_blast_db(tmp_path: Path) -> Path:
    fasta_path = tmp_path / "subjects.fasta"
    db_prefix = tmp_path / "db" / "tiny_sequences"
    db_prefix.parent.mkdir()
    fasta_path.write_text(
        "\n".join(
            [
                ">subject1 known synthetic sequence",
                SUBJECT_SEQUENCE,
                ">subject2 unrelated synthetic sequence",
                "TTTTCCCCAAAAGGGGTTTTCCCC",
                "",
            ]
        )
    )
    completed = subprocess.run(
        [
            str(resolve_blast_binary("makeblastdb")),
            "-in",
            str(fasta_path),
            "-dbtype",
            "nucl",
            "-out",
            str(db_prefix),
            "-parse_seqids",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return db_prefix


def test_real_blastn_runs_and_writes_tsv_artifact(tmp_path: Path) -> None:
    db_prefix = _tiny_blast_db(tmp_path)
    specialist = BLASTSpecialist(artifact_store=ContentAddressedStore(tmp_path / "artifacts"))

    response = specialist.run(
        _request(
            {
                "database": str(db_prefix),
                "query_sequence": SUBJECT_SEQUENCE,
                "max_target_seqs": 3,
                "task": "blastn-short",
            }
        )
    )

    assert response.refusal_or_error is None
    claim = response.claims[0]
    assert claim.claim_type == "sequence_alignment"
    assert claim.source_ids == [BLAST_SOURCE_ID]
    payload = json.loads(claim.claim_text.removeprefix("blast_hits="))
    assert payload["top_hits"][0]["sseqid"] == "subject1"
    assert payload["top_hits"][0]["pident"] == 100.0
    assert payload["top_hits"][0]["bitscore"] > 0
    record = response.cost_metadata["artifact_records"][0]  # type: ignore[index]
    assert record["output_format"] == "tsv"
    assert "subject1" in Path(str(record["output_path"])).read_text()


def test_blast_reports_missing_database_without_artifact(tmp_path: Path) -> None:
    specialist = BLASTSpecialist(artifact_store=ContentAddressedStore(tmp_path))

    response = specialist.run(
        _request({"database": str(tmp_path / "missing-db"), "query_sequence": SUBJECT_SEQUENCE})
    )

    assert response.claims == []
    assert response.refusal_or_error is not None
    assert "No alias or index file found" in response.refusal_or_error


def test_blast_trust_anchor_binds_binary_and_version() -> None:
    binary = resolve_blast_binary("blastn")
    anchor = create_blast_trust_anchor(binary, "blastn: 2.17.0+")

    assert anchor.source_id == BLAST_SOURCE_ID
    assert anchor.root_kind.value == "VERIFIED_OPERATIONAL_RECORD"
    assert anchor.trust_value == 0.9
    assert str(binary) in anchor.credential_hash or anchor.credential_hash
