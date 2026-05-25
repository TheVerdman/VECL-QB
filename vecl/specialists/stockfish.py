from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
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

STOCKFISH_SOURCE_ID = "stockfish-v18"
DEFAULT_STOCKFISH_DEPTH = 4
DEFAULT_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class StockfishAnalysis:
    engine_name: str
    bestmove: str
    depth_reached: int
    score_cp: int | None
    score_mate: int | None
    principal_variation: list[str]
    time_ms: int | None
    nodes: int | None


class StockfishSpecialist(SubprocessSpecialist):
    def __init__(
        self,
        specialist_id: str = "stockfish",
        task_types: set[str] | tuple[str, ...] = ("chess_eval",),
        *,
        version: str = "18",
        binary: str | Path | None = None,
        artifact_store: ContentAddressedStore | Path | str | None = None,
        required_major_version: int | None = 18,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        working_directory: str | Path | None = None,
    ) -> None:
        super().__init__(specialist_id, task_types, version=f"stockfish-{version}")
        self.binary = str(resolve_stockfish_binary(binary))
        self.working_directory = (
            Path(working_directory) if working_directory else Path(self.binary).parent
        )
        self.required_major_version = required_major_version
        self.timeout_seconds = timeout_seconds
        if isinstance(artifact_store, ContentAddressedStore):
            self.artifact_store = artifact_store
        else:
            root = artifact_store or os.environ.get("VECL_ARTIFACT_STORE")
            self.artifact_store = ContentAddressedStore(
                root or Path(os.environ.get("TMPDIR", "/tmp")) / "vecl-qb-artifacts"
            )

    def args_from(self, request: SpecialistRequest) -> tuple[str, ...]:
        raise NotImplementedError("Stockfish uses interactive UCI protocol")

    def parse_output(
        self, stdout: str, stderr: str, files: dict[str, bytes]
    ) -> list[SpecialistClaim]:
        raise NotImplementedError("Stockfish parses the interactive UCI transcript")

    def run(self, request: SpecialistRequest) -> SpecialistResponse:
        if not self.can_handle(request):
            return SpecialistResponse(
                request_id=request.request_id,
                specialist_id=self.specialist_id,
                claims=[],
                tenant_id=request.tenant_id,
                refusal_or_error=f"unsupported task_type: {request.task_type}",
            )
        try:
            config_response = maybe_specialist_config_response(
                specialist_id=self.specialist_id,
                version=self.version,
                source_id=STOCKFISH_SOURCE_ID,
                request=request,
                artifact_store=self.artifact_store,
            )
            if config_response is not None:
                return config_response
            fen = _required_text(request.input_payload, "fen")
            depth = _positive_int(
                request.input_payload.get("depth", DEFAULT_STOCKFISH_DEPTH), "depth"
            )
            movetime_ms = _optional_positive_int(
                request.input_payload.get("movetime_ms"), "movetime_ms"
            )
            analysis, transcript = self._analyze(fen, depth, movetime_ms)
            record = self._write_artifact(request, fen, depth, movetime_ms, analysis, transcript)
            claim = self._claim_from_analysis(request, fen, depth, movetime_ms, analysis, record)
        except (FileNotFoundError, TimeoutError, RuntimeError, ValueError) as exc:
            return SpecialistResponse(
                request_id=request.request_id,
                specialist_id=self.specialist_id,
                claims=[],
                tenant_id=request.tenant_id,
                refusal_or_error=str(exc),
            )
        return SpecialistResponse(
            request_id=request.request_id,
            specialist_id=self.specialist_id,
            claims=[claim],
            tenant_id=request.tenant_id,
            cost_metadata={
                "artifact_records": [record.to_payload()],
                "engine_name": analysis.engine_name,
                "binary_path": self.binary,
                "depth_reached": analysis.depth_reached,
                "time_ms": analysis.time_ms,
                "nodes": analysis.nodes,
            },
        )

    def _analyze(
        self, fen: str, depth: int, movetime_ms: int | None
    ) -> tuple[StockfishAnalysis, str]:
        process = subprocess.Popen(
            [self.binary],
            cwd=self.working_directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_queue = _stdout_queue(process.stdout)
        output_lines: list[str] = []
        transcript_lines = [f"$ {self.binary}"]
        try:
            _send(process, "uci", transcript_lines)
            _read_until(stdout_queue, "uciok", output_lines, transcript_lines, self.timeout_seconds)
            engine_name = _engine_name(output_lines)
            if self.required_major_version is not None:
                _require_major_version(engine_name, self.required_major_version)
            _send(process, "setoption name Threads value 1", transcript_lines)
            _send(process, "setoption name Hash value 16", transcript_lines)
            _send(process, "setoption name nodestime value 1", transcript_lines)
            _send(process, "isready", transcript_lines)
            _read_until(
                stdout_queue, "readyok", output_lines, transcript_lines, self.timeout_seconds
            )
            _send(process, "ucinewgame", transcript_lines)
            _send(process, "isready", transcript_lines)
            _read_until(
                stdout_queue, "readyok", output_lines, transcript_lines, self.timeout_seconds
            )
            _send(process, f"position fen {fen}", transcript_lines)
            go_command = (
                f"go movetime {movetime_ms}" if movetime_ms is not None else f"go depth {depth}"
            )
            _send(process, go_command, transcript_lines)
            _read_until(
                stdout_queue, "bestmove ", output_lines, transcript_lines, self.timeout_seconds
            )
            _send(process, "quit", transcript_lines)
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                _terminate_process(process)
        stderr = process.stderr.read() if process.stderr is not None else ""
        if process.returncode not in (0, None):
            raise RuntimeError(
                stderr.strip() or f"stockfish exited with status {process.returncode}"
            )
        if stderr.strip():
            transcript_lines.append("<stderr>")
            transcript_lines.extend(stderr.splitlines())
        return _parse_analysis(output_lines), "\n".join(transcript_lines) + "\n"

    def _write_artifact(
        self,
        request: SpecialistRequest,
        fen: str,
        depth: int,
        movetime_ms: int | None,
        analysis: StockfishAnalysis,
        transcript: str,
    ) -> ArtifactRecord:
        parent_event_id = str(request.provenance_context.get("parent_event_id") or "standalone")
        input_hash = stable_hash(
            {
                "fen": fen,
                "depth": depth,
                "movetime_ms": movetime_ms,
                "engine_name": analysis.engine_name,
                "binary_path": self.binary,
            }
        )
        return self.artifact_store.write_text(
            transcript,
            producer_specialist_id=self.specialist_id,
            producer_version=self.version,
            input_hash=input_hash,
            output_format="txt",
            parent_event_id=parent_event_id,
        )

    def _claim_from_analysis(
        self,
        request: SpecialistRequest,
        fen: str,
        depth: int,
        movetime_ms: int | None,
        analysis: StockfishAnalysis,
        record: ArtifactRecord,
    ) -> SpecialistClaim:
        if analysis.score_cp is not None:
            score_text = f"eval_cp={analysis.score_cp}"
        elif analysis.score_mate is not None:
            score_text = f"mate_in={analysis.score_mate}"
        else:
            score_text = "eval_cp=unreported"
        pv_text = " ".join(analysis.principal_variation)
        time_text = (
            f"time_ms={analysis.time_ms}" if analysis.time_ms is not None else "time_ms=unreported"
        )
        budget_text = f"movetime_ms={movetime_ms}" if movetime_ms is not None else f"depth={depth}"
        claim_text = (
            f"bestmove={analysis.bestmove}; {score_text}; pv={pv_text}; "
            f"{time_text}; depth_reached={analysis.depth_reached}; budget={budget_text}"
        )
        claim_id = (
            "stockfish-"
            + stable_hash(
                {
                    "fen": fen,
                    "depth": depth,
                    "movetime_ms": movetime_ms,
                    "engine_name": analysis.engine_name,
                    "bestmove": analysis.bestmove,
                    "score_cp": analysis.score_cp,
                    "score_mate": analysis.score_mate,
                    "principal_variation": analysis.principal_variation,
                    "time_ms": analysis.time_ms,
                    "depth_reached": analysis.depth_reached,
                }
            )[:16]
        )
        return SpecialistClaim(
            claim_id=claim_id,
            specialist_id=self.specialist_id,
            claim_text=claim_text,
            claim_type="chess_eval",
            confidence=0.9,
            evidence_ids=[stable_hash({"fen": fen})],
            source_ids=[STOCKFISH_SOURCE_ID],
            artifact_ids=[record.artifact_id],
        )


def resolve_stockfish_binary(binary: str | Path | None = None) -> Path:
    raw_candidate = binary or os.environ.get("STOCKFISH_BINARY")
    if raw_candidate:
        candidate = Path(raw_candidate)
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
        resolved = shutil.which(str(candidate))
        if resolved:
            return Path(resolved)
    resolved = shutil.which("stockfish")
    if resolved:
        return Path(resolved)
    raise FileNotFoundError(
        "Stockfish binary not found; install stable Stockfish 18 or set STOCKFISH_BINARY"
    )


def create_stockfish_trust_anchor(
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
        anchor_id=f"stockfish-v18-{credential_hash[:12]}",
        source_id=STOCKFISH_SOURCE_ID,
        root_kind=TrustRootKind.VERIFIED_OPERATIONAL_RECORD,
        trust_value=0.9,
        issued_by="vecl-qb",
        issued_at=issued,
        expires_at=issued + timedelta(days=365),
        credential_hash=credential_hash,
    )


def _send(process: subprocess.Popen[str], command: str, transcript: list[str]) -> None:
    if process.stdin is None:
        raise RuntimeError("stockfish stdin is unavailable")
    transcript.append(f"> {command}")
    process.stdin.write(command + "\n")
    process.stdin.flush()


def _stdout_queue(stream: Any) -> queue.Queue[str]:
    if stream is None:
        raise RuntimeError("stockfish stdout is unavailable")
    output_queue: queue.Queue[str] = queue.Queue()

    def read_stdout() -> None:
        for line in stream:
            output_queue.put(line.rstrip("\n"))

    thread = threading.Thread(target=read_stdout, daemon=True)
    thread.start()
    return output_queue


def _read_until(
    output_queue: queue.Queue[str],
    marker: str,
    output_lines: list[str],
    transcript: list[str],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for Stockfish")
        try:
            line = output_queue.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for Stockfish") from exc
        output_lines.append(line)
        transcript.append(f"< {line}")
        if line == marker.rstrip() or line.startswith(marker):
            return


def _terminate_process(process: subprocess.Popen[str]) -> None:
    try:
        if process.stdin is not None:
            process.stdin.write("quit\n")
            process.stdin.flush()
        process.wait(timeout=2)
    except (BrokenPipeError, subprocess.TimeoutExpired):
        process.kill()
        process.wait(timeout=2)


def _engine_name(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("id name "):
            return line.removeprefix("id name ").strip()
    raise RuntimeError("Stockfish did not report an engine name")


def _require_major_version(engine_name: str, required_major: int) -> None:
    tokens = engine_name.split()
    for token in tokens:
        if token.isdigit():
            if int(token) == required_major:
                return
            raise RuntimeError(
                f"Stockfish stable {required_major} is required; found {engine_name}"
            )
    raise RuntimeError(f"could not parse Stockfish version from engine name: {engine_name}")


def _parse_analysis(lines: list[str]) -> StockfishAnalysis:
    engine_name = _engine_name(lines)
    bestmove = ""
    depth = 0
    score_cp: int | None = None
    score_mate: int | None = None
    pv: list[str] = []
    time_ms: int | None = None
    nodes: int | None = None
    for line in lines:
        if line.startswith("bestmove "):
            bestmove = line.split()[1]
        if not line.startswith("info "):
            continue
        tokens = line.split()
        if "depth" in tokens:
            parsed_depth = _token_int(tokens, "depth", depth)
            depth = parsed_depth if parsed_depth is not None else depth
        if "time" in tokens:
            time_ms = _token_int(tokens, "time", time_ms)
        if "nodes" in tokens:
            nodes = _token_int(tokens, "nodes", nodes)
        if "score" in tokens:
            score_index = tokens.index("score")
            if len(tokens) > score_index + 2:
                score_kind = tokens[score_index + 1]
                score_value = int(tokens[score_index + 2])
                if score_kind == "cp":
                    score_cp = score_value
                    score_mate = None
                elif score_kind == "mate":
                    score_mate = score_value
                    score_cp = None
        if "pv" in tokens:
            pv = tokens[tokens.index("pv") + 1 :]
    if not bestmove:
        raise RuntimeError("Stockfish did not return a bestmove")
    return StockfishAnalysis(
        engine_name=engine_name,
        bestmove=bestmove,
        depth_reached=depth,
        score_cp=score_cp,
        score_mate=score_mate,
        principal_variation=pv,
        time_ms=time_ms,
        nodes=nodes,
    )


def _token_int(tokens: list[str], marker: str, default: int | None) -> int | None:
    index = tokens.index(marker)
    if len(tokens) <= index + 1:
        return default
    return int(tokens[index + 1])


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)
