from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.chess_answer_eval_gemma_stockfish import (
    AnswerParseError,
    ChessAnswer,
    build_answer_prompt,
    build_tool_context,
    parse_answer_response,
    score_answer,
)
from vecl.provenance.events import stable_hash
from vecl.specialists.artifacts import ContentAddressedStore


def test_build_answer_prompt_includes_stockfish_claim_and_transcript(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    transcript = "$ stockfish\ninfo depth 8 score cp 42 pv e2e4 e7e5\nbestmove e2e4\n"
    transcript_record = store.write_text(
        transcript,
        producer_specialist_id="stockfish",
        producer_version="stockfish-18",
        input_hash=stable_hash({"fen": "startpos"}),
        output_format="txt",
        parent_event_id="evt-step",
    )
    report_record = store.write_text(
        json.dumps({"analysis": "bestmove=e2e4"}),
        producer_specialist_id="stockfish-formatter",
        producer_version="v1",
        input_hash=stable_hash({"analysis": "bestmove=e2e4"}),
        output_format="json",
        parent_event_id="evt-format",
    )
    context = build_tool_context(
        entry={
            "input_payload": {
                "query": "What should White play and why?",
                "fen": "startpos",
            }
        },
        chain_answer_text=(
            "bestmove=e2e4; eval_cp=42; pv=e2e4 e7e5; time_ms=1; "
            "depth_reached=8; budget=depth=8 stockfish_chain_report={}"
        ),
        artifact_payloads=[transcript_record.to_payload(), report_record.to_payload()],
    )

    prompt = build_answer_prompt(context)

    assert context.expected_bestmove == "e2e4"
    assert "Use the Stockfish evidence" in prompt
    assert "do not describe it as Monte Carlo" in prompt
    assert "bestmove=e2e4" in prompt
    assert "info depth 8 score cp 42" in prompt


def test_parse_answer_response_accepts_fenced_json() -> None:
    answer = parse_answer_response(
        """```json
{"bestmove":"e2e4","engine_eval":"42 cp","principal_variation":["e2e4","e7e5"],"answer":"Stockfish recommends e2e4 because the engine line is e2e4 e7e5.","used_stockfish":true}
```"""
    )

    assert answer.bestmove == "e2e4"
    assert answer.principal_variation == ("e2e4", "e7e5")
    assert answer.used_stockfish is True


def test_parse_answer_response_rejects_non_json() -> None:
    with pytest.raises(AnswerParseError, match="could not parse"):
        parse_answer_response("I would play e2e4.")


def test_score_answer_requires_engine_move_and_stockfish_use() -> None:
    context = build_tool_context(
        entry={"input_payload": {"query": "What should White play?", "fen": "startpos"}},
        chain_answer_text="bestmove=e2e4; eval_cp=42; pv=e2e4 e7e5 stockfish_chain_report={}",
        artifact_payloads=[
            _artifact_payload(
                "txt",
                "$ stockfish\ninfo depth 8 score cp 42 pv e2e4 e7e5\nbestmove e2e4\n",
            )
        ],
    )
    correct = ChessAnswer(
        bestmove="e2e4",
        engine_eval="42 cp",
        principal_variation=("e2e4", "e7e5"),
        answer="Stockfish's engine line supports e2e4 as the best move.",
        used_stockfish=True,
        raw_response="{}",
    )
    hallucinated = ChessAnswer(
        bestmove="d2d4",
        engine_eval="42 cp",
        principal_variation=("d2d4",),
        answer="I prefer d2d4 from general opening principles.",
        used_stockfish=False,
        raw_response="{}",
    )

    assert score_answer(context, correct)["passed"] is True
    assert score_answer(context, hallucinated)["passed"] is False


def _artifact_payload(output_format: str, text: str) -> dict[str, object]:
    root = Path(tempfile.mkdtemp(prefix="vecl-answer-artifacts-"))
    store = ContentAddressedStore(root)
    return store.write_text(
        text,
        producer_specialist_id="test",
        producer_version="v1",
        input_hash=stable_hash({"text": text}),
        output_format=output_format,
        parent_event_id="evt",
    ).to_payload()
