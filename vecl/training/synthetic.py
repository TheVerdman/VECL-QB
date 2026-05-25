from __future__ import annotations

import json

from vecl.specialists.stockfish import STOCKFISH_SOURCE_ID
from vecl.training.tool_use_loop import SupervisedToolUseExample

TENANT_ID = "vecl-training-fixture"
SOURCE_ID = STOCKFISH_SOURCE_ID
SOURCE_AUTHORITY = 0.9
SOURCE_TRUST_ANCHOR_KIND = "VERIFIED_OPERATIONAL_RECORD"


def stockfish_tool_use_examples() -> list[SupervisedToolUseExample]:
    examples = [
        (
            "stockfish-tool-001",
            "Analyze FEN rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2 at depth 12.",
            {
                "specialist_id": "stockfish",
                "task_type": "chess_eval",
                "input_payload": {
                    "fen": "rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",
                    "depth": 12,
                },
            },
        ),
        (
            "stockfish-tool-002",
            "Use the chess engine on 8/8/8/8/8/8/4K3/4k3 w - - 0 1 with depth 8.",
            {
                "specialist_id": "stockfish",
                "task_type": "chess_eval",
                "input_payload": {"fen": "8/8/8/8/8/8/4K3/4k3 w - - 0 1", "depth": 8},
            },
        ),
    ]
    return [
        SupervisedToolUseExample(
            example_id=example_id,
            tenant_id=TENANT_ID,
            source_id=SOURCE_ID,
            authority=SOURCE_AUTHORITY,
            prompt=prompt,
            target_text=json.dumps(target, sort_keys=True, separators=(",", ":")),
            task_kind="tool_call_json",
            metadata={
                "generator": "stockfish_tool_use_examples",
                "trust_anchor_source_id": SOURCE_ID,
                "trust_anchor_kind": SOURCE_TRUST_ANCHOR_KIND,
            },
        )
        for example_id, prompt, target in examples
    ]


def stockfish_final_answer_examples() -> list[SupervisedToolUseExample]:
    examples = [
        (
            "stockfish-answer-001",
            (
                "Verified Stockfish context: bestmove=d7d5; eval_cp=3; depth=12; "
                "pv=d7d5 c2c3. Answer the user: what should Black play after 1.d4 e6?"
            ),
            (
                "Black should play 1...d5. Stockfish evaluates the position as essentially "
                "equal at +0.03 from White's perspective, with principal variation 1...d5 2.c3."
            ),
        ),
        (
            "stockfish-answer-002",
            (
                "Verified Stockfish context: bestmove=e2e4; eval_cp=42; depth=10; "
                "pv=e2e4 e7e5. Give a concise recommendation for White."
            ),
            (
                "White should play e4. The engine gives White a modest edge of +0.42 pawns, "
                "and the principal variation begins 1.e4 e5."
            ),
        ),
    ]
    return [
        SupervisedToolUseExample(
            example_id=example_id,
            tenant_id=TENANT_ID,
            source_id=SOURCE_ID,
            authority=SOURCE_AUTHORITY,
            prompt=prompt,
            target_text=target,
            task_kind="final_answer",
            claim_ids=[f"claim-{example_id}"],
            artifact_ids=[f"artifact-{example_id}"],
            metadata={
                "generator": "stockfish_final_answer_examples",
                "trust_anchor_source_id": SOURCE_ID,
                "trust_anchor_kind": SOURCE_TRUST_ANCHOR_KIND,
            },
        )
        for example_id, prompt, target in examples
    ]
