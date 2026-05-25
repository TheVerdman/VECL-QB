from __future__ import annotations

from vecl.training.corpus_factory import (
    SyntheticCorpusBuild,
    SyntheticCorpusConfig,
    augment_records_with_llm,
    category_counts,
    generate_deterministic_records,
    generate_synthetic_corpus,
)
from vecl.training.corpus_metrics import (
    admitted_llm_records,
    llm_cache_stats,
    semantic_diversity_metrics,
    training_export_records,
)
from vecl.training.corpus_schema import (
    LLMRawCacheEntry,
    ToolUseCorpusRecord,
    append_llm_cache_jsonl,
    corpus_hash,
    read_corpus_jsonl,
    read_llm_cache_jsonl,
    supervised_examples_from_corpus,
    write_corpus_jsonl,
    write_llm_cache_jsonl,
)
from vecl.training.corpus_validation import (
    CorpusValidationIssue,
    CorpusValidationResult,
    corpus_markdown_report,
    validate_corpus_path,
    validate_corpus_records,
    validate_single_record,
)
from vecl.training.scoring import (
    activation_from_gradient_summaries,
    rarity_from_selection_counts,
    slot_selection_counts_from_ledger,
)
from vecl.training.synthetic import (
    stockfish_final_answer_examples,
    stockfish_tool_use_examples,
)
from vecl.training.tool_use_loop import (
    SupervisedToolUseExample,
    ToolUseTrainer,
    ToolUseTrainingConfig,
    ToolUseTrainingReport,
    dataset_hash,
    validate_training_examples,
)

__all__ = [
    "SupervisedToolUseExample",
    "CorpusValidationIssue",
    "CorpusValidationResult",
    "LLMRawCacheEntry",
    "SyntheticCorpusBuild",
    "SyntheticCorpusConfig",
    "ToolUseCorpusRecord",
    "ToolUseTrainer",
    "ToolUseTrainingConfig",
    "ToolUseTrainingReport",
    "activation_from_gradient_summaries",
    "admitted_llm_records",
    "append_llm_cache_jsonl",
    "augment_records_with_llm",
    "category_counts",
    "corpus_hash",
    "corpus_markdown_report",
    "dataset_hash",
    "generate_deterministic_records",
    "generate_synthetic_corpus",
    "llm_cache_stats",
    "rarity_from_selection_counts",
    "read_corpus_jsonl",
    "read_llm_cache_jsonl",
    "semantic_diversity_metrics",
    "slot_selection_counts_from_ledger",
    "stockfish_final_answer_examples",
    "stockfish_tool_use_examples",
    "supervised_examples_from_corpus",
    "training_export_records",
    "validate_corpus_path",
    "validate_corpus_records",
    "validate_single_record",
    "validate_training_examples",
    "write_corpus_jsonl",
    "write_llm_cache_jsonl",
]
