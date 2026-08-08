# Generated data

Only `synthetic/v0-small/corpus.jsonl` is versioned as a deterministic review fixture. The
larger v0-full and v1-hard corpora and their training exports are generated artifacts, not
source code; Git retains their metadata, hashes, and reports so the research record remains
auditable without adding roughly 80 MB to every clone. The v0-full directory also retains
`metadata.historical.json` for the earlier corpus whose generator behavior predated the current
EDA categories; `metadata.json` is the reproducible result of the commands below.

Regenerate the deterministic corpora without an external model or cloud service:

```bash
uv run python scripts/generate_training_corpus.py \
  --size v0-small --seed 1107 --output-dir data/synthetic/v0-small
uv run python scripts/validate_training_corpus.py data/synthetic/v0-small
uv run python scripts/generate_training_corpus.py \
  --size v0-full --seed 1107 --output-dir data/synthetic/v0-full
uv run python scripts/generate_training_corpus.py \
  --size v1-hard --count 25000 --seed 1107 --output-dir data/synthetic/v1-hard
uv run python scripts/validate_training_corpus.py data/synthetic/v1-hard
uv run python scripts/export_training_examples.py data/synthetic/v1-hard
uv run python scripts/report_training_corpus.py data/synthetic/v1-hard
```

Compare the generated dataset hash with the corresponding `metadata.json`. LLM augmentation
is a separate, explicitly opt-in path and is not needed to reproduce these fixtures.
