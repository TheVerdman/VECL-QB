# Contributing

VECL-QB is a research prototype. Keep changes narrow, deterministic, and explicit about what
is implemented versus proposed.

## Local setup

Install [uv](https://docs.astral.sh/uv/), then create the locked development environment:

```bash
uv sync --frozen
```

Run the non-mutating release check before submitting a change:

```bash
uv run make check PYTHON="uv run python"
```

`make format` is the only standard check target that writes to source files. `make coverage`
reports branch and line coverage without enforcing an arbitrary percentage.

Core tests can be run without the locally optional/integration group:

```bash
uv run pytest --ignore=tests/integration --ignore=tests/episodic/test_optional_backends.py
```

## Change expectations

- Add a deterministic regression test for each bug fix.
- Preserve tenant boundaries and append-only provenance for every state mutation.
- Check sparse updates against the CPU oracle; do not describe a delegating accelerator
  scaffold as an independent implementation.
- Treat TLA+ documents as behavioral design contracts until an executable model and a
  reproducible TLC command are checked in.
- Do not commit credentials, local paths, generated corpora, or cloud resource identifiers.

See [AGENTS.md](AGENTS.md) for the repository's invariants and [docs/adr](docs/adr) for design
decisions.
