PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: check coverage format format-check lint test test-core typecheck

check: format-check lint typecheck test

test:
	$(PYTHON) -m pytest

test-core:
	$(PYTHON) -m pytest --ignore=tests/integration --ignore=tests/episodic/test_optional_backends.py

coverage:
	$(PYTHON) -m coverage run -m pytest
	$(PYTHON) -m coverage report

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy vecl

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .
