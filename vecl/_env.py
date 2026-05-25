from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get("VECL_LOAD_DOTENV_FOR_TESTS") != "1":
        return
    dotenv = Path(path)
    if not dotenv.exists():
        return
    for line in dotenv.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
