from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path


def environment_directory(env_var: str, *, prefix: str) -> Path:
    """Return an explicitly configured directory or a unique process-safe temporary one."""
    configured = os.environ.get(env_var, "").strip()
    if configured:
        root = Path(configured).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root
    root = Path(tempfile.mkdtemp(prefix=prefix))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root
