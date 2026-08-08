from __future__ import annotations

import re
from pathlib import Path

import pytest

GCP_SCRIPTS = tuple(
    sorted((Path(__file__).parents[2] / "scripts" / "gcp").glob("submit_vertex_*.sh"))
)


@pytest.mark.parametrize("script_path", GCP_SCRIPTS, ids=lambda path: path.stem)
def test_vertex_launcher_uses_private_work_dir_and_matching_package_uri(
    script_path: Path,
) -> None:
    source = script_path.read_text()
    local_match = re.search(r'PACKAGE_TGZ="\$WORK_DIR/([^"\n]+)"', source)
    remote_match = re.search(r'PACKAGE_URI="\$BUCKET/packages/([^"\n]+)"', source)

    assert local_match is not None
    assert remote_match is not None
    assert local_match.group(1) == remote_match.group(1)
    assert 'WORK_DIR="$(mktemp -d ' in source
    assert "trap cleanup EXIT" in source
    assert 'PROJECT_ID="${PROJECT_ID:?' in source
    assert 'BUCKET="${BUCKET:?' in source
