from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from vecl._archive import extract_tar_safely, extract_zip_safely
from vecl._paths import environment_directory


def test_safe_tar_extracts_regular_file(tmp_path: Path) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        info = tarfile.TarInfo("nested/example.txt")
        content = b"reviewable\n"
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    payload.seek(0)
    with tarfile.open(fileobj=payload, mode="r") as archive:
        extract_tar_safely(archive, tmp_path)

    assert (tmp_path / "nested/example.txt").read_bytes() == b"reviewable\n"


@pytest.mark.parametrize("member_name", ["../outside.txt", "/absolute.txt"])
def test_safe_tar_rejects_path_escape(tmp_path: Path, member_name: str) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        info = tarfile.TarInfo(member_name)
        content = b"unsafe"
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    payload.seek(0)
    with tarfile.open(fileobj=payload, mode="r") as archive:
        with pytest.raises(ValueError, match="archive path|absolute archive"):
            extract_tar_safely(archive, tmp_path / "extract")

    assert not (tmp_path / "outside.txt").exists()


def test_safe_tar_rejects_symbolic_link(tmp_path: Path) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        archive.addfile(info)

    payload.seek(0)
    with tarfile.open(fileobj=payload, mode="r") as archive:
        with pytest.raises(ValueError, match="unsupported tar member type"):
            extract_tar_safely(archive, tmp_path)


def test_safe_zip_rejects_path_escape(tmp_path: Path) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    payload.seek(0)
    with zipfile.ZipFile(payload) as archive:
        with pytest.raises(ValueError, match="archive path escapes"):
            extract_zip_safely(archive, tmp_path / "extract")


def test_environment_directory_uses_configured_or_unique_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured"
    monkeypatch.setenv("VECL_TEST_DIRECTORY", str(configured))
    assert environment_directory("VECL_TEST_DIRECTORY", prefix="unused-") == configured
    assert configured.is_dir()

    monkeypatch.delenv("VECL_TEST_DIRECTORY")
    first = environment_directory("VECL_TEST_DIRECTORY", prefix="vecl-test-")
    second = environment_directory("VECL_TEST_DIRECTORY", prefix="vecl-test-")
    assert first != second
    assert first.is_dir() and second.is_dir()
