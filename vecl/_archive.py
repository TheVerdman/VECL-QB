from __future__ import annotations

import shutil
import stat
import tarfile
import zipfile
from pathlib import Path


def extract_tar_safely(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract regular files/directories while rejecting links, devices, and path escapes."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in archive.getmembers():
        target = _validated_target(root, member.name)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not member.isfile():
            raise ValueError(f"unsupported tar member type: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        _assert_within_root(root, target.parent.resolve(), member.name)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"tar member has no file content: {member.name}")
        with source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        target.chmod(member.mode & 0o777)


def extract_zip_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract regular ZIP entries while rejecting links and path escapes."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in archive.infolist():
        target = _validated_target(root, member.filename)
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"symbolic links are not allowed in ZIP archives: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _assert_within_root(root, target.parent.resolve(), member.filename)
        with archive.open(member, "r") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        if mode:
            target.chmod(mode & 0o777)


def _validated_target(root: Path, member_name: str) -> Path:
    if not member_name or "\x00" in member_name:
        raise ValueError("archive member name must be non-empty and contain no NUL")
    member_path = Path(member_name)
    if member_path.is_absolute():
        raise ValueError(f"absolute archive path is not allowed: {member_name}")
    target = (root / member_path).resolve()
    _assert_within_root(root, target, member_name)
    return target


def _assert_within_root(root: Path, target: Path, member_name: str) -> None:
    if target != root and root not in target.parents:
        raise ValueError(f"archive path escapes destination: {member_name}")
