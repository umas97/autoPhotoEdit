# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixtures for the catalogue tests: a temporary XDG home and a fake card.

Phase 2 is about identity, idempotence and the queue, and none of those need a
real sensor: the importer reads one mebibyte of each file and never asks what is
in it. So the "RAW files" here are bytes with the right extension, which keeps
the import tests at milliseconds instead of minutes and lets them cover the
cases a real card cannot easily be made to show -- two files with identical
content, a file that disappears between two imports, a JPEG pretending to be a
photo of its own.

The tests that do need a real ARW are the ones marked ``fixtures``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def xdg_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point every XDG directory at a temporary one, for one test.

    The settings object is cached process-wide, so the cache is cleared on both
    sides of the test: a leaked temporary path would send the next test's
    catalogue into a directory that no longer exists.
    """
    from ape.config import get_settings
    from ape.db import session as db_session

    root = tmp_path / "xdg"
    for variable in ("XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(variable, str(root / variable.split("_")[1].lower()))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(root / "run"))
    (root / "run").mkdir(parents=True, exist_ok=True)

    get_settings.cache_clear()
    db_session.reset_engine()
    settings = get_settings()
    settings.ensure_dirs()
    try:
        yield root
    finally:
        db_session.reset_engine()
        get_settings.cache_clear()


@pytest.fixture
def catalog(xdg_home: Path):
    """An initialised, empty catalogue. Yields a session factory."""
    from ape.db.session import get_sessionmaker, init_db

    init_db()
    return get_sessionmaker()


def write_raw(folder: Path, name: str, *, content: bytes | None = None, size: int = 4096) -> Path:
    """Create a file that the importer will treat as a RAW.

    ``content`` makes two files identical on purpose, which is how the
    duplicate-detection case of section 15 is set up. Otherwise the bytes are
    derived from the name, so every file is distinct and reproducible.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if content is None:
        seed = name.encode()
        content = (seed * (size // len(seed) + 1))[:size]
    path.write_bytes(content)
    return path


def touch_jpeg(folder: Path, name: str) -> Path:
    path = folder / name
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"jpeg" * 64)
    return path


@pytest.fixture
def card(tmp_path: Path) -> Path:
    """A folder shaped like a memory card: six ARW, one sidecar, one subfolder."""
    folder = tmp_path / "card"
    for index in range(1, 7):
        write_raw(folder, f"DSC0{index:04d}.ARW")
    touch_jpeg(folder, "DSC00001.JPG")
    (folder / "sottocartella").mkdir()
    write_raw(folder / "sottocartella", "DSC09999.ARW")
    write_raw(folder, "IMG_0001.CR2")
    (folder / "appunti.txt").write_text("non è una foto", encoding="utf-8")
    return folder


def file_state(folder: Path) -> dict[str, tuple[int, float]]:
    """Size and mtime of every file under ``folder``, for the section 2 checks."""
    return {
        str(path.relative_to(folder)): (path.stat().st_size, path.stat().st_mtime)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def env_without_xdg() -> dict[str, str]:
    """A copy of the environment, for subprocesses that must share this catalogue."""
    return {key: value for key, value in os.environ.items()}
