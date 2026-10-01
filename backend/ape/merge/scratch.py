# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Temporary full-resolution frames of a merge, on disk in the cache.

A focus stack of eight 24 MP frames is 2.3 GB in single precision and 1.2 GB
in half: neither fits the worker budget of section 26 in RAM. The aligned
frames are written once, as half-precision memory maps, and read back a tile
at a time; the page cache holds what is being worked on and the kernel is free
to drop the rest.

The folder lives under the cache, never next to the sources, and goes away
with the ``with`` block. A worker killed in the middle leaves one behind: the
next merge removes folders whose process is gone.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from ..safety import assert_outside_source

__all__ = ["Scratch"]

_PREFIX = "merge-"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sweep(root: Path) -> None:
    """Remove the folders of merges whose worker died before cleaning up."""
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if not entry.name.startswith(_PREFIX):
            continue
        pid_text = entry.name[len(_PREFIX) :].split("-", 1)[0]
        if pid_text.isdigit() and not _alive(int(pid_text)):
            shutil.rmtree(entry, ignore_errors=True)


class Scratch:
    """A folder of memory-mapped frames for the length of a ``with`` block."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or get_settings().cache_dir / "tmp"
        self.path: Path | None = None
        self._count = 0

    def __enter__(self) -> Scratch:
        _sweep(self._root)
        folder = self._root / f"{_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:8]}"
        assert_outside_source(folder)
        folder.mkdir(parents=True, exist_ok=False)
        self.path = folder
        return self

    def frame(self, shape: tuple[int, ...], dtype: Any = np.float16) -> np.memmap:
        """A new zero-filled array on disk, readable and writable."""
        if self.path is None:
            raise RuntimeError("Scratch.frame fuori dal blocco with")
        self._count += 1
        return np.memmap(self.path / f"{self._count:03d}.bin", dtype=dtype, mode="w+", shape=shape)

    def __exit__(self, *_exc: Any) -> None:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
            self.path = None
