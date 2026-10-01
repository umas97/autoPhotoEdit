# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deciding whether two files are the same photograph (docs/SPEC.md section 15).

A photo is identified by its content, never by its path. That is what makes a
re-import free, a rename harmless and a copy not a second photo -- and it is why
this is a module of its own rather than three lines inside the importer.

Identity is established in two steps, as section 15 prescribes:

1. a **quick key** -- the file size and the SHA-256 of its first mebibyte. One
   read of 1 MB instead of 24, which is the difference between importing a
   thousand photos in seconds and in minutes;
2. only when two files share a quick key, the **full SHA-256** of both. Then
   either they really are the same file, or they are not and both move to the
   full hash.

The two live in the same column of ``Photo.hash``, distinguished by a prefix
(``q:`` and ``s:``), so a photo's identity is one indexed string and not a pair
of columns with a rule about which to believe.

Every read here opens the file ``'rb'`` and nothing else (section 2).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["content_key", "full_content_hash"]

#: How much of a file goes into the quick key.
_PREFIX_BYTES = 1024 * 1024

_READ_CHUNK = 1024 * 1024


def content_key(path: Path) -> str:
    """Quick content identity: size plus the SHA-256 of the first mebibyte.

    Two different photographs cannot collide here in practice -- an ARW's first
    mebibyte carries the header, the metadata and the start of the embedded
    JPEG -- but the caller must still treat equality as *probable* identity and
    confirm it with :func:`full_content_hash`.
    """
    digest = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as handle:
        digest.update(handle.read(_PREFIX_BYTES))
    return f"q:{size}:{digest.hexdigest()}"


def full_content_hash(path: Path) -> str:
    """SHA-256 of the whole file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            digest.update(chunk)
    return f"s:{digest.hexdigest()}"
