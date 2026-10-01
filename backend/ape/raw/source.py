# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where the pixels of a photo come from.

A photo is either a RAW in the source folder or a merge of several, whose
pixels are an intermediate in the cache (section 25.1). Everything that decodes
a photo asks this module for the file and hands it to ``decode_linear``, which
reads either; nothing else needs to know which of the two it is.

The one difference that cannot be hidden is what happens when the file is
gone. A RAW that left the source folder is *missing*: the user has to bring it
back. An intermediate that left the cache is *regenerable*: its members are
still there, and running the merge again gives it back bit for bit. So a worker
that finds the file absent asks :func:`make_available`, which rebuilds what can
be rebuilt and reports the rest as missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .intermediate import is_intermediate

__all__ = ["has_raw_file", "make_available", "pixels_of"]


def pixels_of(photo: Any) -> Path | None:
    """The file ``decode_linear`` should read for this photo, if it has one."""
    if photo.path:
        return Path(photo.path)
    if photo.intermediate_path:
        return Path(photo.intermediate_path)
    return None


def has_raw_file(photo: Any) -> bool:
    """Whether a RAW exists for this photo, which is what an XMP sidecar describes."""
    return bool(photo.path)


def make_available(maker: Any, photo_id: int, source: Path) -> Path | None:
    """The photo's file, rebuilt if it is a regenerable intermediate.

    Worker-side only: rebuilding a merge takes seconds to minutes. Returns the
    path to read, or ``None`` when the file is gone for good (a RAW that left
    the source folder), which the caller records as missing.
    """
    if source.is_file():
        return source
    if not is_intermediate(source):
        return None
    from ..merge.rebuild import rebuild_intermediate

    return rebuild_intermediate(maker, photo_id)
