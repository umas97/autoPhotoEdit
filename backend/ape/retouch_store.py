# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The fills of the magic eraser: ``$XDG_DATA_HOME/autophotoedit/retouch/``.

User data, like the painted masks, and for a stronger reason: a fill made by
the optional model cannot be made again once the model is gone. Never evicted,
never removed automatically, counted apart in the Settings screen and named by
``autophotoedit backup``.

A patch is **named by the key of what made it** (``retouch/fills.py``): the
area and its expansion, the engine and its version, the seed, and the
fingerprint of everything upstream -- the proxy, the lens correction, the
removals before it. The fill is a deterministic function of those, so the
name *is* the content, as for the masks, and it can be looked up before it is
computed: a gesture undone finds its patch again, an export finds the one the
editor made. The file also carries the SHA-256 of its payload, checked on
every read.

The format is small and fixed, so that no pickle is ever loaded::

    b"APEPATCH" 0x01 | uint32 LE header length | JSON header | payload

with the payload the structure, float16 ``(h, w, 3)``, then the offsets,
int16 ``(h, w, 2)``, both C order.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import get_settings
from .safety import guarded_open

__all__ = ["Patch", "PatchError", "exists", "load", "path_for", "save"]

_MAGIC = b"APEPATCH\x01"
_NAME = re.compile(r"^[0-9a-f]{64}$")


class PatchError(ValueError):
    """A patch that is missing, damaged or not ours."""


@dataclass(frozen=True)
class Patch:
    """What the eraser's worker made, on the proxy, for one area.

    ``bbox`` is ``(x0, y0, x1, y1)`` in frame coordinates: the region the
    arrays cover, the area plus the margin its sources came from.
    ``structure`` is that region in linear Rec.2020, low-passed -- the large
    shapes of the fill. ``offsets`` say, for every pixel of the filled area,
    where its texture comes from, as ``(dy, dx)`` in pixels of the arrays:
    the render copies the fine detail from there, at its own resolution.
    """

    bbox: tuple[float, float, float, float]
    structure: np.ndarray
    offsets: np.ndarray


def _directory() -> Path:
    return get_settings().retouch_dir


def path_for(name: str) -> Path:
    if not _NAME.match(name):
        raise PatchError("nome di patch non valido")
    return _directory() / f"{name}.patch"


def exists(name: str) -> bool:
    return path_for(name).is_file()


def save(name: str, patch: Patch) -> None:
    """Write a patch under its key. Idempotent: an existing one is kept."""
    target = path_for(name)
    if target.is_file():
        return
    structure = np.ascontiguousarray(patch.structure, dtype=np.float16)
    offsets = np.ascontiguousarray(patch.offsets, dtype=np.int16)
    h, w = structure.shape[:2]
    if structure.shape != (h, w, 3) or offsets.shape != (h, w, 2):
        raise PatchError("forma della patch non valida")
    payload = structure.tobytes() + offsets.tobytes()
    header = json.dumps(
        {
            "bbox": [float(v) for v in patch.bbox],
            "shape": [h, w],
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        sort_keys=True,
    ).encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Aside, then renamed: a crash never leaves a truncated patch under a key
    # that promises a whole one.
    partial = target.with_name(f".{name}.{os.getpid()}.part")
    with guarded_open(partial, "wb") as handle:
        handle.write(_MAGIC + struct.pack("<I", len(header)) + header + payload)
    os.replace(partial, target)


@lru_cache(maxsize=16)
def load(name: str) -> Patch:
    """The patch, read-only arrays. Raises ``PatchError`` if it is gone or damaged.

    Cached: every render of an erased photo reads it. Safe because a name
    never changes its content.
    """
    try:
        data = path_for(name).read_bytes()
    except FileNotFoundError as exc:
        raise PatchError("la patch della rimozione non è più nella sua cartella") from exc
    if not data.startswith(_MAGIC) or len(data) < len(_MAGIC) + 4:
        raise PatchError("la patch della rimozione è danneggiata")
    start = len(_MAGIC) + 4
    (length,) = struct.unpack("<I", data[len(_MAGIC) : start])
    try:
        header = json.loads(data[start : start + length])
        h, w = (int(v) for v in header["shape"])
        bbox = tuple(float(v) for v in header["bbox"])
    except (ValueError, KeyError, TypeError) as exc:
        raise PatchError("la patch della rimozione è danneggiata") from exc
    payload = data[start + length :]
    if hashlib.sha256(payload).hexdigest() != header.get("sha256") or len(payload) != h * w * 10:
        raise PatchError("la patch della rimozione è danneggiata")
    structure = np.frombuffer(payload, dtype=np.float16, count=h * w * 3).reshape(h, w, 3)
    offsets = np.frombuffer(payload, dtype=np.int16, offset=h * w * 6).reshape(h, w, 2)
    linear = structure.astype(np.float32)
    linear.setflags(write=False)
    return Patch(bbox=bbox, structure=linear, offsets=offsets)  # type: ignore[arg-type]
