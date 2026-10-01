# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The painted masks: 8-bit PNGs in ``$XDG_DATA_HOME/autophotoedit/masks/``.

Not cache (section 20.3): a brush stroke is the one thing the program keeps
that cannot be regenerated, so these files live beside the catalogue, are
counted apart and are never evicted.

Each raster is named by the SHA-256 of its PNG and never changes once written.
A stroke makes a new raster and a new ``EditVersion`` pointing at it, so the
history of section 23 -- undo included -- works for masks exactly as for
sliders, and two photos painted alike share one file.

Whatever arrives (the browser sends an RGBA canvas) is stored as a single
8-bit channel: the alpha if there is one, the first channel otherwise.
"""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import get_settings
from .safety import guarded_open

__all__ = ["MAX_BYTES", "MAX_EDGE", "RasterError", "exists", "load", "path_for", "save"]

#: A 4096 px mask is finer than any brush a mouse draws, and a PNG of one is a
#: few hundred KB even when busy. Anything larger is not a mask.
MAX_EDGE = 4096
MAX_BYTES = 32 * 1024 * 1024

_NAME = re.compile(r"^[0-9a-f]{64}$")


class RasterError(ValueError):
    """The upload is not a mask this program can use."""


def _directory() -> Path:
    return get_settings().masks_dir


def path_for(name: str) -> Path:
    if not _NAME.match(name):
        raise RasterError("nome di maschera non valido")
    return _directory() / f"{name}.png"


def exists(name: str) -> bool:
    return path_for(name).is_file()


def _single_channel(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16:
        image = (image >> 8).astype(np.uint8)
    if image.dtype != np.uint8:
        raise RasterError("la maschera deve essere un PNG a 8 o 16 bit")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return np.ascontiguousarray(image[..., 3])
    if image.ndim == 3 and image.shape[2] in (1, 3):
        return np.ascontiguousarray(image[..., 0])
    raise RasterError("formato della maschera non riconosciuto")


def save(data: bytes) -> str:
    """Store an uploaded mask. Returns its name. Idempotent for equal content.

    Raises:
        RasterError: not a PNG, too large, or of an unusable layout.
    """
    if len(data) > MAX_BYTES:
        raise RasterError("maschera troppo grande")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RasterError("la maschera deve essere un PNG")
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RasterError("PNG della maschera illeggibile")
    raster = _single_channel(decoded)
    if max(raster.shape) > MAX_EDGE or min(raster.shape) < 1:
        raise RasterError(f"la maschera deve avere il lato lungo entro {MAX_EDGE} px")

    ok, encoded = cv2.imencode(".png", raster, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RasterError("impossibile codificare la maschera")
    payload = encoded.tobytes()
    name = hashlib.sha256(payload).hexdigest()
    target = path_for(name)
    if target.is_file():
        return name
    # Written aside and renamed: a crash never leaves a truncated mask under a
    # name that promises its content.
    partial = target.with_name(f".{name}.{os.getpid()}.part")
    with guarded_open(partial, "wb") as handle:
        handle.write(payload)
    os.replace(partial, target)
    return name


@lru_cache(maxsize=32)
def load(name: str) -> np.ndarray:
    """The raster, ``(H, W)`` uint8. Raises ``RasterError`` if it is gone.

    Cached: a slider moved on a painted photo reads the same file every render.
    Safe because a name is its content.
    """
    target = path_for(name)
    try:
        payload = target.read_bytes()
    except FileNotFoundError as exc:
        raise RasterError(
            "la maschera disegnata di questa foto non è più nella cartella delle maschere"
        ) from exc
    if hashlib.sha256(payload).hexdigest() != name:
        raise RasterError("la maschera disegnata è danneggiata")
    raster = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if raster is None:
        raise RasterError("la maschera disegnata è danneggiata")
    raster.setflags(write=False)
    return raster
