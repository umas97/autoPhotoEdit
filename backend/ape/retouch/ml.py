# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The magic eraser's optional engine: LaMa, an ONNX inpainting model (docs/SPEC_rimozione.md 5).

LaMa int8 from the OpenCV Zoo, downloaded on request and verified (section 17),
run in a worker on CPU, loaded on first use and dropped after five idle
minutes (``jobs/limits.release_idle_models``).

LaMa sees photographs as the web shows them -- 8-bit sRGB -- at 512 px. So the
area and a square of context around it are taken from the linear frame,
brought to display by a plain sRGB curve after an exposure that puts the
brightest of the context near white (no tone mapping: it only has to be
invertible), filled, and brought back to linear light. Only the hole is taken
from the model; the context stays the photo's.

The result becomes a patch like the classic engine's (``retouch_store.Patch``):
the structure at the model's resolution, low-passed, and for every hole pixel
the place of the known photo its texture looks most like -- found with the
same PatchMatch (``nnf.py``) run on the model's fill -- so the render adds the
real grain at every resolution. The model gives the same bytes for the same
input (two runs compared, four threads, sequential execution); the seed moves
how much context it is shown, so "Altra variante" is another fill.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from ..models_registry import Unavailable, entry_for, feature_status, model_path
from ..pipeline.colorspace import REC2020_TO_XYZ, XYZ_TO_RGB, OutputSpace, _srgb_eotf, _srgb_oetf
from ..pipeline.ops.erase import STRUCTURE_SIGMA, AreaAlpha
from ..retouch_store import Patch
from .classic import hole_region
from .nnf import Level, patchmatch

__all__ = ["FEATURE", "ML_VERSION", "available", "fill_ml", "model_id", "release_if_idle"]

_log = logging.getLogger(__name__)

FEATURE = "inpaint"

#: Bumped when what this module does around the model changes. 3: the edge
#: membrane of 2 is gone -- where a sharp edge (a shadow, a bench) crossed the
#: outline it multiplied the fill by 0.4 to 3.4 and left a grey fog.
ML_VERSION = 3

#: The model's side, fixed by its export.
_SIDE = 512

#: Four threads: a fill is a gesture the user waits for, like the sky model's.
#: 2.0 s at 512 px, against 4.6 on one (measured).
THREADS = 4

IDLE_UNLOAD_S = 300.0

#: Context around the hole, as a multiple of its size; the seed adds a quarter
#: at a time, up to three, which is what makes another variant differ.
_CONTEXT = 2.0
_CONTEXT_STEP = 0.25

#: The brightest context maps here before the sRGB curve: highlights keep a
#: little room instead of clipping flat.
_WHITE = 0.9

_TO_SRGB = (XYZ_TO_RGB[OutputSpace.SRGB] @ REC2020_TO_XYZ).astype(np.float32)
_FROM_SRGB = np.linalg.inv(_TO_SRGB).astype(np.float32)

_REASONS = {
    Unavailable.RUNTIME_MISSING: "onnxruntime non è installato (extra «ml»)",
    Unavailable.NOT_PINNED: "il modello non ha un checksum verificato",
    Unavailable.MODEL_MISSING: (
        "il modello IA della gomma non è scaricato: scaricalo dal pannello Rimozione"
    ),
    Unavailable.CHECKSUM_MISMATCH: (
        "il file del modello IA non corrisponde al checksum ed è stato scartato"
    ),
}

_lock = threading.Lock()
_session: Any = None
_last_used = 0.0


def model_id() -> str:
    """The model and this module's version: part of every IA fill's key."""
    entry = entry_for(FEATURE)
    digest = (entry.sha256 or "none")[:16] if entry else "none"
    return f"{entry.name if entry else 'none'}:{digest}:{ML_VERSION}"


def available() -> tuple[bool, str | None]:
    ok, reason, _entry = feature_status(FEATURE)
    return ok, None if ok else _REASONS.get(reason, "motore IA non disponibile")


def _get_session() -> Any:
    global _session, _last_used
    with _lock:
        if _session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = THREADS
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.log_severity_level = 3
            # The arena stays on: without it this model peaked at 1.26 GB
            # instead of 0.83 (the opposite of the segmentation models).
            _session = ort.InferenceSession(
                str(model_path(entry_for(FEATURE))), options, providers=["CPUExecutionProvider"]
            )
            _log.info("modello IA della gomma caricato")
        _last_used = time.monotonic()
        return _session


def release_if_idle(now: float | None = None, *, force: bool = False) -> bool:
    """Drop the session idle for five minutes (or now, ``force``). True if dropped."""
    global _session
    now = time.monotonic() if now is None else now
    with _lock:
        if _session is None or (not force and now - _last_used < IDLE_UNLOAD_S):
            return False
        _session = None
    _log.info("modello IA della gomma scaricato dalla memoria")
    return True


def _encode(linear: np.ndarray) -> tuple[np.ndarray, float]:
    srgb = np.clip(linear @ _TO_SRGB.T, 0.0, None)
    gain = _WHITE / max(float(np.percentile(srgb, 99.5)), 1e-4)
    display = _srgb_oetf(np.clip(srgb * gain, 0.0, 1.0))
    return np.clip(display, 0.0, 1.0).astype(np.float32), gain


def _decode(display: np.ndarray, gain: float) -> np.ndarray:
    srgb = _srgb_eotf(np.clip(display, 0.0, 1.0)) / np.float32(gain)
    return (srgb @ _FROM_SRGB.T).astype(np.float32)


def _context_box(hole: np.ndarray, box: tuple[int, int, int, int], shape: tuple[int, int],
                 seed: int) -> tuple[int, int, int, int]:
    """A square around the hole, in frame pixels, as large as the seed says."""
    y0, _y1, x0, _x1 = box
    ys, xs = np.nonzero(hole)
    cy, cx = y0 + (ys.min() + ys.max()) / 2.0, x0 + (xs.min() + xs.max()) / 2.0
    size = max(int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1)
    side = int(math.ceil(size * (_CONTEXT + _CONTEXT_STEP * (seed % 4))))
    side = max(side, 64)
    h, w = shape
    top = int(np.clip(round(cy - side / 2), 0, max(0, h - side)))
    left = int(np.clip(round(cx - side / 2), 0, max(0, w - side)))
    return top, min(h, top + side), left, min(w, left + side)


def fill_ml(
    frame: np.ndarray,
    area: AreaAlpha,
    seed: int,
    progress: Callable[[float], None] | None = None,
) -> Patch:
    """Fill ``area`` of ``frame`` with LaMa. Same contract as ``classic.fill_classic``.

    Raises:
        FillUnavailable: the model is not here, or the runtime is missing.
    """
    from .fills import FillUnavailable

    ok, reason = available()
    if not ok:
        raise FillUnavailable(reason or "motore IA non disponibile")
    h, w = frame.shape[:2]
    hole_small, box = hole_region((h, w), area)
    full_hole = np.zeros((h, w), bool)
    full_hole[box[0] : box[1], box[2] : box[3]] = hole_small
    top, bottom, left, right = _context_box(hole_small, box, (h, w), seed)
    crop = frame[top:bottom, left:right]
    hole = full_hole[top:bottom, left:right]
    ch, cw = hole.shape
    side = max(ch, cw)

    display, gain = _encode(crop)
    display[hole] = 0.0
    square = np.zeros((side, side, 3), np.float32)
    square[:ch, :cw] = display
    mask = np.ones((side, side), np.float32)
    mask[:ch, :cw] = hole
    image_in = cv2.resize(square, (_SIDE, _SIDE), interpolation=cv2.INTER_AREA)
    mask_in = cv2.resize(mask, (_SIDE, _SIDE), interpolation=cv2.INTER_AREA) > 0.0
    mask_in = cv2.dilate(mask_in.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    if progress is not None:
        progress(0.1)
    out = _get_session().run(
        None,
        {"image": image_in.transpose(2, 0, 1)[None], "mask": mask_in[None, None]},
    )[0][0].transpose(1, 2, 0)
    if progress is not None:
        progress(0.7)
    filled = cv2.resize(
        np.clip(out, 0.0, 255.0) / 255.0, (side, side), interpolation=cv2.INTER_CUBIC
    )
    result = crop.copy()
    result[hole] = _decode(filled[:ch, :cw], gain)[hole]

    # The patch at the model's resolution: finer than that it knows nothing.
    scale = min(1.0, _SIDE / side)
    size = (max(8, round(cw * scale)), max(8, round(ch * scale)))
    work = cv2.resize(result, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else result
    work_hole = cv2.resize(hole.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.0
    offsets = _texture_sources(work, work_hole, seed)
    structure = cv2.GaussianBlur(work, (0, 0), STRUCTURE_SIGMA, borderType=cv2.BORDER_REFLECT)
    if progress is not None:
        progress(1.0)
    return Patch(bbox=(left / w, top / h, right / w, bottom / h), structure=structure,
                 offsets=offsets)


def _texture_sources(work: np.ndarray, hole: np.ndarray, seed: int) -> np.ndarray:
    """For each hole pixel, where the known photo looks like the model's fill."""
    rng = np.random.default_rng([int(seed), ML_VERSION])
    encoded = np.cbrt(np.maximum(work, 0.0) + np.float32(1e-4)).astype(np.float32)
    level = Level(encoded, hole)
    level.nnf = level.random_sources(len(level.targets), rng)
    patchmatch(level, encoded, rng, 4)
    offsets = np.zeros(hole.shape + (2,), dtype=np.int16)
    ys, xs = np.nonzero(hole)
    rows = level.index[ys, xs]
    offsets[ys, xs, 0] = level.nnf[rows, 0] - ys
    offsets[ys, xs, 1] = level.nnf[rows, 1] - xs
    return offsets
