# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focus stacking (section 25.4): frames that differ in where they are sharp.

1. **Alignment with scale.** Focus breathing changes the magnification from
   frame to frame, so each frame is registered to the reference with a
   similarity -- shift, rotation, scale -- found by ORB and refined by ECC
   (``align.py``), on frames reduced to :data:`~.align.ALIGN_EDGE`.
2. **Fusion.** Laplacian pyramids, at every level the coefficients of the
   locally sharpest frames with a soft transition (``pyramid.py``), a tile at a
   time.
3. **Coverage.** Where no frame is in focus is measured and reported, with an
   overlay for the preview (``coverage.py``), instead of being passed off as
   a finished photo.

At full resolution the aligned frames live in half-precision memory maps in
the cache (``scratch.py``): eight 24 MP frames would not fit the worker in RAM,
and decoding every frame twice -- once to align, once per tile -- would cost
more than writing it once.

Colour space: linear, scene-referred RGB throughout, all frames decoded with
the reference's white balance, as for HDR (section 25.3.1). The merge keeps the
reference's geometry, so the lens stage still corrects it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from .align import ALIGN_EDGE, align, for_matching, residual_shift
from .coverage import WORK_EDGE, coverage_map
from .errors import MergeFailure
from .hdr import MAX_RESIDUAL_PX
from .pyramid import fuse_stack
from .scratch import Scratch
from .warp import shrink, warp_bands, warp_whole

__all__ = ["MAX_SCALE_CHANGE", "run"]

#: Largest change of magnification accepted between a frame and the reference.
#: Focus breathing is a few per cent even over the whole range of a macro lens;
#: more than this is a different framing, and the frames are not a stack.
MAX_SCALE_CHANGE = 0.12


#: Blur applied to both frames before the residual is measured, in pixels of
#: the alignment scale. A sharp tile against a defocused one gives phase
#: correlation spurious shifts of 2-3 px on the fixtures; blurred alike past
#: the defocus they agree to 0.7 px, and a real 2 px error still reads 2.3.
_RESIDUAL_BLUR = 2.0


def _comparable(grey: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(grey, (0, 0), _RESIDUAL_BLUR)


def _size_at(shape: tuple[int, ...], long_edge: int) -> tuple[int, int]:
    """``(width, height)`` that :func:`~.warp.shrink` reduces ``shape`` to."""
    height, width = shape[:2]
    scale = max(width, height) / long_edge
    if scale <= 1.0:
        return width, height
    return max(1, round(width / scale)), max(1, round(height / scale))


class _Frames:
    """The aligned frames: arrays in RAM for a preview, memory maps otherwise."""

    def __init__(self, scratch: Scratch | None) -> None:
        self.scratch = scratch
        self.pixels: list[Any] = []
        self.valid: list[Any] = []

    def add_reference(self, rgb: np.ndarray) -> None:
        if self.scratch is None:
            self.pixels.append(rgb)
        else:
            stored = self.scratch.frame(rgb.shape)
            for top in range(0, rgb.shape[0], 256):
                stored[top : top + 256] = rgb[top : top + 256]
            self.pixels.append(stored)
        self.valid.append(None)

    def add_warped(self, frame: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> None:
        width, height = size
        if self.scratch is None:
            # Bicubic, as at full resolution: the bilinear warp of a reduced
            # frame is a blur the preview would show and the merge would not.
            warped = np.concatenate([band for _, band in warp_bands(frame, matrix, size)])
            self.valid.append(np.isfinite(warped[..., 0]))
            self.pixels.append(np.nan_to_num(warped, nan=0.0, copy=False))
            return
        pixels = self.scratch.frame((height, width, 3))
        valid = self.scratch.frame((height, width), dtype=np.uint8)
        for top, band in warp_bands(frame, matrix, size):
            rows = slice(top, top + band.shape[0])
            valid[rows] = np.isfinite(band[..., 0])
            pixels[rows] = np.nan_to_num(band, nan=0.0, copy=False)
        self.pixels.append(pixels)
        self.valid.append(valid)


def run(
    recipe: Any,
    *,
    preview: bool,
    progress: Callable[[float], None] | None,
    decode: Callable[..., Any],
) -> Any:
    """Stack ``recipe``'s members onto its reference.

    ``decode(member, preview=..., white_balance=...)`` is the engine's decoder.
    Returns a :class:`~.engine.MergeOutcome` whose report says how each frame
    was aligned, how much of the picture no frame covers, and -- for a preview
    -- carries the coverage overlay as ``coverage_image``.

    Raises:
        MergeFailure: a frame cannot be registered to the reference, or its
            framing is too different for a stack.
    """
    from .engine import MergeOutcome

    def step(fraction: float) -> None:
        if progress is not None:
            progress(fraction)

    reference_member = recipe.reference
    reference = decode(reference_member, preview=preview)
    balance = reference.camera.as_shot_multipliers
    height, width = reference.rgb.shape[:2]
    ref_small, scale = shrink(reference.rgb, ALIGN_EDGE)
    ref_grey = for_matching(ref_small)
    small_size = (ref_small.shape[1], ref_small.shape[0])

    entries: list[dict] = []
    covers: list[np.ndarray] = []
    order = list(recipe.members)
    with Scratch() if not preview else contextlib.nullcontext() as scratch:
        frames = _Frames(scratch)
        for index, member in enumerate(order):
            if member.reference:
                frames.add_reference(reference.rgb)
                covers.append(ref_small)
                entries.append({"filename": member.filename, "reference": True, "scale": 1.0})
            else:
                frame = decode(member, preview=preview, white_balance=balance).rgb
                if frame.shape != reference.rgb.shape:
                    raise MergeFailure(f"{member.filename} ha dimensioni diverse dal riferimento")
                small, _ = shrink(frame, ALIGN_EDGE)
                found = align(ref_grey, for_matching(small), model="similarity")
                if found is None:
                    raise MergeFailure(
                        f"{member.filename} non si allinea al riferimento: "
                        "troppo pochi dettagli in comune"
                    )
                magnification = float(np.sqrt(abs(np.linalg.det(found.matrix[:2, :2]))))
                if abs(magnification - 1.0) > MAX_SCALE_CHANGE:
                    raise MergeFailure(
                        f"{member.filename} ha un'inquadratura diversa dal riferimento "
                        f"(scala {magnification:.2f}): non è uno stack"
                    )
                warped_small = warp_whole(small, found.matrix, small_size)
                matched = for_matching(np.nan_to_num(warped_small, nan=0.0))
                residual = residual_shift(_comparable(ref_grey), _comparable(matched))
                covers.append(warped_small)
                frames.add_warped(frame, found.to_full(scale), (width, height))
                del frame, small
                entries.append({
                    "filename": member.filename,
                    "scale": round(magnification, 4),
                    "residual_px": round(residual * scale, 2),
                    "inliers": found.inliers,
                    "refined": found.refined,
                })
            step(0.5 * (index + 1) / len(order))

        overlay_size = _size_at((height, width), WORK_EDGE)
        uncovered, overlay = coverage_map(covers, overlay_size=overlay_size)
        del covers
        reference_index = next(i for i, m in enumerate(order) if m.reference)
        if scratch is not None:
            reference.rgb = None  # the memory map holds it now
        out = np.empty((height, width, 3), dtype=np.float32)
        fuse_stack(
            frames.pixels, frames.valid, out, reference=reference_index,
            progress=lambda f: step(0.5 + 0.45 * f),
        )
    reference.rgb = out
    residual_px = max((e.get("residual_px", 0.0) for e in entries), default=0.0)
    report: dict[str, Any] = {
        "frames": entries,
        "residual_px": residual_px,
        "misaligned": residual_px > MAX_RESIDUAL_PX,
        "uncovered_fraction": uncovered,
    }
    if preview and uncovered > 0:
        report["coverage_image"] = overlay
    step(1.0)
    return MergeOutcome(decoded=reference, report=report, keep_lens=True)
