# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""HDR merge of an exposure bracketing, in linear light (section 25.3).

The RAW is the easy case of HDR: its values are already proportional to light,
so there is no response curve to estimate. Each frame, divided by its exposure
relative to the reference, is a measurement of the same scene radiance; the
merge is their weighted average, and the sigmoid of section 6.2 -- with its
white point further out -- is the tone mapping. No local operator: the program
must not produce the 2010 "HDR look".

**Weights.** A frame measures a pixel well when the pixel is neither clipped
nor lost in noise. Clipping is sharp and known (saturation is 1.0 in the
decoder's normalisation), so the weight fades to zero just below it. Noise is
gradual: photon noise makes the signal-to-noise ratio grow with the exposure,
so every weight is multiplied by the frame's gain -- the longest exposure that
did not clip is the cleanest, and it dominates wherever it is valid.

**Gains** come from EXIF and are *checked* on the pixels both frames exposed
well: nominal and real exposure often differ by a tenth or two of a stop (the
shutter's actual timing, aperture rounding), and a wrong gain shows up as a
band where the weights hand over from one frame to the next.

**Ghosts.** Where a frame disagrees with the reference by more than noise can
explain -- leaves in the wind, someone walking -- it is left out and the
reference alone speaks for that area (section 25.3.5). Disagreement is measured
on the reduced frames already used for the alignment, so it costs nothing at
full resolution and its mask is smooth by construction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..pipeline.colorspace import channel_max
from .align import ALIGN_EDGE, align, for_matching, residual_shift
from .errors import MergeFailure
from .warp import shrink, warp_bands, warp_whole

__all__ = ["HdrOptions", "HdrReport", "merge_hdr"]

#: The weight fades from 1 at this fraction of saturation to 0 at the next. The
#: brightest channel of a colour clipped by one sensor channel lands a little
#: off 1.0 after the camera matrix; starting the fade at 0.75 keeps those
#: pixels out without giving away the well-exposed highlights.
_CLIP_START, _CLIP_END = 0.75, 0.92

#: What the reference always keeps: where every other frame is clipped or a
#: ghost, the reference's own value is the answer, never a division by zero.
_REFERENCE_FLOOR = 1e-3

#: The darkest frame's own floor, a little higher: where *every* frame is
#: clipped, the darkest one is the closest to the truth.
_DARKEST_FLOOR = 2e-3

#: The measured gain replaces the nominal one only when it is within this of it
#: and was measured on enough pixels. Further off is a misread EXIF or a moving
#: scene, and the nominal value is the safer of two unreliable numbers.
_GAIN_TRUST_EV = 0.5
_GAIN_MIN_PIXELS = 2000

#: Luminance band, as a fraction of saturation at the reference's exposure,
#: where both frames of a pair count as well exposed for the gain check.
_WELL_EXPOSED = (0.02, 0.6)

#: A ghost is a disagreement of more than this many stops between a frame and
#: the reference, on reduced frames where the noise has been averaged down.
#: Measured on the fixtures stepped by -2/0/+2 EV with the noise model of
#: ``tests/test_merge_hdr.py``: static pixels stay under 0.15 EV, a patch moved
#: by 20 px goes over 1.
_GHOST_EV = 0.4
#: Offset added to both luminances before the log ratio, so the deep shadows,
#: where the darkest frame is mostly noise, do not read as ghosts.
_GHOST_FLOOR = 0.004

#: A reference aligned to itself is the identity; other frames are expected to
#: land within this many full-resolution pixels. Past it the report says so.
MAX_RESIDUAL_PX = 3.0


@dataclass(frozen=True)
class HdrOptions:
    deghost: bool = True


@dataclass
class HdrReport:
    frames: list[dict] = field(default_factory=list)
    #: Largest alignment residual among the frames, in full-resolution pixels.
    residual_px: float = 0.0
    #: Fraction of the frame where some frame was left out as a ghost.
    ghost_fraction: float = 0.0

    def as_json(self) -> dict:
        return {
            "frames": self.frames,
            "residual_px": round(self.residual_px, 2),
            "ghost_fraction": round(self.ghost_fraction, 4),
            "misaligned": self.residual_px > MAX_RESIDUAL_PX,
        }


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2627 + rgb[..., 1] * 0.678 + rgb[..., 2] * 0.0593


def _weight(band: np.ndarray, gain: float, floor: float) -> np.ndarray:
    """Per-pixel weight of a band in its own exposure (see the module docstring)."""
    peak = channel_max(band)
    fade = np.clip((_CLIP_END - peak) / (_CLIP_END - _CLIP_START), 0.0, 1.0)
    # Smoothstep: no kink where the fade begins, which would show as a contour
    # in a smooth sky crossing the threshold.
    fade = fade * fade * (3.0 - 2.0 * fade)
    return (fade * np.float32(gain) + np.float32(floor)).astype(np.float32)


def _measured_gain(reference: np.ndarray, frame: np.ndarray, nominal: float) -> tuple[float, int]:
    """Median ratio frame / reference over pixels both exposed well."""
    ref_y = _luminance(reference)
    frm_y = _luminance(frame) / nominal
    low, high = _WELL_EXPOSED
    both = (
        (ref_y > low) & (ref_y < high) & (frm_y > low) & (frm_y < high)
        & (channel_max(reference) < _CLIP_START) & np.isfinite(frm_y)
        & (channel_max(np.nan_to_num(frame, nan=1.0)) < _CLIP_START)
    )
    count = int(both.sum())
    if count < _GAIN_MIN_PIXELS:
        return nominal, count
    ratio = float(np.median(frm_y[both] / ref_y[both]))
    return nominal * ratio, count


def _ghost_mask(reference: np.ndarray, warped: np.ndarray, gain: float) -> np.ndarray:
    """Where ``warped`` (already at the reference's geometry) is not the same scene.

    Returns a float32 map in 0..1 at the reduced size, 1 = leave the frame out.
    Only pixels both frames exposed validly can disagree; elsewhere the weights
    already decide.
    """
    # Out-of-gamut colours have a slightly negative luminance: clamped, or the
    # ratio below goes negative and its log is NaN.
    ref_y = np.maximum(_luminance(reference), 0.0)
    frm_y = np.maximum(_luminance(np.nan_to_num(warped, nan=0.0)), 0.0) / gain
    valid = (
        (channel_max(reference) < _CLIP_END)
        & (channel_max(np.nan_to_num(warped, nan=1.0)) < _CLIP_END)
    )
    difference = np.abs(np.log2((frm_y + _GHOST_FLOOR) / (ref_y + _GHOST_FLOOR)))
    ghost = ((difference > _GHOST_EV) & valid).astype(np.uint8)
    # Isolated pixels are noise, not a subject: open, then grow a margin so the
    # fade into the reference-only area happens outside the moving object.
    size = max(3, round(max(ghost.shape) / 400) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    ghost = cv2.morphologyEx(ghost, cv2.MORPH_OPEN, kernel)
    ghost = cv2.dilate(ghost, kernel, iterations=3).astype(np.float32)
    sigma = max(1.0, max(ghost.shape) / 500)
    return np.clip(cv2.GaussianBlur(ghost, (0, 0), sigma) * 1.5, 0.0, 1.0)


@dataclass(frozen=True)
class _Prepared:
    matrix: np.ndarray  # output -> frame, full resolution
    gain: float
    ghost: np.ndarray | None  # reduced size, 0..1


def _prepare(
    ref_small: np.ndarray, small: np.ndarray, scale: float, nominal: float, options: HdrOptions,
    index: int,
) -> tuple[_Prepared, dict]:
    ref_grey = for_matching(ref_small, 1.0)
    grey = for_matching(small, nominal)
    usable = ((ref_grey > 8) & (ref_grey < 240)).astype(np.uint8)
    found = align(ref_grey, grey, model="homography", mask=usable)
    if found is None:
        raise MergeFailure(
            f"lo scatto {index + 1} non si allinea al riferimento: "
            "troppo diverso o troppo povero di dettagli"
        )
    size = (ref_small.shape[1], ref_small.shape[0])
    warped = warp_whole(small, found.matrix, size)
    residual = residual_shift(ref_grey, for_matching(np.nan_to_num(warped, nan=0.0), nominal))
    gain, pixels = _measured_gain(ref_small, warped, nominal)
    if abs(math.log2(gain / nominal)) > _GAIN_TRUST_EV:
        gain = nominal
    ghost = _ghost_mask(ref_small, warped, gain) if options.deghost else None
    report = {
        "ev_nominal": round(math.log2(nominal), 2),
        "ev_used": round(math.log2(gain), 2),
        "gain_pixels": pixels,
        "residual_px": round(residual * scale, 2),
        "inliers": found.inliers,
        "refined": found.refined,
        "ghost_fraction": 0.0 if ghost is None else round(float((ghost > 0.5).mean()), 4),
    }
    return _Prepared(found.to_full(scale), gain, ghost), report


def merge_hdr(
    reference: np.ndarray,
    others: Sequence[tuple[Callable[[], np.ndarray], float]],
    options: HdrOptions | None = None,
    *,
    progress: Callable[[float], None] | None = None,
) -> tuple[np.ndarray, HdrReport]:
    """Merge a bracketing onto its reference frame.

    Args:
        reference: the reference frame, linear, as decoded. It becomes the
            accumulator and is not preserved.
        others: ``(load, ev)`` for every other frame: ``load()`` decodes it
            (with the reference's white balance, section 25.3.1) and ``ev`` is
            its nominal exposure in stops relative to the reference. Frames are
            loaded one at a time and released after use.
        options: :class:`HdrOptions`.

    Returns:
        The merged frame -- linear, scene-referred, at the reference's exposure
        and geometry, values above 1.0 where the darker frames saw what the
        reference clipped -- and a :class:`HdrReport`.

    Raises:
        MergeFailure: a frame cannot be aligned.
    """
    options = options or HdrOptions()
    height, width = reference.shape[:2]
    ref_small, scale = shrink(reference, ALIGN_EDGE)
    ref_small = ref_small.copy() if ref_small is reference else ref_small
    gains = [2.0 ** ev for _, ev in others]
    darkest = min(gains) if gains and min(gains) < 1.0 else 1.0

    total = np.zeros((height, width), dtype=np.float32)
    ghost_any = np.zeros(ref_small.shape[:2], dtype=np.float32)
    report = HdrReport(frames=[{"ev_nominal": 0.0, "ev_used": 0.0, "reference": True}])
    accumulator = reference
    floor = _DARKEST_FLOOR if darkest == 1.0 else _REFERENCE_FLOOR
    for top in range(0, height, 256):
        band = accumulator[top : top + 256]
        weight = _weight(band, 1.0, floor)
        total[top : top + 256] = weight
        band *= weight[..., None]

    for index, ((load, _ev), nominal) in enumerate(zip(others, gains, strict=True)):
        frame = load()
        small, _ = shrink(frame, ref_small.shape[1] if width >= height else ref_small.shape[0])
        prepared, entry = _prepare(ref_small, small, scale, nominal, options, index)
        report.frames.append(entry)
        report.residual_px = max(report.residual_px, entry["residual_px"])
        ghost_full = None
        if prepared.ghost is not None:
            ghost_any = np.maximum(ghost_any, prepared.ghost)
            ghost_full = cv2.resize(prepared.ghost, (width, height), interpolation=cv2.INTER_LINEAR)
        frame_floor = _DARKEST_FLOOR if nominal == darkest else 0.0
        for top, band in warp_bands(frame, prepared.matrix, (width, height)):
            valid = np.isfinite(band[..., 0])
            band = np.nan_to_num(band, nan=0.0, copy=False)
            weight = _weight(band, nominal, frame_floor)
            if ghost_full is not None:
                weight *= 1.0 - ghost_full[top : top + band.shape[0]]
            weight *= valid
            rows = slice(top, top + band.shape[0])
            total[rows] += weight
            accumulator[rows] += band * (weight / np.float32(prepared.gain))[..., None]
        del frame, ghost_full
        if progress is not None:
            progress((index + 1) / (len(others) + 1))

    for top in range(0, height, 256):
        accumulator[top : top + 256] /= np.maximum(total[top : top + 256], 1e-12)[..., None]
    report.ghost_fraction = float((ghost_any > 0.5).mean())
    return accumulator, report
