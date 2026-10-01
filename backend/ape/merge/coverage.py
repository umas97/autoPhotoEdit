# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Where no frame of a focus stack is in focus (section 25.4.4).

"Le zone in cui nessun frame è a fuoco vengono segnalate": the stack returns
its result *and* a map of where that result is still soft, instead of a photo
sharp in half its area and silent about the rest.

Softness is measured as a blur width, on edges, with the re-blur method of
Zhuo and Sim (2011): blurring an edge of width ``s`` by a known ``s0`` lowers
its gradient peak by ``R = sqrt(s^2 + s0^2) / s``, so ``s = s0 / sqrt(R^2 - 1)``.
A step edge gives the same answer whatever its contrast and whatever the
scene around it, which is what an absolute "is this in focus" needs; a ratio
of band energies depends on the texture's spectrum, and on the fixtures it
put in-focus grass and out-of-focus walls on the same side of any threshold.

The edges are *located* on a blurred copy, where a defocused edge still is
one, and *measured* at the peaks of the unblurred gradient. A region with no
edges gives no evidence and is never flagged: a clear sky is not incomplete.

Everything runs on frames reduced to :data:`WORK_EDGE`, the same size for the
preview and the full merge, so the two report the same map. Blur widths below
are in pixels of that size.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .warp import shrink

__all__ = ["WORK_EDGE", "coverage_map"]

#: Long edge the map is computed at: the preview's.
WORK_EDGE = 1024

#: Known blur of the re-blur method. One pixel: small enough that sharp edges
#: still give a ratio well above one, large enough to be measured on 8 bits.
_S0 = 1.0

#: Blur of the copy the edges are located on, and the gradient an edge needs
#: there, on the 0..255 gamma-encoded grey. Five pixels is past the widest
#: defocus the map is asked to see, so a soft edge is found as surely as a
#: sharp one; 8 is a step of about 40 levels seen through that blur. Wider
#: defocus erases the edges themselves, and the region gives no evidence.
_LOCATE_SIGMA, _LOCATE_GRADIENT = 5.0, 8.0

#: A region is soft when the sharpest frame's edges there are wider than this.
#: In focus the fixtures measure 0.5-0.9 px at 1024 px (1.4 for a frame that
#: was soft to begin with); defocused by 2 px, 1.7-2.6.
BLUR_LIMIT_PX = 1.6

#: Least density of edge samples in a region for its blur to be believed.
_MIN_EVIDENCE = 0.002

#: Colour of an uncovered pixel in the overlay, BGRA as OpenCV writes it: red,
#: see-through, like the selection overlay of the masks.
_OVERLAY = (60, 60, 220, 120)


def _grey(linear: np.ndarray) -> np.ndarray:
    grey = linear[..., 0] * 0.2627 + linear[..., 1] * 0.678 + linear[..., 2] * 0.0593
    grey = np.clip(np.nan_to_num(grey, nan=0.0), 0.0, 1.0)
    return (grey ** (1.0 / 2.2) * 255.0).astype(np.float32)


def _gradient(grey: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _edge_blur(linear: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blur width at the edge samples of one frame, and where the samples are."""
    grey = _grey(linear)
    fine = _gradient(grey)
    reblurred = _gradient(cv2.GaussianBlur(grey, (0, 0), _S0))
    located = _gradient(cv2.GaussianBlur(grey, (0, 0), _LOCATE_SIGMA)) > _LOCATE_GRADIENT
    peaks = fine >= cv2.dilate(fine, np.ones((3, 3), np.uint8))
    samples = located & peaks & valid & (fine > 0)
    ratio = fine / np.maximum(reblurred, 1e-3)
    width = _S0 / np.sqrt(np.maximum(ratio * ratio - 1.0, 1e-4))
    return np.minimum(width, 8.0).astype(np.float32), samples.astype(np.float32)


def coverage_map(
    frames: Sequence[np.ndarray], *, overlay_size: tuple[int, int] | None = None
) -> tuple[float, np.ndarray]:
    """Where the sharpest frame is still soft.

    Args:
        frames: the aligned frames, linear RGB at the reference's geometry,
            NaN outside a frame's coverage.
        overlay_size: ``(width, height)`` of the overlay; the working size by
            default.

    Returns:
        The fraction of the picture no frame covers, and an 8-bit BGRA overlay:
        transparent where some frame is in focus, red where none is.
    """
    best: np.ndarray | None = None
    evidence: np.ndarray | None = None
    for frame in frames:
        small, _ = shrink(frame, WORK_EDGE)
        valid = np.isfinite(small[..., 0])
        width, samples = _edge_blur(small, valid)
        window = max(small.shape[:2]) / 60.0
        density = cv2.GaussianBlur(samples, (0, 0), window)
        regional = cv2.GaussianBlur(width * samples, (0, 0), window) / np.maximum(density, 1e-9)
        regional[density < _MIN_EVIDENCE] = np.inf
        best = regional if best is None else np.minimum(best, regional)
        evidence = density if evidence is None else np.maximum(evidence, density)
    assert best is not None and evidence is not None
    soft = (evidence >= _MIN_EVIDENCE) & (best > BLUR_LIMIT_PX) & np.isfinite(best)
    soft = soft.astype(np.uint8)
    # A blob smaller than a hundredth of the frame is an edge case of the
    # measurement, not a zone the stack missed.
    size = max(3, round(max(soft.shape) / 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    soft = cv2.morphologyEx(soft, cv2.MORPH_OPEN, kernel)
    fraction = float(soft.mean())
    overlay = np.zeros((*soft.shape, 4), dtype=np.uint8)
    overlay[soft.astype(bool)] = _OVERLAY
    if overlay_size is not None and overlay_size != (soft.shape[1], soft.shape[0]):
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_NEAREST)
    return round(fraction, 4), overlay
