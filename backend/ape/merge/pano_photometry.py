# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A panorama's exposure gains and its lens's vignetting, from the overlaps.

Where two frames overlap they see the same point of the scene, from different
places in the frame: the ratio of the two values is the exposure difference of
the frames times the ratio of their vignetting at those two places -- and
nothing of the scene, whatever its own gradients (a blue sky has a strong
one). So both are solved together, in logarithms, where they are linear:

    log g_i + log I_i - L(r_i) = log g_j + log I_j - L(r_j),
    L(r) = a r^2 + b r^4 + c r^6,

with ``r`` the distance from the frame's centre over its half diagonal, one
``L`` for all frames (one lens, one aperture), and the reference's gain fixed
at one: its exposure is the one the user metered. It is the photometric
calibration panorama programs do, reduced to what a sweep of one lens needs.

Why it is here: without a lens profile nothing corrects the vignetting before
stitching, and every seam shows as a band in a smooth sky -- the user's
panorama 07319-07335 at 28 mm f/2.8, stitched without the A063's profile, had
steps of 5-6 per cent at its seams. With a profile, what is left is the
profile's own error, 2-5 per cent on the same seams. The correction is applied
as each frame is read (``pano_compose``); the frames themselves are never
changed.

The falloff is large on a fast zoom: the A063 at 28 mm f/2.8 measured here
at about two stops at the edge of the samples without its profile, and one
and a half *with* it -- lensfun's profile for it corrects some 0.6 stops. The
fit is trimmed -- parallax, something that moved, a misregistered edge are
outliers -- and checked against physics: a falloff that brightens towards the
corners, or darkens them past a tenth, is a fit of the scene rather than of
the lens, and the gains are solved alone, as before. Past the largest radius
the overlaps reached the polynomial is an extrapolation, so it is held at its
last value there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["NO_VIGNETTING", "Photometry", "falloff", "radius2", "solve"]

#: Pixels whose luminance counts, as a fraction of saturation: out of the noise
#: and below clipping, in both frames of a pair.
_RANGE = (0.01, 0.85)

#: Samples taken from one overlap. A few thousand pin three coefficients down;
#: a cap per pair keeps a large overlap from outvoting a small one.
_PER_PAIR = 6000

#: Fewest usable pixels for an overlap to count at all.
_MIN_PIXELS = 500

#: Trimming: residuals past this many robust sigmas are dropped, over this
#: many rounds.
_TRIM_SIGMA, _TRIM_ROUNDS = 2.5, 3

#: Pull of the vignetting coefficients towards zero, per sample: enough to keep
#: two frames with a thin overlap from inventing a falloff, far too little to
#: bias a real one.
_RIDGE = 0.02

#: Physics: a frame may lose at most this much at the edge of the samples
#: (a tenth left, 3.3 stops), and the falloff may rise by at most this much.
_MIN_CORNER, _MAX_RISE = 0.1, 0.02

#: Percentile of the sampled radii taken as the end of what the fit knows.
_REACH_PERCENTILE = 99


@dataclass(frozen=True)
class Photometry:
    gains: list[float]
    #: ``(a, b, c, reach)``: the coefficients of ``L(r)`` and the largest
    #: squared radius the fit saw; zeros when no falloff was fitted.
    vignetting: tuple[float, float, float, float]

    @property
    def corner_ev(self) -> float:
        """How much darker the frame was at the edge of the fit, in stops (negative)."""
        return round(math.log2(float(falloff(np.float32(1.0), self.vignetting))), 2)


def radius2(width: int, height: int) -> np.ndarray:
    """Squared distance from the centre over the squared half diagonal."""
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    return (((xs - cx) ** 2 + (ys - cy) ** 2) / np.float32(cx * cx + cy * cy)).astype(np.float32)


NO_VIGNETTING = (0.0, 0.0, 0.0, 0.0)


def falloff(r2: np.ndarray, vignetting: Sequence[float]) -> np.ndarray:
    """The vignetting ``V = exp(L(r))`` at squared radii ``r2``, held past the reach."""
    a, b, c, reach = (np.float32(v) for v in vignetting)
    r2 = np.minimum(r2, reach)
    return np.exp(r2 * (a + r2 * (b + r2 * c))).astype(np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2627 + rgb[..., 1] * 0.678 + rgb[..., 2] * 0.0593


def _overlap(a, b):
    """Slices of two warped rectangles over their intersection, or None."""
    (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    return (
        (slice(y0 - ay, y1 - ay), slice(x0 - ax, x1 - ax)),
        (slice(y0 - by, y1 - by), slice(x0 - bx, x1 - bx)),
    )


def _samples(warped, masks, radii, corners):
    """Per sample: frame i, frame j, r2 in each, log I_j - log I_i."""
    rects = [(x, y, m.shape[1], m.shape[0]) for (x, y), m in zip(corners, masks, strict=True)]
    rows = []
    for i in range(len(warped)):
        for j in range(i + 1, len(warped)):
            both = _overlap(rects[i], rects[j])
            if both is None:
                continue
            si, sj = both
            yi, yj = _luminance(warped[i][si]), _luminance(warped[j][sj])
            good = (masks[i][si] > 0) & (masks[j][sj] > 0)
            for y in (yi, yj):
                good &= (y > _RANGE[0]) & (y < _RANGE[1])
            where = np.flatnonzero(good)
            if where.size < _MIN_PIXELS:
                continue
            # Evenly spaced, not random: the same recipe, the same samples.
            where = where[np.linspace(0, where.size - 1, min(where.size, _PER_PAIR)).astype(int)]
            ri, rj = radii[i][si].ravel()[where], radii[j][sj].ravel()[where]
            target = np.log(yj.ravel()[where]) - np.log(yi.ravel()[where])
            rows.append((i, j, ri, rj, target))
    return rows


def _fit(rows, count: int, reference: int, with_vignetting: bool):
    unknowns = count + (3 if with_vignetting else 0)
    blocks, targets = [], []
    for i, j, ri, rj, target in rows:
        block = np.zeros((target.size, unknowns))
        block[:, i], block[:, j] = 1.0, -1.0
        if with_vignetting:
            for k in range(3):
                block[:, count + k] = -(ri ** (k + 1) - rj ** (k + 1))
        blocks.append(block)
        targets.append(target)
    a = np.vstack(blocks)
    b = np.concatenate(targets).astype(np.float64)
    keep = np.ones(b.size, bool)
    solution = np.zeros(unknowns)
    for _ in range(_TRIM_ROUNDS):
        extra_a = [np.eye(1, unknowns, reference) * 1e3]
        extra_b = [0.0]
        if with_vignetting:
            ridge = _RIDGE * math.sqrt(max(1, int(keep.sum())))
            for k in range(3):
                extra_a.append(np.eye(1, unknowns, count + k) * ridge)
                extra_b.append(0.0)
        solution, *_ = np.linalg.lstsq(
            np.vstack([a[keep], *extra_a]), np.concatenate([b[keep], extra_b]), rcond=None
        )
        residual = a @ solution - b
        sigma = 1.4826 * float(np.median(np.abs(residual[keep]))) + 1e-6
        keep = np.abs(residual) < _TRIM_SIGMA * sigma
    return solution


def _plausible(vignetting) -> bool:
    curve = falloff(np.linspace(0.0, vignetting[3], 21, dtype=np.float32), vignetting)
    return float(curve[-1]) >= _MIN_CORNER and float(np.diff(curve).max()) <= _MAX_RISE


def solve(warped, masks, radii, corners, reference: int) -> Photometry:
    """Gains (reference at one) and the shared vignetting, from the overlaps.

    ``warped`` are linear frames on the panorama's surface, ``radii`` the
    squared radius of each warped pixel in its own frame (:func:`radius2`
    warped alike).
    """
    count = len(warped)
    rows = _samples(warped, masks, radii, corners)
    if not rows:
        return Photometry(gains=[1.0] * count, vignetting=NO_VIGNETTING)
    solution = _fit(rows, count, reference, with_vignetting=True)
    sampled = np.concatenate([np.concatenate([r[2], r[3]]) for r in rows])
    reach = float(np.percentile(sampled, _REACH_PERCENTILE))
    vignetting = (*(float(v) for v in solution[count:]), reach)
    if not _plausible(vignetting):
        solution = _fit(rows, count, reference, with_vignetting=False)
        vignetting = NO_VIGNETTING
    return Photometry(gains=[float(math.exp(g)) for g in solution[:count]],
                      vignetting=vignetting)  # type: ignore[arg-type]
