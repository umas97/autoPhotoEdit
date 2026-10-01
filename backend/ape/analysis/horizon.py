# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Horizons: one straight line across the frame that divides two worlds.

The second cue of the straightening (section 6.4), for the frames where nothing
stands upright -- a sea, a lake, a plain. It is the riskier cue: the ground is
full of straight lines that are not level (a path, a ridge of snow, a shore
receding in perspective), so a line counts as a horizon only when it is long,
nearly continuous, straight, and separates regions that look different. Each
test in :func:`find_horizon` was added for a line in the user's photos that
passed the ones before it and was not a horizon.
"""

from __future__ import annotations

import math

import numpy as np

from .lines import FAMILY_WINDOW_DEG, FamilyFit

__all__ = ["find_horizon"]

#: A horizon must span this fraction of the width and add up to this much
#: length (the rest being trees, boats, people standing in front of it).
_HORIZON_SPAN = 0.5
_HORIZON_LENGTH = 0.35

#: How much of its own span the pieces of a horizon must cover. A horizon is
#: nearly continuous -- 0.92 and more for the far shore of 05634 at every
#: rotation tried -- while the ice edge of 05635, a ground line with gaps where
#: it bends, covers 0.47 to 0.69.
_HORIZON_COVERAGE = 0.85

#: RMS distance of the pieces' end points from the fitted line, as a fraction
#: of the width.
_HORIZON_STRAIGHTNESS = 0.002

#: How different the two sides of a horizon must look: 8% of full scale in
#: the most different channel. Measured: the far edge of the frozen lake in
#: 05634 separates dark shore from ice by far more; the snow ridge on the ice
#: of 05636 by less than 4%.
_HORIZON_CONTRAST = 0.08


def _collinear_groups(
    segments: np.ndarray, length: np.ndarray, angle_deg: float, long_edge: float
) -> list[np.ndarray]:
    """Indices of segments sharing one line at ``angle_deg``, within 1% of the frame."""
    theta = math.radians(angle_deg)
    mid_x = (segments[:, 0] + segments[:, 2]) / 2
    mid_y = (segments[:, 1] + segments[:, 3]) / 2
    # Offset perpendicular to a line at that angle (image y points down).
    rho = mid_x * math.sin(theta) + mid_y * math.cos(theta)
    order = np.argsort(rho)
    tolerance = 0.01 * long_edge
    groups, start = [], 0
    for stop in range(1, len(order) + 1):
        if stop == len(order) or rho[order[stop]] - rho[order[stop - 1]] > tolerance:
            groups.append(order[start:stop])
            start = stop
    return groups


def find_horizon(
    segments: np.ndarray,
    deviation: np.ndarray,
    length: np.ndarray,
    width: int,
    long_edge: float,
    image: np.ndarray | None,
) -> FamilyFit | None:
    """One straight line across the frame that divides two different worlds.

    Looks, at every candidate angle of the family's window, for segments that
    lie on the *same* line -- that angle to within a degree and the same offset
    from the centre to within 1% of the frame. A real horizon broken by trees
    is many collinear pieces; the parallel shores of a lake receding in
    perspective are several lines, none of which spans the frame.

    Candidates are tried longest first, and the first that passes every test
    wins: it spans half the width, it is straight -- a sagging cable or barrier
    tape (05635) spans the frame too, and bends -- and it separates two regions
    that look different (``_separates``).
    """
    if len(segments) == 0 or image is None:
        return None
    candidates: dict[tuple[int, ...], float] = {}
    for candidate in np.arange(-FAMILY_WINDOW_DEG, FAMILY_WINDOW_DEG + 1e-9, 0.25):
        near = np.nonzero(np.abs(deviation - candidate) < 1.0)[0]
        if len(near) == 0:
            continue
        for group in _collinear_groups(segments[near], length[near], candidate, long_edge):
            members = tuple(sorted(int(i) for i in near[group]))
            candidates[members] = float(length[list(members)].sum())
    for members_tuple, total in sorted(candidates.items(), key=lambda item: -item[1]):
        if total < _HORIZON_LENGTH * width:
            break  # sorted: nothing after this is long enough either
        fit = _horizon_fit(segments, length, np.array(members_tuple), total, width, long_edge)
        if fit is None or fit.line is None:
            continue
        fit.contrast = _separates(image, fit.line)
        if fit.contrast >= _HORIZON_CONTRAST:
            return fit
    return None


def _horizon_fit(
    segments: np.ndarray,
    length: np.ndarray,
    members: np.ndarray,
    total: float,
    width: int,
    long_edge: float,
) -> FamilyFit | None:
    """The line through one group of collinear pieces, if it is long and straight."""
    ends_x = np.concatenate([segments[members, 0], segments[members, 2]])
    ends_y = np.concatenate([segments[members, 1], segments[members, 3]])
    span = float(np.ptp(ends_x)) / width
    if span < _HORIZON_SPAN or len(ends_x) < 4:
        return None
    weights = np.concatenate([length[members], length[members]])
    (slope, intercept), cov = np.polyfit(ends_x, ends_y, 1, w=np.sqrt(weights), cov=True)
    straightness = float(
        np.sqrt(np.average((ends_y - (slope * ends_x + intercept)) ** 2, weights=weights))
    )
    if straightness > _HORIZON_STRAIGHTNESS * width:
        return None
    # The fit's own error, but never below what the end points' localisation
    # allows: LSD places an end to about a pixel, over the span of the line.
    sigma = max(
        math.degrees(math.sqrt(max(0.0, float(cov[0, 0])))),
        math.degrees(math.atan(1.0 / (span * width))),
    )
    fit = FamilyFit(
        roll_deg=-math.degrees(math.atan(slope)),
        coherence=min(1.0, total / (span * width)),
        support=total / long_edge,
        segments=int(len(members)),
        roll_sigma=sigma,
    )
    if fit.coherence < _HORIZON_COVERAGE:
        return None
    fit.line = (float(slope), float(intercept), float(ends_x.min()), float(ends_x.max()))
    return fit


def _separates(image: np.ndarray, line: tuple[float, float, float, float]) -> float:
    """How different the two sides of a line look, 0 (same) .. 1.

    A horizon divides two worlds -- sky and sea, far shore and water, trees and
    ice -- while a straight line *on* the ground, like the snow ridge across the
    frozen lake of 05636, has the same stuff on both sides. Compared on two
    strips 3% of the frame high, just above and just below, over the line's
    own span: the mean of each channel, in the 0..255 of the proxy.
    """
    slope, intercept, x0, x1 = line
    height, width = image.shape[:2]
    band = max(3, round(0.03 * height))
    xs = np.arange(max(0, int(x0)), min(width, int(x1) + 1), 2)
    if len(xs) < 10:
        return 0.0
    ys = slope * xs + intercept
    channels = image.shape[2] if image.ndim == 3 else 1
    above, below = [], []
    for offset in range(2, band + 2):
        ya, yb = np.round(ys - offset).astype(int), np.round(ys + offset).astype(int)
        ok_a = (ya >= 0) & (ya < height)
        ok_b = (yb >= 0) & (yb < height)
        above.append(image[ya[ok_a], xs[ok_a]].reshape(-1, channels))
        below.append(image[yb[ok_b], xs[ok_b]].reshape(-1, channels))
    a, b = np.concatenate(above).astype(np.float64), np.concatenate(below).astype(np.float64)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(np.abs(a.mean(axis=0) - b.mean(axis=0)).max() / 255.0)


