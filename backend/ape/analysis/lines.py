# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Line families: which way the straight things in a frame lean, and how surely.

The measuring half of the straightening of section 6.4 (``straighten.py`` holds
the decision). A *family* is the set of segments near one axis -- horizontal or
vertical -- and fitting it answers one question: what roll of the camera, and
what convergence towards a vanishing point, explains their deviations from the
axis. Perspective is modelled because a camera pointed up makes verticals lean
without being rolled at all; only the roll is a tilt to correct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "AGREEMENT_DEG",
    "FAMILY_WINDOW_DEG",
    "FamilyFit",
    "fit_rolled",
    "segments",
]

#: How far from an axis a segment may be and still count as "meant to be" on
#: it. The detector is asked to find tilts up to 8 degrees; two more give the
#: fit room to see the whole distribution around such a tilt.
FAMILY_WINDOW_DEG = 10.0

#: A segment agrees with the fit when its residual is below this.
AGREEMENT_DEG = 0.5

#: A family is confident when this fraction of its length agrees with the fit
#: *and* the agreeing length adds up to at least ``_MIN_SUPPORT`` of the long
#: edge. Measured on the 24 fixtures (mostly forest, snow, logs, portraits):
#: natural scenes stay below 0.45, while the lake shore and the building of
#: 05634, 05635 and 05641 reach 0.6 and more.
_MIN_COHERENCE = 0.55

_MIN_SUPPORT = 0.6

#: Convergences tried, in 1 / (distance to the vanishing point in long edges).
#: 1.2 is a vanishing point less than a frame away from the centre: a wide
#: lens pointed steeply up a facade. Beyond that the frame is not "tilted", it
#: is a picture of converging lines, and no roll estimate is meaningful.
_CONVERGENCE_GRID = tuple(np.round(np.linspace(-1.2, 1.2, 49), 3))

#: Largest standard error of the roll a family may have and still decide.
_MAX_ROLL_SIGMA = 0.3


@dataclass(slots=True)
class FamilyFit:
    roll_deg: float
    coherence: float
    #: Agreeing length, as a multiple of the long edge.
    support: float
    segments: int
    #: 1 / distance to the vanishing point, in long edges. 0 is parallel.
    convergence: float = 0.0
    #: Standard error of ``roll_deg``, degrees.
    roll_sigma: float = 0.0
    #: For a horizon: ``(slope, intercept, x0, x1)`` in work-image pixels.
    line: tuple[float, float, float, float] | None = None
    #: For a horizon: how different its two sides look (see ``_separates``).
    contrast: float = 0.0

    @property
    def confident(self) -> bool:
        return (
            self.coherence >= _MIN_COHERENCE
            and self.support >= _MIN_SUPPORT
            and self.roll_sigma <= _MAX_ROLL_SIGMA
        )


def segments(grey: np.ndarray) -> np.ndarray:
    """``(N, 4)`` float segments ``x1, y1, x2, y2`` found by LSD."""
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = detector.detect(grey)[0]
    if lines is None:
        return np.empty((0, 4), dtype=np.float64)
    return lines.reshape(-1, 4).astype(np.float64)


def _robust_roll(
    residual_base: np.ndarray, length: np.ndarray, start: float
) -> tuple[float, np.ndarray]:
    """Tukey-weighted location of ``residual_base`` and the final weights.

    A 1.5 degree scale: segments further than that from the current estimate
    stop pulling on it at all, which is what lets a building with a few stray
    diagonals still find its verticals.
    """
    roll = start
    weight = length
    for _ in range(5):
        u = np.clip(np.abs(residual_base - roll) / 1.5, 0.0, 1.0)
        weight = length * (1 - u**2) ** 2
        if weight.sum() <= 0:
            break
        roll = float(np.average(residual_base, weights=weight))
    return roll, weight


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def _predict(across: np.ndarray, along: np.ndarray, roll: float, c: float) -> np.ndarray:
    return roll + np.degrees(np.arctan(across * c / np.maximum(1e-6, 1 - along * c)))


def _refine(
    deviation: np.ndarray,
    across: np.ndarray,
    along: np.ndarray,
    length: np.ndarray,
    roll: float,
    c: float,
) -> tuple[float, float]:
    """Least squares on ``(roll, c)`` from the grid's best point.

    The grid maximises how much length agrees within half a degree, which is
    robust but flat: with the segments in two columns -- a pillar on the left,
    a wall on the right -- several pairs of roll and convergence make the same
    segments agree, and the grid picks one at random. The least-squares
    minimum inside that plateau is unique. A soft-L1 loss at the agreement
    scale keeps the stray diagonals the grid already ignored from pulling.
    """
    from scipy.optimize import least_squares

    weight = np.sqrt(length / length.sum())

    def residuals(x: np.ndarray) -> np.ndarray:
        return weight * (deviation - _predict(across, along, float(x[0]), float(x[1])))

    bound = float(max(abs(v) for v in _CONVERGENCE_GRID))
    result = least_squares(
        residuals,
        x0=np.array([roll, c]),
        bounds=([-45.0, -bound], [45.0, bound]),
        loss="soft_l1",
        f_scale=AGREEMENT_DEG * float(weight.mean()),
    )
    return float(result.x[0]), float(result.x[1])


def _roll_sigma(
    residual: np.ndarray,
    across: np.ndarray,
    along: np.ndarray,
    length: np.ndarray,
    c: float,
    free: bool,
) -> float:
    """Standard error of the roll, from the weighted least-squares covariance.

    The weights are the segment lengths times Tukey's biweight of the final
    residual, so outliers neither count as evidence nor inflate the scatter.
    When the convergence is free, its correlation with the roll is what the
    covariance carries: segments in only two columns leave the pair badly
    determined, and the roll's error grows to say so.
    """
    u = np.clip(np.abs(residual) / 1.5, 0.0, 1.0)
    weight = length * (1 - u**2) ** 2
    used = weight > 0
    parameters = 2 if free else 1
    if used.sum() <= parameters:
        return float("inf")
    w, r = weight[used], residual[used]
    variance = float((w * r**2).sum() / w.sum())
    # Pieces of one pillar are not independent measurements: count segments
    # by their share of the length, not one each.
    effective = float(w.sum() ** 2 / (w**2).sum())
    if effective <= parameters:
        return float("inf")
    variance *= effective / (effective - parameters)
    if not free:
        return math.sqrt(variance / effective)
    denominator = np.maximum(1e-6, 1 - along[used] * c)
    d_c = np.degrees(across[used] / denominator**2 / (1 + (across[used] * c / denominator) ** 2))
    jac = np.stack([np.ones_like(d_c), d_c], axis=1)
    normal = (jac * (w / w.mean())[:, None]).T @ jac
    try:
        cov = np.linalg.inv(normal) * variance * len(w) / effective
    except np.linalg.LinAlgError:
        return float("inf")
    return math.sqrt(max(0.0, float(cov[0, 0])))


def _fit_family(
    deviation: np.ndarray,
    across: np.ndarray,
    along: np.ndarray,
    length: np.ndarray,
    long_edge: float,
) -> FamilyFit | None:
    """Fit ``deviation ~ roll + atan(across * c / (1 - along * c))``.

    That is the deviation of a line through a segment at ``(across, along)``
    -- coordinates from the centre, in long edges, ``along`` the family's own
    axis -- that runs to a vanishing point at distance ``1 / c`` on the rolled
    axis. ``c = 0`` is parallel lines. The convergence is searched on a grid
    (the fit is not convex in it) and the roll fitted robustly at each step;
    the pair that makes the most length agree wins.
    """
    if len(deviation) < 2:
        return None
    best: tuple[float, float, float] | None = None  # (agreeing, roll, c)
    spread = np.ptp(across) if len(across) else 0.0
    # With every segment in one place, convergence and roll cannot be told
    # apart: the lines are taken as parallel.
    grid = _CONVERGENCE_GRID if spread >= 0.25 else (0.0,)
    for c in grid:
        offset = np.degrees(np.arctan(across * c / np.maximum(1e-6, 1 - along * c)))
        base = deviation - offset
        roll, _ = _robust_roll(base, length, _weighted_median(base, length))
        agreeing = float(length[np.abs(base - roll) < AGREEMENT_DEG].sum())
        # Parallel is the simpler explanation: a convergence must win clearly.
        if c == 0.0:
            agreeing *= 1.05
        if best is None or agreeing > best[0]:
            best = (agreeing, roll, c)
    assert best is not None
    _, roll, c = best
    free = len(grid) > 1
    if free:
        roll, c = _refine(deviation, across, along, length, roll, c)
    residual = deviation - _predict(across, along, roll, c)
    agreeing = float(length[np.abs(residual) < AGREEMENT_DEG].sum())
    return FamilyFit(
        roll_deg=roll,
        coherence=float(agreeing / length.sum()),
        support=float(agreeing / long_edge),
        segments=int(len(deviation)),
        convergence=float(c),
        roll_sigma=_roll_sigma(residual, across, along, length, c, free),
    )


def _rotate(x: np.ndarray, y: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return c * x - s * y, s * x + c * y


def fit_rolled(
    dev: np.ndarray, across: np.ndarray, along: np.ndarray, length: np.ndarray, long_edge: float
) -> FamilyFit | None:
    """:func:`_fit_family`, then again in the frame the first roll levels.

    The vanishing-point model is written in the camera's upright frame; with the
    frame rolled by a few degrees, positions measured in image axes are off by
    that much and the perspective term misreads it. One refit in the de-rotated
    frame removes the bias -- measured on 05641 turned by 5 degrees: from 1.4
    degrees of error to below 0.2.
    """
    fit = _fit_family(dev, across, along, length, long_edge)
    if fit is None or abs(fit.roll_deg) < 0.05:
        return fit
    a, b = _rotate(across, along, -fit.roll_deg)
    refit = _fit_family(dev, a, b, length, long_edge)
    return refit if refit is not None else fit


