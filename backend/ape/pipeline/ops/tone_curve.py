# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tone curve: a parametric four-region curve plus a free user spline.

Both halves are display-referred and both are built as lookup tables, so the
renderer can fold them into the rest of the tone chain and pay for a single pass
over the pixels.

The parametric half divides the range into shadows, darks, lights and
highlights and pushes each with a Gaussian-weighted offset. Amplitude is capped
below ``sigma * sqrt(e)`` so every individual region has a derivative under 1,
and the regions are composed rather than summed -- together those two facts make
the curve monotonic for any combination of sliders, including all four at their
extremes.

The free half is a monotonic PCHIP spline through the user's control points.
PCHIP rather than a natural cubic spline for exactly one reason: a natural
spline overshoots between widely spaced points and can fold back on itself,
which shows up as posterised bands in a smooth sky.
"""

from __future__ import annotations

import numpy as np

from ..lut import LUT_SIZE, apply_lut, compose, identity_lut
from ..params import ToneCurveParams

__all__ = ["apply", "build_lut"]

#: Centres of the four parametric regions, from shadows to highlights.
_REGION_CENTRES = (0.125, 0.375, 0.625, 0.875)
_REGION_SIGMA = 0.16
# Below sigma * sqrt(e) = 0.264, so each region's derivative stays under 1.
_REGION_AMPLITUDE = 0.18


def _parametric_lut(params: ToneCurveParams, size: int) -> np.ndarray:
    amounts = (params.shadows, params.darks, params.lights, params.highlights)
    if all(abs(a) < 1e-9 for a in amounts):
        return identity_lut(size)

    x = identity_lut(size).astype(np.float64)
    for amount, centre in zip(amounts, _REGION_CENTRES, strict=True):
        if abs(amount) < 1e-9:
            continue
        weight = np.exp(-0.5 * np.square((x - centre) / _REGION_SIGMA))
        x = np.clip(x + _REGION_AMPLITUDE * amount * weight, 0.0, 1.0)
    return x.astype(np.float32)


def _spline_lut(points: list[tuple[float, float]], size: int) -> np.ndarray:
    if not points:
        return identity_lut(size)

    from scipy.interpolate import PchipInterpolator

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    # Anchor the ends unless the user already pinned them, so a curve defined
    # only in the midtones leaves black and white where they were.
    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, 0.0)
    if xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(1.0)

    spline = PchipInterpolator(np.asarray(xs), np.asarray(ys), extrapolate=True)
    sampled = spline(identity_lut(size).astype(np.float64))
    # PCHIP is monotonic between the control points, but the user's points
    # themselves may not be, and the ends may extrapolate outside the range.
    return np.clip(np.maximum.accumulate(sampled), 0.0, 1.0).astype(np.float32)


def build_lut(params: ToneCurveParams, size: int = LUT_SIZE) -> np.ndarray:
    """Combined table: parametric regions first, then the user's spline."""
    return compose(_parametric_lut(params, size), _spline_lut(params.points, size))


def apply(img: np.ndarray, params: ToneCurveParams) -> np.ndarray:
    """Apply the tone curve per channel to a display-referred image in [0, 1]."""
    return apply_lut(img, build_lut(params))
