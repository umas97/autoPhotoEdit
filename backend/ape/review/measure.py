# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the confidence of section 9.1 needs from a photo's pixels, measured once.

Two measurements, both on the neutral browsing proxy and both taken by the
``predict`` job next to the exposure anchor (``style/auto.py``), so the
confidence itself never opens an image and can be recomputed for a whole
project on every slider of the review screen:

**How mixed the light is** (:func:`wb_spread_mired`). The frame is cut into a
4 x 4 grid and the grey world of ``auto.py`` -- the least chromatic quarter of
the usable pixels -- is taken in every cell. Under one illuminant the cells
agree, whatever their content; a tungsten-lit stage in a daylight hall, or a
window in a lamp-lit room, puts cells tens of mired apart. What is reported is
the spread along the blue-amber axis only, the one a temperature slider moves:
the green-magenta axis is where foliage and skin sit, and counting it would
call every garden "mixed light".

**The tonal tails** (:func:`tails`). The scene value of each pixel's brightest
channel, read back through the inverse of the neutral sigmoid, summarised as
the EV at a dozen quantiles at each end. From that table and a
set of parameters, ``review/tonal.py`` tells what fraction of the frame the edit
would push to white or to black -- the "clipping severo" term -- without a
render.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..pipeline.ops.white_balance import adaptation_matrix
from ..pipeline.params import ToneParams
from ..style.auto import _SRGB_TO_REC2020, _Y_REC2020, _inverse_sigmoid, _linear

__all__ = ["GRID", "TAIL_FRACTIONS", "measure", "neutral_tails", "tails", "wb_spread_mired"]

#: Cells per side of the white balance grid. 4 x 4 cells of a 512 px frame are
#: 128 px wide: large enough to hold a few hundred usable pixels, small enough
#: that half a frame lit differently fills whole cells.
GRID = 4

#: A cell needs this many usable pixels to vote. Below it the grey world of a
#: cell is the colour of whatever object filled it.
_MIN_PIXELS = 400

#: Fractions of the frame at which the tails are sampled. Denser where the
#: thresholds of section 9.1 (2% white, 5% black) sit.
TAIL_FRACTIONS = (0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35)

_WORK_EDGE = 512

#: Display code of half the first step of an 8-bit sRGB proxy. A pixel below
#: it is at 0 in the JPEG: somewhere under the first level, and the neutral
#: curve is flat down there, so its inverse would put it at the bottom of the
#: table (-14 EV) and a night sky would stay black under any brightening. It
#: is read at the neutral black point instead, which is by construction where
#: the curve reaches 0. Measured on the user's frames, both ways: night sky
#: DSC06318 (93% of the proxy at 0) after +3 EV, render 0.7% crushed, estimate
#: 0% here, 35% from the bottom of the table; dark stage DSC06356 (over 35% at
#: 0) after +2 EV, render 57%, estimate over 35% here, 0% if read at the first
#: level instead.
_FIRST_LEVEL_CODE = (0.5 / 255.0 / 12.92) ** (1.0 / 2.4)


def _display(neutral_srgb: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` display-linear Rec.2020 at :data:`_WORK_EDGE`."""
    height, width = neutral_srgb.shape[:2]
    scale = _WORK_EDGE / max(height, width)
    image = neutral_srgb
    if scale < 1:
        image = cv2.resize(
            image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    return _linear(image) @ _SRGB_TO_REC2020.T


@lru_cache(maxsize=64)
def _mired_per_log_ratio(as_shot_k: float, as_shot_tint: float) -> float:
    """Mired of white balance correction per unit of ``log(B/R)`` of a grey.

    Measured on the adaptation the pipeline itself applies, so "40 mired
    apart" means what the temperature slider would have to move by.
    """
    as_mired = 1e6 / as_shot_k

    def log_ratio(shift: float) -> float:
        target = round(1e6 / (as_mired - shift), 1)
        rgb = adaptation_matrix(target, as_shot_tint, as_shot_k, as_shot_tint) @ np.ones(3)
        return float(np.log(rgb[2] / rgb[0]))

    slope = (log_ratio(10.0) - log_ratio(-10.0)) / 20.0
    return 1.0 / slope if abs(slope) > 1e-9 else 0.0


def _cell_ratio(cell: np.ndarray) -> tuple[float, int] | None:
    """``log(B/R)`` of the cell's greyest quarter, and how many pixels voted."""
    pixels = cell.reshape(-1, 3)
    luma = pixels @ _Y_REC2020
    usable = (luma > 0.02) & (pixels.max(axis=1) < 0.97) & (pixels.min(axis=1) > 1e-4)
    count = int(usable.sum())
    if count < _MIN_PIXELS:
        return None
    rgb = pixels[usable]
    chroma = rgb / rgb.sum(axis=1, keepdims=True)
    distance = np.linalg.norm(chroma - 1.0 / 3.0, axis=1)
    mean = rgb[distance <= np.percentile(distance, 25.0)].mean(axis=0)
    return float(np.log(mean[2] / mean[0])), count


def wb_spread_mired(
    display: np.ndarray, *, as_shot_temperature_k: float, as_shot_tint: float
) -> float | None:
    """How far apart, in mired, the cells of the frame want the white balance.

    The distance between the 20th and 80th percentile of the cells' estimates,
    each cell weighted by its usable pixels: two or three odd cells (a red
    wall, a patch of sky) move neither percentile, half a frame under another
    light moves one of them all the way. ``None`` when fewer than four cells
    can vote -- a night frame, a frame of one flat colour.
    """
    height, width = display.shape[:2]
    rows = np.linspace(0, height, GRID + 1).astype(int)
    columns = np.linspace(0, width, GRID + 1).astype(int)
    ratios, weights = [], []
    for r in range(GRID):
        for c in range(GRID):
            found = _cell_ratio(display[rows[r] : rows[r + 1], columns[c] : columns[c + 1]])
            if found is not None:
                ratios.append(found[0])
                weights.append(found[1])
    if len(ratios) < 4:
        return None
    order = np.argsort(ratios)
    values = np.asarray(ratios)[order]
    cumulative = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    cumulative /= cumulative[-1]
    low = float(np.interp(0.2, cumulative, values))
    high = float(np.interp(0.8, cumulative, values))
    return abs(high - low) * abs(_mired_per_log_ratio(as_shot_temperature_k, as_shot_tint))


def tails(display: np.ndarray) -> dict[str, list[float]]:
    """Scene EV of each pixel's brightest channel, at :data:`TAIL_FRACTIONS`.

    ``high[i]`` is the EV exceeded by a fraction ``TAIL_FRACTIONS[i]`` of the
    frame, ``low[i]`` the EV a fraction falls below. The brightest channel for
    both ends: one channel at the top is a burnt colour, but a pixel is black
    only when all three are -- with the darkest channel instead, every
    saturated sky and red jacket counted as crushed (69 of the user's 128
    frames "over 5% black" at their automatic exposure, against 15 counting
    pixels whose three channels are all black). Pixels at the top of the
    neutral curve read as its top: the proxy cannot say how far beyond it they
    were.
    """
    neutral = ToneParams()
    xs, ys = _inverse_sigmoid(neutral.model_dump_json())
    pixels = display.reshape(-1, 3)
    peak = np.power(np.clip(pixels.max(axis=1), 0.0, 1.0), 1.0 / 2.4)
    ev_peak = np.interp(peak, xs, ys)
    ev_peak[peak < _FIRST_LEVEL_CODE] = neutral.black_point_ev
    fractions = np.asarray(TAIL_FRACTIONS)
    return {
        "high": [round(float(v), 3) for v in np.quantile(ev_peak, 1.0 - fractions)],
        "low": [round(float(v), 3) for v in np.quantile(ev_peak, fractions)],
    }


def neutral_tails(neutral_srgb: np.ndarray) -> dict[str, list[float]]:
    """:func:`tails` of a neutral sRGB rendering, ``uint8`` or float in [0, 1].

    A float rendering is quantised to 8 bits first, as the proxy the photos
    are measured on is: the first level is what ``_FIRST_LEVEL_CODE`` reads.
    """
    image = neutral_srgb
    if image.dtype != np.uint8:
        image = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return tails(_display(image))


def measure(neutral_srgb: np.ndarray, *, as_shot_temperature_k: float, as_shot_tint: float) -> dict:
    """Both measurements, as stored in ``Photo.analysis["auto"]``."""
    display = _display(neutral_srgb)
    spread = wb_spread_mired(
        display, as_shot_temperature_k=as_shot_temperature_k, as_shot_tint=as_shot_tint
    )
    return {
        "wb_spread_mired": None if spread is None else round(spread, 2),
        "tails": tails(display),
    }
