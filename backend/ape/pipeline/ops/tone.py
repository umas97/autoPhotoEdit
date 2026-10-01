# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The sigmoid tone mapping, and the shadows/highlights shaping that follows it.

This module owns the one transition in the pipeline from scene-referred to
display-referred. Everything before it is physics; everything after it is taste.

**The curve.** Rather than a gamma, a parametric sigmoid in the (EV, display
code) plane, built from two segments that meet at the pivot::

    segment(dx) = dy_max * t / (1 + t**power) ** (1/power),   t = slope*dx/dy_max

The segment has exactly ``slope`` at the pivot, rises monotonically, and
approaches ``dy_max`` asymptotically -- so black and white are approached but
never reached by a hard clip, which is what keeps the deep shadows and the
specular highlights from turning into flat patches. ``power`` is the toe (below
the pivot) or the shoulder (above it): low values bend early and gently, high
values stay straight and then turn sharply.

``contrast`` multiplies the slope, and 1.0 is calibrated to the slope a plain
linear-to-display transfer would have at middle grey -- so ``contrast = 1``
means "no S-curve at all", and the 1.2 default is the modest S a normal
photographic rendering carries.

``black_point_ev`` and ``white_point_ev`` then say which part of that curve is
mapped onto the display range, which is where the scene's dynamic range gets
decided.

**Hue stability.** A sigmoid applied independently to R, G and B desaturates
highlights beautifully and shifts their hue badly: whichever channel saturates
first drags the colour with it, and a sunset turns yellow. Following AgX, chroma
is compressed towards the achromatic axis *before* the curve and expanded again
after, with the expansion deliberately weaker than the compression so that some
of the filmic desaturation survives. ``chroma_preservation`` scales both.
"""

from __future__ import annotations

import numpy as np

from ..colorspace import DISPLAY_GAMMA, DISPLAY_GREY, MIDDLE_GREY, luminance
from ..lut import LUT_SIZE, apply_lut, apply_range_lut, identity_lut
from ..params import ToneParams, ToneShapingParams

__all__ = [
    "apply_sigmoid",
    "apply_shaping",
    "build_shaping_lut",
    "display_lightness",
    "sigmoid_response",
]

#: Slope at the pivot, in display code per stop, of a transfer that leaves the
#: scene alone. ``contrast = 1.0`` reproduces it exactly.
BASE_SLOPE = float(np.log(2.0) / DISPLAY_GAMMA * DISPLAY_GREY)

#: Range and resolution of the scene-side table. 40 stops is more than any
#: sensor delivers, and 4096 samples put the step at 0.01 EV.
_EV_MIN, _EV_MAX = -24.0, 16.0

#: Peak chroma compression before the curve, at ``chroma_preservation = 1``.
_INSET_MAX = 0.35
#: The expansion is weaker than the compression on purpose: the difference is
#: the highlight desaturation that makes the rendering look photographic.
_OUTSET_RATIO = 0.80

# Shaping constants. The Gaussian amplitude is kept below sigma * sqrt(e) so
# that each bump has a derivative smaller than 1 everywhere, which guarantees
# the composed shaping curve stays monotonic -- a non-monotonic tone curve
# inverts local contrast and looks broken.
_SHAPE_AMPLITUDE = 0.20
_SHAPE_SIGMA = 0.22
_SHADOW_CENTRE = 0.25
_HIGHLIGHT_CENTRE = 0.75
#: How far the blacks/whites sliders move the endpoints of the range.
_POINT_TRAVEL = 0.15


def _segment(dx: np.ndarray, dy_max: float, slope: float, power: float) -> np.ndarray:
    """Monotonic segment with the given slope at 0, asymptotic to ``dy_max``."""
    t = slope * dx / dy_max
    return dy_max * t / np.power(1.0 + np.power(t, power), 1.0 / power)


def _raw_sigmoid(ev: np.ndarray, params: ToneParams) -> np.ndarray:
    """The curve before the black/white points crop it, in display code."""
    shifted = ev - params.pivot
    slope = params.contrast * BASE_SLOPE
    distance = np.abs(shifted)
    below = DISPLAY_GREY - _segment(distance, DISPLAY_GREY, slope, params.toe)
    above = DISPLAY_GREY + _segment(distance, 1.0 - DISPLAY_GREY, slope, params.shoulder)
    return np.where(shifted < 0.0, below, above)


def sigmoid_response(ev: np.ndarray | float, params: ToneParams) -> np.ndarray:
    """Display code produced by ``ev`` stops relative to middle grey, in [0, 1]."""
    ev_array = np.atleast_1d(np.asarray(ev, dtype=np.float64))
    raw = _raw_sigmoid(ev_array, params)
    low = float(_raw_sigmoid(np.array([params.black_point_ev]), params)[0])
    high = float(_raw_sigmoid(np.array([params.white_point_ev]), params)[0])
    span = max(high - low, 1e-6)
    return np.clip((raw - low) / span, 0.0, 1.0)


def _scene_lut(params: ToneParams, size: int = LUT_SIZE) -> np.ndarray:
    """Sigmoid sampled uniformly over ``[_EV_MIN, _EV_MAX]`` stops."""
    ev = np.linspace(_EV_MIN, _EV_MAX, size, dtype=np.float64)
    return sigmoid_response(ev, params).astype(np.float32)


def display_lightness(luma: np.ndarray, params: ToneParams) -> np.ndarray:
    """Display code a grey of linear luminance ``luma`` gets from this sigmoid.

    What a luminance range of a parametric mask selects on (``ops/masks.py``):
    how bright a pixel will look, not how many photons it caught -- the same
    "highlights" before and after the exposure slider moves.
    """
    ev = np.maximum(luma, np.float32(1e-10)) / np.float32(MIDDLE_GREY)
    np.log2(ev, out=ev)
    return apply_range_lut(ev, _scene_lut(params), _EV_MIN, _EV_MAX)


def apply_sigmoid(img: np.ndarray, params: ToneParams) -> np.ndarray:
    """Scene-referred linear Rec.2020 -> display-referred, perceptually encoded.

    Args:
        img: ``(H, W, 3)`` float32, linear, scene-referred Rec.2020. Values above
            1.0 are expected and handled.
        params: the sigmoid's shape.

    Returns:
        ``(H, W, 3)`` float32 in [0, 1], Rec.2020 primaries, encoded with
        ``DISPLAY_GAMMA``. This is the input space of every operation that
        follows.
    """
    # In place from the first new array on -- the input is never written --
    # because each temporary of a proxy is 34 MB of fresh pages and this stage
    # is bound by memory traffic. Same arithmetic, same order, same bits.
    inset = _INSET_MAX * params.chroma_preservation
    if inset > 0.0:
        luma = luminance(img)[..., None]
        working = img * np.float32(1.0 - inset)
        working += luma * np.float32(inset)
        np.maximum(working, 1e-10, out=working)
    else:
        working = np.maximum(img, 1e-10)

    response = _scene_lut(params)
    # Anchoring on MIDDLE_GREY is what ties every EV-denominated parameter --
    # here, in exposure, in the decoder's baseline -- to the same reference.
    working /= np.float32(MIDDLE_GREY)
    ev = np.log2(working, out=working)
    encoded = apply_range_lut(ev, response, _EV_MIN, _EV_MAX)

    outset = inset * _OUTSET_RATIO
    if outset > 0.0:
        luma = luminance(encoded)[..., None]
        encoded -= luma * np.float32(outset)
        encoded /= np.float32(1.0 - outset)

    return np.clip(encoded, 0.0, 1.0, out=encoded)


def build_shaping_lut(params: ToneShapingParams, size: int = LUT_SIZE) -> np.ndarray:
    """Table for shadows / highlights / whites / blacks, display-referred.

    Order matters: the endpoints move first, so the shadow and highlight bumps
    act on the range the user can actually see.
    """
    x = identity_lut(size).astype(np.float64)

    black = -_POINT_TRAVEL * params.blacks
    white = 1.0 + _POINT_TRAVEL * params.whites
    if abs(black) > 1e-9 or abs(white - 1.0) > 1e-9:
        x = np.clip((x - black) / max(white - black, 1e-6), 0.0, 1.0)

    # Applied one after the other rather than summed: each bump is individually
    # monotonic, and a composition of monotonic functions stays monotonic, while
    # a sum of two overlapping bumps would not be.
    regions = (
        (params.shadows, _SHADOW_CENTRE),
        (params.highlights, _HIGHLIGHT_CENTRE),
    )
    for amount, centre in regions:
        if abs(amount) < 1e-9:
            continue
        weight = np.exp(-0.5 * np.square((x - centre) / _SHAPE_SIGMA))
        x = np.clip(x + _SHAPE_AMPLITUDE * amount * weight, 0.0, 1.0)

    return x.astype(np.float32)


def apply_shaping(img: np.ndarray, params: ToneShapingParams) -> np.ndarray:
    """Apply the shaping curve per channel to a display-referred image."""
    lut = build_shaping_lut(params)
    return apply_lut(img, lut)
