# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Colour: saturation, vibrance, the eight HSL bands and split toning.

All of it is display-referred, operating on the perceptually encoded [0, 1]
image the tone mapping produced. That is deliberate: saturation applied to
scene-linear values behaves nothing like what the slider promises, because a
fixed multiplier on the distance from grey means something completely different
at 0.02 and at 0.8.

The four controls are separate because they answer different questions:

* **saturation** scales every colour by the same factor -- honest and blunt;
* **vibrance** scales muted colours more than saturated ones and holds back the
  hues skin lives in, which is what makes it safe to push a landscape without
  turning the people in it orange;
* **the HSL bands** are surgical, eight overlapping hue windows with their own
  hue, saturation and luminance offsets;
* **split toning** tints shadows and highlights in opposite directions, the one
  colour operation that is about mood rather than correction.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..colorspace import channel_max, channel_min, luminance
from ..lut import apply_range_lut
from ..params import HSL_BAND_HUES, HSL_BANDS, ColorParams, SplitToningParams

__all__ = ["apply"]

#: Hue window half-width, in degrees. The bands are 30-60 degrees apart, so 45
#: keeps neighbours overlapping smoothly instead of leaving seams between them.
_BAND_HALF_WIDTH = 45.0

#: Full-slider hue rotation, in degrees. Larger and a band starts crossing into
#: its neighbour's territory, which is not what a hue *adjustment* should do.
_HUE_TRAVEL = 30.0

#: Hue window that vibrance protects: oranges and reds, where skin lives.
_SKIN_CENTRE = 25.0
_SKIN_HALF_WIDTH = 35.0
_SKIN_PROTECTION = 0.65

#: Samples of every hue-indexed table. The adjustments are smooth raised cosines
#: spanning tens of degrees, so a quarter-degree step is far finer than anything
#: the curves themselves resolve. The last entry repeats the first so that a hue
#: just under 360 interpolates back round to 0 instead of clamping.
_HUE_LUT_SIZE = 1441


def _raised_cosine(delta: np.ndarray, half_width: float) -> np.ndarray:
    """Smooth 1-at-the-centre, 0-at-the-edge window over an angular distance."""
    return np.where(
        delta < half_width, 0.5 * (1.0 + np.cos(np.pi * delta / half_width)), 0.0
    )


def _band_weights(hue_deg: np.ndarray) -> np.ndarray:
    """Raised-cosine weight of each of the eight bands, shaped ``(..., 8)``.

    Normalised so the weights sum to 1 wherever any band responds, which is what
    keeps a colour sitting between two bands from receiving both adjustments at
    full strength.

    Only ever called on the hue axis of a lookup table, never on an image: at
    full resolution the ``(..., 8)`` intermediate would be eight times the size
    of the frame, and it was by far the most expensive thing in the pipeline.
    """
    centres = np.asarray(HSL_BAND_HUES, dtype=np.float32)
    delta = np.abs(hue_deg[..., None] - centres)
    delta = np.minimum(delta, 360.0 - delta)
    weights = _raised_cosine(delta, _BAND_HALF_WIDTH).astype(np.float32)
    total = weights.sum(axis=-1, keepdims=True)
    return np.divide(weights, total, out=np.zeros_like(weights), where=total > 1e-6)


def _hue_axis() -> np.ndarray:
    return np.linspace(0.0, 360.0, _HUE_LUT_SIZE, dtype=np.float32)


@lru_cache(maxsize=1)
def _skin_weight_lut() -> np.ndarray:
    """How much vibrance is held back at each hue, to protect skin tones."""
    delta = np.abs(_hue_axis() - _SKIN_CENTRE)
    delta = np.minimum(delta, 360.0 - delta)
    protection = _SKIN_PROTECTION * _raised_cosine(delta, _SKIN_HALF_WIDTH)
    return (1.0 - protection).astype(np.float32)


def _hsl_luts(params: ColorParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue shift, saturation gain and luminance gain as functions of hue.

    The eight bands are collapsed into three tables once, on 1441 samples, and
    the image then pays three single-channel lookups instead of eight weighted
    passes over all its pixels.
    """
    weights = _band_weights(_hue_axis())
    hue = np.zeros(len(HSL_BANDS), dtype=np.float32)
    saturation = np.zeros(len(HSL_BANDS), dtype=np.float32)
    lightness = np.zeros(len(HSL_BANDS), dtype=np.float32)
    for index, name in enumerate(HSL_BANDS):
        band = params.band(name)
        hue[index] = band.hue * _HUE_TRAVEL
        saturation[index] = band.saturation
        lightness[index] = band.luminance
    return weights @ hue, weights @ saturation, weights @ lightness


def _apply_saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """Uniform scaling of the distance from the achromatic axis."""
    luma = luminance(img)[..., None]
    return np.clip(luma + (img - luma) * np.float32(1.0 + amount), 0.0, 1.0)


def _apply_vibrance(img: np.ndarray, amount: float) -> np.ndarray:
    """Saturation weighted by how unsaturated a colour already is, skin protected."""
    luma = luminance(img)[..., None]
    chroma = img - luma
    peak = channel_max(img)
    trough = channel_min(img)
    # Cheap saturation estimate; exact enough to steer a weighting.
    current = (peak - trough) / np.maximum(peak, 1e-6)

    headroom = 1.0 - np.clip(current, 0.0, 1.0)
    weight = headroom * headroom

    hue = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 0]
    weight = weight * apply_range_lut(hue, _skin_weight_lut(), 0.0, 360.0)

    factor = (1.0 + np.float32(amount) * weight)[..., None]
    return np.clip(luma + chroma * factor, 0.0, 1.0)


def _has_hsl(params: ColorParams) -> bool:
    """True when at least one band actually asks for something.

    Needed because a pydantic model is always truthy: an ``hsl`` dict full of
    zeroed bands would otherwise trigger the whole HSV round trip for nothing.
    """
    return any(
        band.hue or band.saturation or band.luminance for band in params.hsl.values()
    )


def _apply_hsl(img: np.ndarray, params: ColorParams) -> np.ndarray:
    """Per-band hue rotation, saturation and luminance, through HSV."""
    hue_lut, sat_lut, lum_lut = _hsl_luts(params)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]

    hsv[..., 0] = np.mod(hue + apply_range_lut(hue, hue_lut, 0.0, 360.0), 360.0)
    hsv[..., 1] = np.clip(
        hsv[..., 1] * (1.0 + apply_range_lut(hue, sat_lut, 0.0, 360.0)), 0.0, 1.0
    )
    hsv[..., 2] = np.clip(
        hsv[..., 2] * (1.0 + apply_range_lut(hue, lum_lut, 0.0, 360.0)), 0.0, 1.0
    )
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)


def _apply_split_toning(img: np.ndarray, params: SplitToningParams) -> np.ndarray:
    """Tint shadows and highlights towards two independent hues."""
    luma = luminance(img)
    # The balance slider slides the crossover point between the two halves.
    pivot = 0.5 + 0.35 * params.balance
    shadow_weight = np.clip(1.0 - luma / max(pivot, 1e-3), 0.0, 1.0) ** 2
    highlight_weight = np.clip((luma - pivot) / max(1.0 - pivot, 1e-3), 0.0, 1.0) ** 2

    out = img
    for hue, strength, weight in (
        (params.shadow_hue, params.shadow_saturation, shadow_weight),
        (params.highlight_hue, params.highlight_saturation, highlight_weight),
    ):
        if strength <= 0.0:
            continue
        patch = np.zeros((1, 1, 3), dtype=np.float32)
        patch[0, 0] = (float(hue), 1.0, 1.0)
        tint = cv2.cvtColor(patch, cv2.COLOR_HSV2RGB)[0, 0]
        tint_luma = float(np.dot(tint, [0.2627, 0.6780, 0.0593]))
        # Subtracting the tint's own luminance keeps the toning from also
        # brightening or darkening the region it lands on.
        offset = (tint - tint_luma) * np.float32(strength * 0.5)
        out = out + offset * weight[..., None]

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def apply(img: np.ndarray, params: ColorParams) -> np.ndarray:
    """Apply every colour adjustment to a display-referred image.

    Args:
        img: ``(H, W, 3)`` float32 in [0, 1], Rec.2020 primaries, perceptually
            encoded (the output of the tone mapping).
        params: saturation, vibrance, HSL bands and split toning.

    Returns:
        Same space and range.
    """
    out = np.ascontiguousarray(img, dtype=np.float32)

    if params.vibrance:
        out = _apply_vibrance(out, params.vibrance)
    if params.saturation:
        out = _apply_saturation(out, params.saturation)
    if _has_hsl(params):
        out = _apply_hsl(out, params)
    split = params.split_toning
    if split.shadow_saturation > 0.0 or split.highlight_saturation > 0.0:
        out = _apply_split_toning(out, split)

    return out
