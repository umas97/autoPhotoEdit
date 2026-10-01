# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Light noise reduction, scene-referred.

Denoising belongs before the tone mapping: the sigmoid lifts the shadows by
several stops, and whatever noise is still there when it does gets lifted with
them. Afterwards is too late.

Sensor noise is signal-dependent -- shot noise grows with the square root of the
signal -- so the filtering happens on ``sqrt(linear)``, a variance-stabilising
transform under which the noise is roughly uniform across the tonal range and a
single filter strength means the same thing in the shadows and in the highlights.

Luminance and chrominance are separated because they need very different
treatment: chroma noise is low-frequency and can be smoothed hard without any
visible loss of detail, luminance noise cannot. That is why the chrominance
default is non-zero and the luminance default is zero.
"""

from __future__ import annotations

import numpy as np

from ..colorspace import luminance
from ..filters import guided_filter
from ..params import NoiseParams

__all__ = ["apply"]

# Edge-preservation thresholds, expressed in the square-root signal space where
# the whole range spans 0..1. Chroma tolerates a much larger epsilon because a
# genuine chroma edge is far stronger than chroma noise.
_CHROMA_EPS_BASE = 0.006
_CHROMA_EPS_SPAN = 0.050
_LUMA_EPS_BASE = 0.002
_LUMA_EPS_SPAN = 0.012
# The luminance filter deliberately works at a quarter of the chroma radius:
# beyond that it stops removing grain and starts removing texture.
_LUMA_RADIUS_FACTOR = 0.25


def apply(img: np.ndarray, params: NoiseParams) -> np.ndarray:
    """Denoise a linear scene-referred image.

    Args:
        img: ``(H, W, 3)`` float32, linear, scene-referred Rec.2020.
        params: luminance and chrominance strengths in [0, 1] and a
            long-edge-relative radius.

    Returns:
        Same space and range as the input.
    """
    if params.luminance <= 0.0 and params.chrominance <= 0.0:
        return img

    # Square roots taken in place of the clipped copies: every temporary here
    # is a full frame of fresh pages.
    stabilised = np.clip(img, 0.0, None)
    np.sqrt(stabilised, out=stabilised)
    luma = np.clip(luminance(img), 0.0, None)
    np.sqrt(luma, out=luma)
    shape = img.shape

    if params.chrominance > 0.0:
        eps = (_CHROMA_EPS_BASE + _CHROMA_EPS_SPAN * params.chrominance) ** 2
        # In place, here and below: at full resolution every one of these is a
        # 290 MB frame, and the stabilised signal is rebuilt from the filtered
        # chroma anyway. Same arithmetic, same bits.
        chroma = stabilised
        chroma -= luma[..., None]
        # The three channels in one call: they share the guide.
        chroma = guided_filter(luma, chroma, params.radius, eps, shape, out=chroma)
        chroma += luma[..., None]
        stabilised = chroma

    if params.luminance > 0.0:
        eps = (_LUMA_EPS_BASE + _LUMA_EPS_SPAN * params.luminance) ** 2
        radius = max(params.radius * _LUMA_RADIUS_FACTOR, 0.0005)
        smoothed_luma = guided_filter(luma, luma, radius, eps, shape)
        # Blend rather than replace: full strength would be a wash at the top of
        # the slider, and the slider needs to be usable along its whole travel.
        delta = (smoothed_luma - luma) * np.float32(params.luminance)
        stabilised += delta[..., None]

    np.clip(stabilised, 0.0, None, out=stabilised)
    return np.square(stabilised, out=stabilised).astype(np.float32, copy=False)
