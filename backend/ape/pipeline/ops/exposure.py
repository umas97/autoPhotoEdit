# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exposure: a linear gain, applied scene-referred."""

from __future__ import annotations

import numpy as np

from ..params import ExposureParams

__all__ = ["apply"]


def apply(img: np.ndarray, params: ExposureParams, extra_ev: float = 0.0) -> np.ndarray:
    """Scale a linear scene-referred image by ``2 ** ev``.

    Args:
        img: ``(H, W, 3)`` float32, linear, scene-referred Rec.2020.
        params: the user's exposure, in stops.
        extra_ev: additional stops applied on top, used by the renderer to fold
            in the decoder's baseline exposure without polluting ``EditParams``
            with a camera-dependent number.

    Returns:
        Same space and range as the input, scaled. Values above 1.0 are kept:
        the tone mapping is what decides where the highlights land.
    """
    total = params.ev + extra_ev
    if abs(total) < 1e-9:
        return img
    return (img * np.float32(2.0**total)).astype(np.float32, copy=False)
