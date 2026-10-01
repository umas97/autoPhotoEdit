# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reconstruction of the channels the sensor clipped.

When one channel saturates before the others, the recorded colour is wrong in a
very visible way: a blown sky goes cyan, a blown skin highlight goes magenta.
Two things happen here, in this order:

1. **Level reconstruction.** Where one or two channels are clipped and at least
   one is not, the clipped channel is re-estimated from the surviving ones,
   using the ratio between channels measured on the pixels that are bright but
   *not yet* clipped. That ratio is a global statistic of the image, which is
   crude compared to a local solve but costs nothing and is right far more often
   than leaving the channel pinned at the clipping point.

2. **Chroma roll-off.** Approaching the clipping point, colour is progressively
   given up in favour of luminance, so the transition into a blown area fades to
   white instead of turning into a coloured blob with a hard edge.

Scene-referred and before tone mapping: reconstructed values are allowed to go
above 1.0, and it is the sigmoid that decides where they land.
"""

from __future__ import annotations

import numpy as np

from ..colorspace import channel_max, channel_min
from ..params import HighlightRecoveryParams

__all__ = ["apply"]

# Pixels between this fraction of the threshold and the threshold itself are the
# sample used to measure inter-channel ratios: bright enough to be representative
# of the highlights, not so bright that they are already compromised.
_SAMPLE_FLOOR = 0.75


def _channel_ratios(img: np.ndarray, brightest: np.ndarray, threshold: float) -> np.ndarray:
    """Average ratio of each channel to the frame's brightest channel, in highlights.

    ``brightest`` is ``channel_max(img)``, which the caller already has.
    """
    sample = (brightest > _SAMPLE_FLOOR * threshold) & (brightest < threshold)
    if np.count_nonzero(sample) < 64:
        # Not enough well-exposed highlight data to learn anything; assume the
        # highlights are neutral, which is the safe guess.
        return np.ones(3, dtype=np.float32)
    values = img[sample]
    reference = np.maximum(values.max(axis=-1, keepdims=True), 1e-6)
    ratios = np.mean(values / reference, axis=0)
    return np.clip(ratios, 0.25, 1.0).astype(np.float32)


def apply(img: np.ndarray, params: HighlightRecoveryParams) -> np.ndarray:
    """Recover clipped highlights of a linear scene-referred image.

    Args:
        img: ``(H, W, 3)`` float32, linear, scene-referred Rec.2020, sensor
            saturation at 1.0.
        params: recovery strength and the fraction of white counted as clipped.

    Returns:
        Same space; values may exceed 1.0 where a channel was reconstructed.
    """
    if params.strength <= 0.0:
        return img

    threshold = np.float32(params.threshold)
    # "Some channel clipped" is "the brightest one is", and "all clipped" is
    # "the darkest one is": the same test as ``any``/``all`` over the channel
    # axis, whose reduction runs several times slower (see ``channel_max``).
    brightest = channel_max(img)
    any_clipped = brightest >= threshold
    if not bool(any_clipped.any()):
        return img

    out = img.astype(np.float32, copy=True)
    ratios = _channel_ratios(img, brightest, float(threshold))

    # 1. Level reconstruction, only where something survived to reconstruct from.
    partial = any_clipped & (channel_min(img) < threshold)
    if bool(partial.any()):
        pixels = out[partial]
        mask = pixels >= threshold
        # What the brightest surviving channel implies for the whole pixel.
        survivors = np.where(mask, 0.0, pixels / ratios)
        implied = survivors.max(axis=-1, keepdims=True)
        rebuilt = np.maximum(pixels, implied * ratios)
        strength = np.float32(params.strength)
        out[partial] = np.where(mask, pixels + (rebuilt - pixels) * strength, pixels)

    # 2. Chroma roll-off towards the clipping point: a smoothstep from
    # _SAMPLE_FLOOR * threshold to threshold. Below its start the weight is 0
    # and ``out + (peak - out) * 0`` is ``out`` to the bit, so only the pixels
    # above it -- the highlights, a small part of most frames -- are computed.
    peak = channel_max(out)
    start = _SAMPLE_FLOOR * threshold
    bright = peak > start
    if bool(bright.any()):
        top = peak[bright][:, None]
        pixels = out[bright]
        t = np.clip((top - start) / max(float(threshold - start), 1e-6), 0.0, 1.0)
        t = (t * t * (3.0 - 2.0 * t) * np.float32(params.strength)).astype(np.float32)
        out[bright] = pixels + (top - pixels) * t
    return out
