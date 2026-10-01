# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Output sharpening: an unsharp mask, applied last.

Sharpening is output-referred by nature -- it compensates for the resampling and
for the display, so it has to happen *after* the final resize, on the pixels
that will actually be written. Sharpening before a downscale throws the work
away; sharpening a full-resolution image destined for a 1200 px web export
sharpens detail nobody will ever see.

The threshold is what separates sharpening from noise amplification: differences
below it are faded out smoothly, so flat skies and shadow noise are left alone
while edges are not.
"""

from __future__ import annotations

import numpy as np

from ..colorspace import luminance
from ..filters import gaussian_blur, sigma_to_pixels
from ..params import SharpenParams

__all__ = ["apply", "unsharp"]


def apply(img: np.ndarray, params: SharpenParams) -> np.ndarray:
    """Sharpen a display-referred image.

    Args:
        img: ``(H, W, 3)`` float32 in [0, 1], Rec.2020, perceptually encoded,
            already at its final pixel dimensions.
        params: amount, long-edge-relative radius and edge threshold.

    Returns:
        Same space and range.
    """
    if params.amount <= 0.0:
        return img
    return unsharp(img, sigma_to_pixels(params.radius, img.shape), params.amount, params.threshold)


def unsharp(img: np.ndarray, sigma: float, amount: float, threshold: float) -> np.ndarray:
    """The unsharp mask itself, with the radius already in pixels.

    Separate from :func:`apply` for the output sharpening of the export
    (``RenderOptions.output_sharpening``), whose radius is a fraction of an
    *output* pixel by definition and so has no business being relative to the
    long edge.

    Args:
        img: ``(H, W, 3)`` float32 in [0, 1], perceptually encoded.
        sigma: Gaussian sigma in pixels.
        amount: gain on the detail layer.
        threshold: edges weaker than this are faded out.
    """
    if amount <= 0.0:
        return img
    luma = luminance(img)
    detail = luma - gaussian_blur(luma, sigma)

    if threshold > 0.0:
        # Smooth gate rather than a hard one: a hard threshold leaves a visible
        # discontinuity exactly where edges fade into flat areas.
        gate = np.clip(np.abs(detail) / np.float32(threshold), 0.0, 1.0)
        detail = detail * (gate * gate * (3.0 - 2.0 * gate))

    sharpened = np.clip(luma + detail * np.float32(amount), 0.0, 1.0)
    ratio = (sharpened / np.maximum(luma, 1e-4))[..., None]
    return np.clip(img * ratio, 0.0, 1.0).astype(np.float32, copy=False)
