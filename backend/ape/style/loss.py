# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The loss of the style inversion (section 8.2.3), and what it compares on.

CIEDE2000 (weight 1.0), a Wasserstein-1 distance between the L* distributions
(0.5) and between the a*/b* ones (0.3), the difference in local contrast as the
spread of the L* gradient (0.2), and an L2 pull towards a prior (0.05, applied
by ``learn.py`` in the perceptual units of ``vector.UNITS``). Everything here
is a pure function of images or samples.
"""

from __future__ import annotations

import cv2
import numpy as np

from .colordiff import delta_e_2000

__all__ = [
    "COMPARE_SIGMA",
    "CONTRAST_SIGMA",
    "TRIM",
    "W_CHROMA",
    "W_CONTRAST",
    "W_DELTA_E",
    "W_LUMA",
    "W_REG",
    "blur",
    "gradient_spread",
    "loss_terms",
    "to_display",
    "wasserstein",
]

#: Fraction of pixels the ΔE keeps (see the module docstring).
TRIM = 0.97

#: Blur before measuring local contrast, relative to the long edge: 3 px at
#: 512, well above the one-pixel scale sharpening works at.
CONTRAST_SIGMA = 0.006

#: Both images are compared after a Gaussian blur of this sigma, relative to
#: the long edge (1 px at 512). Measured on the user's pairs, about a third of
#: the per-pixel ΔE at 512 px is detail rather than colour -- the reference's
#: output sharpening, resampling, the residual of two different lens profiles
#: on grass and foliage -- and a style has no business fitting any of it: one
#: pixel of blur takes it out (7.4 -> 4.8 on a meadow) and leaves every
#: tonal and colour difference in place.
COMPARE_SIGMA = 0.002

W_DELTA_E, W_LUMA, W_CHROMA, W_CONTRAST, W_REG = 1.0, 0.5, 0.3, 0.2, 0.05

_SRGB_TO_REC2020_DISPLAY: np.ndarray | None = None


def gradient_spread(lightness: np.ndarray, mask: np.ndarray) -> float:
    """Spread of the L* gradient at the scale of local contrast, not of sharpness.

    The blur takes out the output sharpening the reference was exported with
    (and the fine detail our half-size decode never had): measured without it,
    every pair asked for maximum clarity to imitate an unsharp mask.
    """
    sigma = max(1.0, max(lightness.shape) * CONTRAST_SIGMA)
    lightness = cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma)
    gx = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.hypot(gx, gy)[mask].std())


def blur(image: np.ndarray) -> np.ndarray:
    sigma = COMPARE_SIGMA * max(image.shape[:2])
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma) if sigma >= 0.3 else image


def wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """W1 between two equal-size samples: the mean gap of their order statistics."""
    return float(np.abs(np.sort(a) - np.sort(b)).mean())


def loss_terms(
    ours_lab: np.ndarray, ref_lab: np.ndarray, *, trim: float = TRIM
) -> dict[str, float]:
    """ΔE and distribution distances between two ``(N, 3)`` Lab samples."""
    de = delta_e_2000(ours_lab, ref_lab)
    keep = max(1, int(trim * len(de)))
    trimmed = (
        float(np.partition(de, keep - 1)[:keep].mean()) if keep < len(de) else float(de.mean())
    )
    return {
        "delta_e": float(de.mean()),
        "delta_e_trimmed": trimmed,
        "luma": wasserstein(ours_lab[:, 0], ref_lab[:, 0]),
        "chroma": 0.5
        * (wasserstein(ours_lab[:, 1], ref_lab[:, 1]) + wasserstein(ours_lab[:, 2], ref_lab[:, 2])),
    }


def to_display(srgb: np.ndarray) -> np.ndarray:
    """sRGB -> the display-encoded Rec.2020 the colour stage works in."""
    global _SRGB_TO_REC2020_DISPLAY
    if _SRGB_TO_REC2020_DISPLAY is None:
        from ..pipeline.colorspace import REC2020_TO_XYZ, XYZ_TO_RGB, OutputSpace

        _SRGB_TO_REC2020_DISPLAY = (
            np.linalg.inv(REC2020_TO_XYZ) @ np.linalg.inv(XYZ_TO_RGB[OutputSpace.SRGB])
        ).astype(np.float32)
    x = np.clip(srgb, 0.0, 1.0).astype(np.float32)
    linear = np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))
    rec = np.clip(linear @ _SRGB_TO_REC2020_DISPLAY.T, 0.0, 1.0)
    return np.power(rec, 1.0 / 2.4).astype(np.float32)
