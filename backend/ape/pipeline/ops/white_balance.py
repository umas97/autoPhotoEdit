# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""White balance as a chromatic adaptation in the working space.

The decode has already neutralised the camera's as-shot illuminant: after it, a
grey card shot under that illuminant sits exactly on the D65 axis of the working
space. This operation moves that neutral point to whatever illuminant the user
declares, by adapting *from* the target illuminant *to* the as-shot one in a
von Kries cone space (CAT02).

That direction is the one that gives the expected feel: raising the temperature
says "the light was bluer than I thought", so the correction warms the picture,
exactly as the slider does in every other raw processor.

Doing it here, colorimetrically, rather than as per-channel gains in camera
space, keeps the operation camera-independent -- which is what lets a learned
style profile move between bodies (section 8.4).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ...raw.whitepoint import illuminant_xyz
from ..colorspace import REC2020_TO_XYZ, XYZ_TO_REC2020, apply_matrix
from ..params import WhiteBalanceParams

__all__ = ["adaptation_matrix", "apply"]


@lru_cache(maxsize=256)
def adaptation_matrix(
    target_k: float, target_tint: float, source_k: float, source_tint: float
) -> np.ndarray:
    """Rec.2020 -> Rec.2020 matrix that re-neutralises the image on a new illuminant.

    Cached: sliders move in small steps and consecutive photos of a scene share a
    temperature, so the same handful of matrices come up again and again.
    """
    import colour

    target_xyz = illuminant_xyz(target_k, target_tint)
    source_xyz = illuminant_xyz(source_k, source_tint)
    cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(
        target_xyz, source_xyz, transform="CAT02"
    )
    matrix = XYZ_TO_REC2020 @ cat @ REC2020_TO_XYZ

    # Preserve the luminance of a neutral: without this, changing temperature
    # would also change brightness, and the user would have to chase it with the
    # exposure slider.
    neutral = matrix @ np.ones(3, dtype=np.float64)
    luma = float(REC2020_TO_XYZ[1] @ neutral)
    if luma > 1e-9:
        matrix = matrix / luma
    return np.ascontiguousarray(matrix, dtype=np.float64)


def apply(
    img: np.ndarray,
    params: WhiteBalanceParams,
    *,
    as_shot_temperature_k: float,
    as_shot_tint: float,
) -> np.ndarray:
    """Re-balance a linear scene-referred image.

    Args:
        img: ``(H, W, 3)`` float32, linear, scene-referred Rec.2020.
        params: the target illuminant, or ``mode='as_shot'`` for a no-op.
        as_shot_temperature_k: illuminant the decode already neutralised.
        as_shot_tint: its tint, same convention as ``params.tint``.

    Returns:
        Same space and range as the input.

    The as-shot reference arrives as an argument rather than inside
    ``EditParams`` on purpose: it belongs to the *photo*, not to the *look*, and
    keeping it out is what lets the same parameters be applied to a frame from a
    different body (docs/SPEC.md section 8.4). The function stays pure.
    """
    if params.mode == "as_shot":
        return img
    if (
        abs(params.temperature_k - as_shot_temperature_k) < 1.0
        and abs(params.tint - as_shot_tint) < 0.01
    ):
        return img
    matrix = adaptation_matrix(
        float(params.temperature_k),
        float(params.tint),
        float(as_shot_temperature_k),
        float(as_shot_tint),
    )
    return apply_matrix(img, matrix)
