# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Correlated colour temperature, tint and the camera's own white balance.

Two conversions live here, and they are the only place in the pipeline that
needs to know anything camera-specific about colour:

* *illuminant -> XYZ*, used by the white balance operation to build a chromatic
  adaptation transform;
* *camera multipliers -> illuminant*, used once at decode time to work out which
  illuminant the camera's as-shot setting was neutralising.

The tint convention follows Adobe's: raising the slider makes the picture more
magenta, lowering it greener, and one unit is 1/3000 of a CIE 1960 UCS Duv,
which is what makes our numbers recognisable to anyone coming from Lightroom.
The tint describes the *illuminant*, like the temperature: a positive tint is a
light above the Planckian locus (greenish, Duv > 0), and neutralising a greener
light is what turns the picture magenta.

Measured on the user's 65 Lightroom edits: with this sign our as-shot tint and
Lightroom's agree in sign and nearly in size (``export/xmp_adobe.py``). Until
parameters version 2 the sign was the opposite -- the slider turned the picture
green -- and ``pipeline/params.py`` migrates older values.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "TINT_PER_DUV",
    "camera_multipliers_to_illuminant",
    "illuminant_xyz",
    "xyz_to_cct_tint",
]

#: Adobe's scaling between the tint slider and CIE 1960 Duv.
TINT_PER_DUV = 3000.0

# Robertson's table covers 1667 K to infinity; clamping keeps the inverse
# lookup inside the range where it is actually defined.
_CCT_MIN, _CCT_MAX = 1667.0, 25000.0


def illuminant_xyz(cct: float, tint: float = 0.0) -> np.ndarray:
    """XYZ of the illuminant at ``cct`` kelvin, offset from the Planckian locus by ``tint``.

    Normalised to Y = 1, which is what a chromatic adaptation transform expects.
    """
    import colour

    cct = float(np.clip(cct, _CCT_MIN, _CCT_MAX))
    d_uv = tint / TINT_PER_DUV
    uv = colour.temperature.CCT_to_uv(
        np.array([cct, d_uv]), method="Robertson 1968"
    )
    xy = colour.UCS_uv_to_xy(uv)
    return np.asarray(colour.xy_to_XYZ(xy), dtype=np.float64)


def xyz_to_cct_tint(xyz: np.ndarray) -> tuple[float, float]:
    """Inverse of :func:`illuminant_xyz`. Returns ``(cct_kelvin, tint)``."""
    import colour

    xy = colour.XYZ_to_xy(np.asarray(xyz, dtype=np.float64))
    uv = colour.xy_to_UCS_uv(xy)
    cct, d_uv = colour.temperature.uv_to_CCT(uv, method="Robertson 1968")
    return float(np.clip(cct, _CCT_MIN, _CCT_MAX)), float(d_uv * TINT_PER_DUV)


def camera_multipliers_to_illuminant(
    multipliers: np.ndarray, cam_from_xyz: np.ndarray
) -> np.ndarray:
    """Which illuminant a set of camera white-balance multipliers neutralises.

    The DNG model states that the camera's neutral is ``cam_from_xyz @ XYZ_illum``
    and that the multipliers are its reciprocal; inverting both steps recovers the
    illuminant, up to a scale that we normalise away.

    Args:
        multipliers: the three RGB multipliers, any scaling.
        cam_from_xyz: the camera's colour matrix as LibRaw reports it, *not*
            renormalised -- the multipliers live in that same unnormalised space.

    Returns:
        XYZ of the illuminant, normalised to Y = 1.
    """
    mults = np.asarray(multipliers, dtype=np.float64)[:3]
    if np.any(mults <= 0) or not np.all(np.isfinite(mults)):
        raise ValueError(f"invalid white balance multipliers: {multipliers!r}")
    neutral = 1.0 / (mults / mults[1])
    xyz = np.linalg.solve(np.asarray(cam_from_xyz, dtype=np.float64), neutral)
    if xyz[1] <= 0:
        raise ValueError("white balance multipliers imply a non-physical illuminant")
    return xyz / xyz[1]


def normalise_camera_matrix(cam_from_xyz: np.ndarray, white_xyz: np.ndarray) -> np.ndarray:
    """Rescale the rows of a camera matrix so ``white_xyz`` maps to (1, 1, 1).

    After this, feeding the matrix an image whose white balance has already been
    applied puts the neutral axis exactly on ``white_xyz`` -- which is how the
    decode lands on a D65-referred working space without a separate adaptation.
    """
    matrix = np.asarray(cam_from_xyz, dtype=np.float64)
    response = matrix @ np.asarray(white_xyz, dtype=np.float64)
    if np.any(np.abs(response) < 1e-9):
        raise ValueError("camera matrix maps the reference white to zero in some channel")
    return matrix / response[:, None]
