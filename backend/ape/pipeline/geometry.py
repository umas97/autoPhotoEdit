# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Straightening and crop (docs/SPEC.md section 6.4).

The lens correction is not here: it belongs at the head of the chain, on the
frame as the sensor recorded it (``lens.py``). What is here happens late, after
local contrast, on display-referred pixels -- and happens **once**: rotation,
the implicit crop that hides the corners the rotation empties, and the user's
own crop are folded into a single affine map, so the frame is resampled one
time instead of three.

Conventions, all resolution-independent so the proxy and the export agree:

* ``rotation_deg`` > 0 turns the picture **counter-clockwise**, as OpenCV and
  every protractor do. A horizon that rises to the right by 2 degrees is
  levelled with ``rotation_deg = -2``.
* The *straightened frame* is the largest rectangle of the original aspect
  ratio that fits inside the rotated picture, centred. At 0 degrees it is the
  whole frame. It is the minimum crop section 6.4 allows: nothing that is
  picture is lost, nothing that is not picture is shown.
* ``crop`` is normalised to the straightened frame, so a crop drawn in the
  interface stays where it was drawn when the rotation is adjusted afterwards.

Colour space in and out: whatever the stage before produced (display-referred
working space at this point of the chain), float32, 3 channels.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .params import CropRect, GeometryParams

__all__ = [
    "apply",
    "inscribed_scale",
    "is_identity",
    "output_size",
    "straightened_size",
]

#: Rotations smaller than this are not rotations: a thousandth of a degree moves
#: the corner of a 6000 px frame by a tenth of a pixel.
_EPSILON_DEG = 1e-3


def is_identity(params: GeometryParams) -> bool:
    """True when neither rotation nor crop changes the frame."""
    return abs(params.rotation_deg) < _EPSILON_DEG and params.crop is None


def inscribed_scale(width: float, height: float, rotation_deg: float) -> float:
    """How much of each side survives a rotation, for a centred same-aspect crop.

    The rectangle ``k * (W, H)`` rotated back by the angle has to fit inside the
    original ``W x H``: its half-extents are ``k (W|cos| + H|sin|) / 2`` and
    ``k (W|sin| + H|cos|) / 2``, whence the two bounds.
    """
    theta = math.radians(abs(rotation_deg))
    if theta < math.radians(_EPSILON_DEG):
        return 1.0
    c, s = math.cos(theta), math.sin(theta)
    return min(width / (width * c + height * s), height / (width * s + height * c))


def straightened_size(width: int, height: int, rotation_deg: float) -> tuple[float, float]:
    """Size in pixels, not rounded, of the straightened frame."""
    k = inscribed_scale(width, height, rotation_deg)
    return width * k, height * k


def output_size(width: int, height: int, params: GeometryParams) -> tuple[int, int]:
    """``(width, height)`` of the frame after this stage, in whole pixels."""
    sw, sh = straightened_size(width, height, params.rotation_deg)
    crop = params.crop or CropRect()
    return max(1, round(sw * crop.width)), max(1, round(sh * crop.height))


def _slice(img: np.ndarray, crop: CropRect) -> np.ndarray:
    """A crop without rotation is an exact slice: no resampling at all."""
    height, width = img.shape[:2]
    x0 = min(width - 1, max(0, round(crop.x * width)))
    y0 = min(height - 1, max(0, round(crop.y * height)))
    x1 = min(width, max(x0 + 1, round((crop.x + crop.width) * width)))
    y1 = min(height, max(y0 + 1, round((crop.y + crop.height) * height)))
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def apply(img: np.ndarray, params: GeometryParams) -> np.ndarray:
    """Rotate, trim the empty corners, crop -- in one resampling.

    Args:
        img: ``(H, W, 3)`` float32, lens-corrected, upright.
        params: rotation in degrees (counter-clockwise positive) and a crop
            normalised to the straightened frame.

    Returns:
        A new array, smaller than or equal to ``img``; ``img`` itself when the
        parameters are the identity.
    """
    if is_identity(params):
        return img
    crop = params.crop or CropRect()
    if abs(params.rotation_deg) < _EPSILON_DEG:
        return _slice(img, crop)

    height, width = img.shape[:2]
    sw, sh = straightened_size(width, height, params.rotation_deg)
    out_w, out_h = output_size(width, height, params)

    # Continuous coordinates: pixel i covers [i, i + 1], its centre is i + 0.5,
    # the origin is the centre of the frame. Working in these, rather than in
    # OpenCV's integer centres, is what keeps the map exact at any resolution.
    sx = crop.width * sw / out_w
    sy = crop.height * sh / out_h
    offset = np.array(
        [crop.x * sw - sw / 2 + 0.5 * sx, crop.y * sh - sh / 2 + 0.5 * sy], dtype=np.float64
    )
    theta = math.radians(params.rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    # OpenCV's forward rotation (counter-clockwise on screen, y pointing down)
    # is [[c, s], [-s, c]]; the inverse map, output -> source, its transpose.
    inverse = np.array([[c, -s], [s, c]], dtype=np.float64)
    matrix = np.empty((2, 3), dtype=np.float64)
    matrix[:, :2] = inverse @ np.diag([sx, sy])
    matrix[:, 2] = inverse @ offset + np.array([width / 2 - 0.5, height / 2 - 0.5])

    # Bicubic: a bilinear rotation visibly softens a 24 MP frame at 1:1, which
    # is where a photographer checks a straightened horizon. Reflection at the
    # border covers the sub-pixel slack of the rounding, never real picture.
    return cv2.warpAffine(
        img,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
