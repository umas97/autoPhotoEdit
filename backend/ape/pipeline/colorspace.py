# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Working space, output spaces, transfer functions and gamut mapping.

Two distinct spaces travel through the pipeline and must not be confused:

``scene``
    Linear, scene-referred, ITU-R BT.2020 primaries, D65 white, float32,
    normalised so that a diffuse white sits near 1.0. Values above 1.0 are legal
    and carry the specular highlights. Everything physical (white balance,
    exposure, highlight recovery, denoise) happens here.

``display``
    Display-referred, still BT.2020 primaries, but perceptually encoded with a
    pure power function (``DISPLAY_GAMMA``) and bounded to [0, 1]. Everything
    perceptual (tone shaping, curves, colour, clarity, sharpening) happens here,
    because that is the space those operations were conceived in: a tone curve
    drawn on a 0-255 axis behaves as the user expects only in a perceptual
    encoding.

The sigmoid tone mapping (``ops/tone.py``) is the one bridge between the two.

The matrices are derived from the published primaries rather than hardcoded, so
that the derivation is auditable; all spaces here share the D65 white point, so
no chromatic adaptation is needed between them.
"""

from __future__ import annotations

from enum import StrEnum

import cv2
import numpy as np

# Scene-linear value of a photographic middle grey card (18% reflectance). Used
# as the anchor of the tone mapping and of every EV-denominated parameter.
MIDDLE_GREY = 0.1845

# Perceptual encoding of the display-referred stage. 2.4 rather than 2.2 because
# it is the exponent of the sRGB/BT.1886 EOTF, so a straight line in this space
# is close to a straight line on the screen.
DISPLAY_GAMMA = 2.4

# Display-encoded position of middle grey: 0.18 ** (1 / 2.4).
DISPLAY_GREY = float(0.18 ** (1.0 / DISPLAY_GAMMA))


class OutputSpace(StrEnum):
    SRGB = "srgb"
    DISPLAY_P3 = "display-p3"
    ADOBE_RGB = "adobe-rgb"
    REC2020 = "rec2020"


# (red, green, blue) chromaticities and white point, CIE 1931 xy.
_D65 = (0.3127, 0.3290)
_PRIMARIES: dict[str, tuple[tuple[float, float], ...]] = {
    OutputSpace.SRGB: ((0.6400, 0.3300), (0.3000, 0.6000), (0.1500, 0.0600)),
    OutputSpace.DISPLAY_P3: ((0.6800, 0.3200), (0.2650, 0.6900), (0.1500, 0.0600)),
    OutputSpace.ADOBE_RGB: ((0.6400, 0.3300), (0.2100, 0.7100), (0.1500, 0.0600)),
    OutputSpace.REC2020: ((0.7080, 0.2920), (0.1700, 0.7970), (0.1310, 0.0460)),
}

# Encoding exponent of each output space. sRGB and Display P3 share the sRGB
# piecewise curve; Adobe RGB uses a pure 563/256 power.
_ADOBE_GAMMA = 563.0 / 256.0


def _xy_to_xyz(xy: tuple[float, float]) -> np.ndarray:
    x, y = xy
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def _rgb_to_xyz_matrix(primaries: tuple[tuple[float, float], ...], white: tuple[float, float]):
    """Standard derivation of an RGB-to-XYZ matrix from primaries and white point."""
    m = np.column_stack([_xy_to_xyz(p) for p in primaries])
    scale = np.linalg.solve(m, _xy_to_xyz(white))
    return m * scale


RGB_TO_XYZ: dict[str, np.ndarray] = {
    name: _rgb_to_xyz_matrix(prim, _D65) for name, prim in _PRIMARIES.items()
}
XYZ_TO_RGB: dict[str, np.ndarray] = {name: np.linalg.inv(m) for name, m in RGB_TO_XYZ.items()}

#: Working space matrices, exposed under stable names.
REC2020_TO_XYZ = RGB_TO_XYZ[OutputSpace.REC2020]
XYZ_TO_REC2020 = XYZ_TO_RGB[OutputSpace.REC2020]

#: Rec.2020 luminance weights (row 1 of the RGB-to-XYZ matrix).
LUMA_WEIGHTS = REC2020_TO_XYZ[1].astype(np.float32)


def apply_matrix(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 colour matrix to an ``(H, W, 3)`` image, in float32."""
    m = np.ascontiguousarray(matrix.T, dtype=np.float32)
    flat = img.reshape(-1, 3)
    return (flat @ m).reshape(img.shape).astype(np.float32, copy=False)


def channel_max(img: np.ndarray) -> np.ndarray:
    """Per-pixel maximum across channels, as ``(H, W)`` float32.

    Pairwise ``np.maximum`` on channel views rather than ``img.max(axis=-1)``:
    the reduction over the last, shortest axis has a bad access pattern and runs
    several times slower on a full-resolution frame.
    """
    return np.maximum(np.maximum(img[..., 0], img[..., 1]), img[..., 2])


def channel_min(img: np.ndarray) -> np.ndarray:
    """Per-pixel minimum across channels, as ``(H, W)`` float32."""
    return np.minimum(np.minimum(img[..., 0], img[..., 1]), img[..., 2])


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec.2020 relative luminance of a linear image. Returns ``(H, W)`` float32."""
    return (img @ LUMA_WEIGHTS).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Perceptual encoding of the display-referred stage
# --------------------------------------------------------------------------- #


def _power(values: np.ndarray, exponent: float) -> np.ndarray:
    """Elementwise power on non-negative float32.

    ``cv2.pow`` rather than ``np.power``: it is roughly three times faster on the
    tens of millions of elements a full-resolution frame carries, and the whole
    output stage is made of these. It needs a contiguous float32 array and
    non-negative input, both of which the callers here guarantee.

    OpenCV treats a 1-D array as a single-column matrix and hands back an
    ``(N, 1)`` result. Left alone, that broadcasts against the input in the very
    next expression and silently produces an ``(N, N)`` array -- which is how a
    corrupt tone response curve ends up inside an ICC profile. The shape is
    restored here rather than at each call site.
    """
    clipped = np.ascontiguousarray(np.clip(values, 0.0, None), dtype=np.float32)
    return cv2.pow(clipped, float(exponent)).reshape(clipped.shape)


def display_encode(linear: np.ndarray) -> np.ndarray:
    """Display-linear [0, 1] -> perceptual [0, 1]. Negatives are clamped."""
    return _power(linear, 1.0 / DISPLAY_GAMMA)


def display_decode(encoded: np.ndarray) -> np.ndarray:
    """Perceptual [0, 1] -> display-linear [0, 1]. Exact inverse of the above."""
    return _power(encoded, DISPLAY_GAMMA)


# --------------------------------------------------------------------------- #
# Output transfer functions
# --------------------------------------------------------------------------- #


def _srgb_oetf(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(np.clip(x, 0.0, 1.0), dtype=np.float32)
    # The power is evaluated everywhere and the few elements below the knee
    # patched afterwards, in place: on a proxy every extra temporary is 34 MB
    # of fresh pages, and the output stage is bound by memory, not arithmetic.
    # Already clipped, so straight to ``cv2.pow`` rather than ``_power``.
    out = cv2.pow(x, 1.0 / 2.4).reshape(x.shape)
    out *= 1.055
    out -= 0.055
    linear = x <= 0.0031308
    out[linear] = x[linear] * 12.92
    return out


def _srgb_eotf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def _rec2020_oetf(x: np.ndarray) -> np.ndarray:
    # BT.2020 uses the BT.709 curve; beta/alpha at 12-bit precision.
    alpha, beta = 1.0993, 0.0181
    x = np.clip(x, 0.0, 1.0)
    return np.where(x < beta, x * 4.5, alpha * _power(x, 0.45) - (alpha - 1.0))


def encode_transfer(linear: np.ndarray, space: OutputSpace) -> np.ndarray:
    """Apply the output space's OETF to display-linear [0, 1] values."""
    if space in (OutputSpace.SRGB, OutputSpace.DISPLAY_P3):
        out = _srgb_oetf(linear)
    elif space == OutputSpace.ADOBE_RGB:
        out = _power(np.clip(linear, 0.0, 1.0), 1.0 / _ADOBE_GAMMA)
    elif space == OutputSpace.REC2020:
        out = _rec2020_oetf(linear)
    else:  # pragma: no cover - StrEnum is exhaustive
        raise ValueError(f"unknown output space {space!r}")
    return out.astype(np.float32, copy=False)


def decode_transfer(encoded: np.ndarray, space: OutputSpace) -> np.ndarray:
    """Inverse of :func:`encode_transfer`, for reading reference images back in."""
    if space in (OutputSpace.SRGB, OutputSpace.DISPLAY_P3):
        out = _srgb_eotf(encoded)
    elif space == OutputSpace.ADOBE_RGB:
        out = _power(np.clip(encoded, 0.0, 1.0), _ADOBE_GAMMA)
    elif space == OutputSpace.REC2020:
        x = np.clip(encoded, 0.0, 1.0)
        alpha, beta = 1.0993, 0.0181
        out = np.where(
            x < beta * 4.5, x / 4.5, _power((x + alpha - 1.0) / alpha, 1.0 / 0.45)
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown output space {space!r}")
    return out.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Gamut mapping
# --------------------------------------------------------------------------- #

# Chroma distance at which compression starts (below it, colours pass through
# untouched) and the distance that is mapped to the gamut boundary. A threshold
# of 0.80 leaves the great majority of real-world colours bit-identical while
# still giving the compression somewhere to work.
_GAMUT_THRESHOLD = 0.80
_GAMUT_LIMIT = 1.60


def compress_gamut(rgb: np.ndarray, threshold: float = _GAMUT_THRESHOLD) -> np.ndarray:
    """Pull out-of-gamut colours back inside by compressing chroma, not clipping.

    Works on display-linear RGB already expressed in the *target* primaries,
    where an out-of-gamut colour shows up as a negative channel. Distance from
    the achromatic axis is measured per channel as ``(max - c) / max``; a value
    above 1 means that channel went negative. That distance is compressed with a
    smooth, monotonic curve that is the identity below ``threshold`` and
    asymptotes to 1, so hue and lightness are preserved while saturation gives
    way -- which is what perceptual gamut mapping is supposed to trade.
    """
    out = rgb.astype(np.float32, copy=True)
    achromatic = channel_max(out)
    # Pixels at or below black have no chroma to compress and would divide by 0.
    safe = np.maximum(achromatic, np.float32(1e-6))
    # A pixel is touched when its furthest channel lies beyond ``threshold``:
    # (achromatic - min) / safe > threshold. Found with two passes, so the
    # curve below -- several more, a square root among them -- runs on those
    # pixels only: a few percent of a real frame, the saturated flowers and
    # lights, rather than all of it as soon as one pixel is out of gamut.
    touched = channel_min(out) < achromatic - np.float32(threshold) * safe
    if bool(touched.any()):
        pixels = out[touched]
        peak = achromatic[touched][:, None]
        scale = safe[touched][:, None]
        distance = (peak - pixels) / scale
        span = _GAMUT_LIMIT - threshold
        excess = distance - threshold
        compressed = np.where(
            excess > 0.0,
            threshold + span * (excess / span) / np.sqrt(1.0 + np.square(excess / span)),
            distance,
        )
        out[touched] = peak - compressed * scale
    # Whatever survives above white is a luminance overflow, not a chroma one.
    np.clip(out, 0.0, 1.0, out=out)
    return out


def working_to_output(display_linear: np.ndarray, space: OutputSpace) -> np.ndarray:
    """Rec.2020 display-linear -> target primaries, gamut-compressed, still linear."""
    if space != OutputSpace.REC2020:
        matrix = XYZ_TO_RGB[space] @ REC2020_TO_XYZ
        converted = apply_matrix(display_linear, matrix)
    else:
        converted = display_linear.astype(np.float32, copy=True)
    return compress_gamut(converted)


def to_lab(display_linear_rec2020: np.ndarray) -> np.ndarray:
    """CIE L*a*b* of a Rec.2020 display-linear image, for tests and loss functions.

    ``colour`` is imported lazily: it is a heavy import and the render path does
    not need it.
    """
    import colour

    xyz = apply_matrix(display_linear_rec2020, REC2020_TO_XYZ).astype(np.float64)
    return colour.XYZ_to_Lab(xyz, illuminant=np.array(_D65))


def delta_e_2000(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """Per-pixel CIEDE2000 difference between two Lab images."""
    import colour

    return colour.difference.delta_E_CIE2000(lab_a, lab_b)
