# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""darktable's module parameters, as the bytes darktable 5.6 stores in an XMP.

A darktable history entry carries its module's parameters as the raw bytes of a
C struct (``src/iop/<module>.c``, the ``dt_iop_<module>_params_t`` named by
``DT_MODULE_INTROSPECTION``), hex-encoded. darktable refuses an entry whose
size does not match its own struct for that version, so each packer below
mirrors one struct field for field, in order, little-endian, with ``int`` for
enums and ``gboolean``; the version it is for is next to it. The layouts are
those of the 5.6.1 sources; a later darktable upgrades them through its own
``legacy_params``, as it does for every old edit.

Only the parameters this program has an opinion on are computed; every other
field is the module's own ``$DEFAULT``.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DtModule",
    "ashift",
    "bilat",
    "colorbalancergb",
    "colorequal",
    "crop",
    "exposure",
    "fit_sigmoid",
    "lens",
    "shadhi",
    "sharpen",
    "sigmoid",
    "temperature",
    "tonecurve",
]


@dataclass(frozen=True, slots=True)
class DtModule:
    """One history entry: which module, which version, which bytes."""

    operation: str
    version: int
    params: bytes
    enabled: bool = True


def _floats(*values: float) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _ints(*values: int) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def temperature(multipliers: np.ndarray, *, as_shot: bool) -> DtModule:
    """White balance as channel coefficients (v4: red, green, blue, various, preset).

    ``various`` is the second green of the Bayer quad. Preset 0 is "as shot",
    2 is "user modified".
    """
    red, green, blue = (float(m) for m in multipliers[:3])
    return DtModule(
        "temperature", 4, _floats(red, green, blue, green) + _ints(0 if as_shot else 2)
    )


def exposure(ev: float) -> DtModule:
    """v7: mode, black, exposure, deflicker percentile/target, two compensations.

    Both compensations off: the exposure bias is already in our number, and the
    highlight-preservation compensation would add a correction for a module
    setting this history does not use.
    """
    return DtModule("exposure", 7, _ints(0) + _floats(0.0, ev, 50.0, -4.0) + _ints(0, 0))


def sigmoid(contrast: float, skew: float, hue_preservation: float) -> DtModule:
    """v3: contrast, skew, white/black targets, processing, hue preservation,
    three insets and rotations, purity, base primaries."""
    return DtModule(
        "sigmoid",
        3,
        _floats(contrast, skew, 100.0, 0.0152)
        + _ints(0)
        + _floats(hue_preservation, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        + _ints(0),
    )


#: darktable's scene middle grey, which its sigmoid maps onto itself.
DT_MIDDLE_GREY = 0.1845


def _dt_sigmoid(x: np.ndarray, contrast: float, skew: float) -> np.ndarray:
    """``sigmoid.c``'s curve with the default targets: scene linear -> display linear."""
    grey, white, black = DT_MIDDLE_GREY, 1.0, 0.000152

    def curve(value, magnitude, paper_exp, fog, film_power, paper_power):
        film = np.power(fog + np.maximum(value, 0.0), film_power)
        return magnitude * np.power(film / (paper_exp + film), paper_power)

    delta = 1e-6
    ref_exp = grey**contrast * (1.0 / grey - 1.0)
    ref_slope = (
        curve(grey + delta, 1.0, ref_exp, 0.0, contrast, 1.0)
        - curve(grey - delta, 1.0, ref_exp, 0.0, contrast, 1.0)
    ) / (2 * delta)
    paper_power = 5.0 ** (-skew)
    temp_exp = grey * ((white / grey) ** (1.0 / paper_power) - 1.0)
    temp_slope = (
        curve(grey + delta, white, temp_exp, 0.0, 1.0, paper_power)
        - curve(grey - delta, white, temp_exp, 0.0, 1.0, paper_power)
    ) / (2 * delta)
    film_power = ref_slope / temp_slope
    white_grey = (white / grey) ** (1.0 / paper_power) - 1.0
    white_black = (black / white) ** (-1.0 / paper_power) - 1.0
    fog = grey * white_grey ** (1 / film_power) / (
        white_black ** (1 / film_power) - white_grey ** (1 / film_power)
    )
    paper_exp = (fog + grey) ** film_power * white_grey
    return curve(x, white, paper_exp, fog, film_power, paper_power)


def fit_sigmoid(
    target_ev: np.ndarray, target_linear: np.ndarray
) -> tuple[float, float, float]:
    """darktable (contrast, skew, exposure offset) that best follows a tone response.

    Args:
        target_ev: stops relative to *our* middle grey (0.18).
        target_linear: the display-linear value we produce there.

    Returns:
        ``(contrast, skew, offset_ev)``; the offset goes into the exposure
        module, and absorbs our pivot and the black/white point normalisation,
        which darktable's curve has no parameter for.

    The fit is least squares in display-encoded terms (gamma 2.4), so that the
    shadows count as much as the eye gives them: a coarse grid, then
    Nelder-Mead. About 10 ms. Measured residual, on four tone settings from
    -7 to +5 EV: mean 0.008 of display code (two 8-bit steps), worst 0.029 at
    the deepest shadow -- darktable's curve never reaches black, ours does at
    the black point.
    """
    from scipy.optimize import minimize

    goal = np.power(np.clip(target_linear, 0.0, 1.0), 1 / 2.4)

    def cost(values: np.ndarray) -> float:
        contrast, skew, offset = (float(v) for v in values)
        if not (0.1 <= contrast <= 10.0 and -1.0 <= skew <= 1.0 and abs(offset) <= 4.0):
            return 1e3
        scene = 0.18 * np.power(2.0, target_ev + offset)
        with np.errstate(all="ignore"):
            ours = np.power(np.clip(_dt_sigmoid(scene, contrast, skew), 0.0, 1.0), 1 / 2.4)
        error = float(np.mean((ours - goal) ** 2))
        return error if math.isfinite(error) else 1e3

    grid = [
        (c, s, 0.0) for c in np.linspace(0.8, 3.0, 12) for s in np.linspace(-0.6, 0.6, 7)
    ]
    start = min(grid, key=lambda cs: cost(np.array(cs)))
    result = minimize(
        cost, np.array(start), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-9}
    )
    contrast, skew, offset = (float(v) for v in result.x)
    return float(np.clip(contrast, 0.1, 10.0)), float(np.clip(skew, -1.0, 1.0)), offset


def colorbalancergb(
    *,
    vibrance: float,
    chroma: float,
    shadows_hue: float,
    shadows_chroma: float,
    highlights_hue: float,
    highlights_chroma: float,
) -> DtModule:
    """v5, 33 fields: the four wheels, fall-offs, chroma/saturation/brilliance
    per range, the v3/v4 fulcrums, vibrance and contrast, the formula."""
    wheels = (
        0.0, shadows_chroma, shadows_hue,  # shadows Y, C, H
        0.0, 0.0, 0.0,  # midtones
        0.0, highlights_chroma, highlights_hue,  # highlights
        0.0, 0.0, 0.0,  # global
    )
    masks = (1.0, 0.0, 1.0)  # shadows weight, white fulcrum, highlights weight
    chroma_ranges = (0.0, 0.0, chroma, 0.0)  # shadows, highlights, global, midtones
    saturation = (0.0, 0.0, 0.0, 0.0, 0.0)  # global, highlights, midtones, shadows, hue angle
    brilliance = (0.0, 0.0, 0.0, 0.0)
    # mask fulcrum, vibrance, contrast fulcrum, contrast
    tail = (DT_MIDDLE_GREY, vibrance, DT_MIDDLE_GREY, 0.0)
    payload = _floats(*wheels, *masks, *chroma_ranges, *saturation, *brilliance, *tail) + _ints(1)
    return DtModule("colorbalancergb", 5, payload)


def colorequal(hue_deg: list[float], saturation: list[float], brightness: list[float]) -> DtModule:
    """v4: seven options, then saturation, hue and brightness of the eight nodes, node shift.

    The eight nodes -- red, orange, yellow, green, cyan, blue, lavender, magenta --
    are our eight HSL bands in the same order.
    """
    options = _floats(0.1, 1.0, 0.0, 1.0, 1.5, 1.0) + _ints(1)
    return DtModule(
        "colorequal",
        4,
        options + _floats(*saturation) + _floats(*hue_deg) + _floats(*brightness) + _floats(0.0),
    )


def shadhi(shadows: float, highlights: float, radius_px: float) -> DtModule:
    """v5: order, radius, shadows, white point, highlights, reserved, compress,
    two colour corrections, flags, approximation, algorithm (bilateral)."""
    return DtModule(
        "shadhi",
        5,
        _ints(0)
        + _floats(radius_px, shadows, 0.0, highlights, 0.0, 50.0, 100.0, 50.0)
        + struct.pack("<I", 127)
        + _floats(1e-6)
        + _ints(1),
    )


def bilat(detail: float) -> DtModule:
    """v3 "local contrast": local laplacian mode, sigma r and s, detail, midtone range."""
    return DtModule("bilat", 3, _ints(1) + _floats(0.5, 0.5, detail, 0.5))


def sharpen(radius_px: float, amount: float, threshold: float) -> DtModule:
    return DtModule("sharpen", 1, _floats(radius_px, amount, threshold))


_TONECURVE_NODES = 20


def tonecurve(points: list[tuple[float, float]]) -> DtModule:
    """v5, RGB mode with per-channel application: one curve on L, a and b identity.

    ``points`` map display-linear input to display-linear output -- darktable's
    pipe is linear until the output profile, so an encoded curve has to be
    translated before it gets here (``xmp_darktable._curve_nodes``).
    """
    nodes = points[:_TONECURVE_NODES]
    curves = b""
    for channel in range(3):
        channel_nodes = nodes if channel == 0 else [(0.0, 0.0), (1.0, 1.0)]
        padded = list(channel_nodes) + [(0.0, 0.0)] * (_TONECURVE_NODES - len(channel_nodes))
        curves += b"".join(_floats(x, y) for x, y in padded)
    counts = _ints(len(nodes), 2, 2)
    types = _ints(2, 2, 2)  # MONOTONE_HERMITE
    # autoscale 3 = "RGB, linked channels"; preset 0; unbound 1; preserve colours
    # 0 = none, i.e. the curve on each channel, which is what our LUT does.
    return DtModule("tonecurve", 5, curves + counts + types + _ints(3, 0, 1, 0))


def crop(left: float, top: float, right: float, bottom: float) -> DtModule:
    """v3: the four edges, normalised to the module's input; free aspect ratio."""
    return DtModule("crop", 3, _floats(left, top, right, bottom) + _ints(0, 0))


_ASHIFT_SAVED_LINES = 50


def ashift(
    rotation_deg: float,
    focal_length: float,
    crop_factor: float,
    box: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
) -> DtModule:
    """v5 "rotate and perspective", rotation only.

    ``box`` is ``(left, right, top, bottom)`` as fractions of the bounding box
    of the rotated frame. darktable's automatic cropping is computed by its
    interface and *stored* here, so a history written without the interface
    has to carry the numbers itself (``xmp_darktable._straightened_box``);
    the mode is recorded as "original format" so the interface keeps it.
    """
    head = _floats(rotation_deg, 0.0, 0.0, 0.0, focal_length, crop_factor, 100.0, 1.0)
    modes = _ints(0, 2)  # generic lens model, crop to original format
    box_bytes = _floats(*box)
    lines = _floats(*([0.0] * (_ASHIFT_SAVED_LINES * 4))) + _ints(0) + _floats(*([0.0] * 8))
    return DtModule("ashift", 5, head + modes + box_bytes + lines)


def _text(value: str, size: int = 128) -> bytes:
    encoded = value.encode("utf-8")[: size - 1]
    return encoded + b"\0" * (size - len(encoded))


def lens(
    camera: str, lens_name: str, focal: float, aperture: float, crop_factor: float
) -> DtModule:
    """v10, lensfun method, every correction, the optics left to darktable.

    ``has_been_set`` is written as false, which tells darktable to fill camera,
    lens, focal length and aperture from the file's EXIF through its own
    lensfun database -- exactly what it does when the user opens the file. With
    the names written explicitly (and ``has_been_set`` true) darktable 5.6
    loaded the entry but corrected nothing: measured on an A7 III with the
    A063, the frame came out 3.4% off in scale, the size of the uncorrected
    distortion, and with auto-detection within 0.06% of ours. The names are
    still written, for a person reading the history; the consequence is that a
    lens the user associated by hand here is not carried over.
    """
    payload = (
        _ints(1, 7, 0)  # lensfun, distortion + TCA + vignetting, correct
        + _floats(1.0, crop_factor, focal, aperture, 1000.0)
        + _ints(1)  # rectilinear
        + _text(camera)
        + _text(lens_name)
        + _ints(0)  # no TCA override
        + _floats(1.0, 1.0)  # tca_r, tca_b
        + _floats(1.0, 1.0, 1.0, 1.0, 1.0)  # embedded-metadata fine tunes, scale v1
        + _ints(1)  # metadata algorithm v2
        + _floats(1.0)  # scale_md
        + _ints(0)  # not set: darktable detects the optics itself
        + _floats(0.0, 0.5, 0.5, 0.0, 0.0)  # manual vignette off, reserved
    )
    return DtModule("lens", 10, payload)
