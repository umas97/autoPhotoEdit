# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""CIELab and CIEDE2000 in float32, fast enough to sit inside an optimiser.

``pipeline/colorspace.py`` has the reference versions, through
``colour-science`` in float64: right for a test, wrong for the style inversion
of section 8.2, which evaluates the difference four hundred times per pair. The
formulas here are the same (Sharma, Wu and Dalal 2005 for CIEDE2000), written
out on flat arrays; ``tests/test_style_invert.py`` holds them to the reference
within 0.01 ΔE.

Inputs are sRGB-encoded values in [0, 1] -- what the renderer writes and what a
reference JPEG contains -- so both sides of a comparison go through exactly the
same conversion.
"""

from __future__ import annotations

import numpy as np

__all__ = ["delta_e_2000", "srgb_to_lab"]

#: Linear sRGB -> XYZ (D65), IEC 61966-2-1, rows pre-divided by the white point
#: so that the Lab step needs no further normalisation.
_SRGB_TO_XYZ_N = (
    np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    / np.array([[0.95047], [1.0], [1.08883]])
).astype(np.float32)

_DELTA = 6.0 / 29.0
_TWENTY_FIVE_7 = np.float32(25.0**7)


def srgb_to_lab(srgb: np.ndarray) -> np.ndarray:
    """``(..., 3)`` sRGB in [0, 1] -> ``(..., 3)`` CIELab (D65), float32."""
    x = np.asarray(srgb, dtype=np.float32)
    linear = np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))
    xyz = linear.astype(np.float32) @ _SRGB_TO_XYZ_N.T
    f = np.where(xyz > _DELTA**3, np.cbrt(xyz), xyz / (3 * _DELTA * _DELTA) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def delta_e_2000(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """Per-pixel CIEDE2000 between two Lab arrays of the same shape."""
    l1, a1, b1 = lab_a[..., 0], lab_a[..., 1], lab_a[..., 2]
    l2, a2, b2 = lab_b[..., 0], lab_b[..., 1], lab_b[..., 2]

    c_bar = 0.5 * (np.hypot(a1, b1) + np.hypot(a2, b2))
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + _TWENTY_FIVE_7)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    two_pi = np.float32(2.0 * np.pi)
    h1p = np.mod(np.arctan2(b1, a1p), two_pi)
    h2p = np.mod(np.arctan2(b2, a2p), two_pi)
    achromatic = (c1p * c2p) == 0

    dl = l2 - l1
    dc = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > np.pi, dh - two_pi, np.where(dh < -np.pi, dh + two_pi, dh))
    dh = np.where(achromatic, 0.0, dh)
    d_big_h = 2.0 * np.sqrt(c1p * c2p) * np.sin(dh / 2.0)

    l_bar = 0.5 * (l1 + l2)
    cp_bar = 0.5 * (c1p + c2p)
    h_sum = h1p + h2p
    h_bar = (
        np.where(
            np.abs(h1p - h2p) > np.pi,
            np.where(h_sum < two_pi, h_sum + two_pi, h_sum - two_pi),
            h_sum,
        )
        / 2.0
    )
    h_bar = np.where(achromatic, h_sum, h_bar)

    t = (
        1.0
        - 0.17 * np.cos(h_bar - np.pi / 6.0)
        + 0.24 * np.cos(2.0 * h_bar)
        + 0.32 * np.cos(3.0 * h_bar + np.pi / 30.0)
        - 0.20 * np.cos(4.0 * h_bar - 63.0 * np.pi / 180.0)
    )
    l_offset = (l_bar - 50.0) ** 2
    s_l = 1.0 + 0.015 * l_offset / np.sqrt(20.0 + l_offset)
    s_c = 1.0 + 0.045 * cp_bar
    s_h = 1.0 + 0.015 * cp_bar * t
    d_theta = (np.pi / 6.0) * np.exp(-(((np.degrees(h_bar) - 275.0) / 25.0) ** 2))
    cp_bar7 = cp_bar**7
    r_t = -2.0 * np.sqrt(cp_bar7 / (cp_bar7 + _TWENTY_FIVE_7)) * np.sin(2.0 * d_theta)

    term_c = dc / s_c
    term_h = d_big_h / s_h
    return np.sqrt((dl / s_l) ** 2 + term_c**2 + term_h**2 + r_t * term_c * term_h).astype(
        np.float32, copy=False
    )
