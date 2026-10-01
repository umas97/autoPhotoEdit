# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The style vector: the part of ``EditParams`` a style learns (section 8.2).

A fixed, ordered list of numbers, because every consumer of a style is linear
algebra: the inversion searches over it, the k-NN averages it, the coherence of
section 8.4 pulls it towards a median, the review of section 9.2 propagates a
*delta* of it. ``EditParams`` stays the contract (section 5): a vector becomes
parameters only through :func:`to_params`, and parameters become a vector only
through :func:`from_params`.

Two entries are relative on purpose, because in absolute terms they describe the
photo rather than the taste:

* **white balance** is a shift from the camera's as-shot illuminant, in mired
  (perceptually even) and tint units. "A bit warmer than the camera" transfers
  from a tungsten interior to a sunset; "3200 K" does not.
* **exposure** is an offset from the photo's *exposure anchor* -- the gain that
  would put its scene key on middle grey (``style/auto.py``). "Half a stop
  brighter than neutral" transfers from a dark frame to a bright one; "+1.5 EV"
  does not.

Split toning is stored as two Cartesian tint vectors rather than hue/strength,
so that averaging two samples of nearly the same hue on either side of 0/360
lands between them instead of on the opposite side of the wheel.

Geometry and masks are not part of a style (section 8.2): :func:`to_params`
takes them from the ``base`` parameters untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..pipeline.params import HSL_BANDS, EditParams, HSLBand

__all__ = [
    "BOUNDS",
    "NAMES",
    "NEUTRAL",
    "StyleContext",
    "UNITS",
    "clip",
    "from_params",
    "index",
    "neutral_vector",
    "to_params",
]


@dataclass(frozen=True, slots=True)
class StyleContext:
    """What a photo contributes to reading or writing a style vector."""

    as_shot_temperature_k: float
    as_shot_tint: float
    #: Stops that would put this photo's scene key on middle grey.
    exposure_anchor_ev: float = 0.0


#: ``(name, low, high, neutral)``. The bounds are the *search* range of the
#: inversion, tighter than the physical ranges of ``EditParams`` where the
#: physical range is wider than any photographic edit: a 16-stop black point
#: or a 6x shoulder are there for synthetic tests, not for looks.
_SPEC: tuple[tuple[str, float, float, float], ...] = (
    ("wb_mired_shift", -90.0, 90.0, 0.0),  # +: warmer than the camera
    ("wb_tint_shift", -40.0, 40.0, 0.0),  # +: more magenta
    ("exposure_offset", -3.0, 3.0, 0.0),
    ("highlight_strength", 0.0, 1.0, 0.7),
    ("black_point_ev", -12.0, -4.0, -8.0),
    ("white_point_ev", 2.0, 8.0, 4.5),
    ("contrast", 0.6, 2.4, 1.2),
    ("pivot", -1.5, 1.5, 0.0),
    ("toe", 0.5, 3.0, 1.2),
    ("shoulder", 0.5, 3.0, 1.5),
    ("shadows", -1.0, 1.0, 0.0),
    ("highlights", -1.0, 1.0, 0.0),
    ("whites", -1.0, 1.0, 0.0),
    ("blacks", -1.0, 1.0, 0.0),
    ("vibrance", -1.0, 1.0, 0.0),
    ("saturation", -1.0, 1.0, 0.0),
    ("local_shadows", -1.0, 1.0, 0.0),
    ("local_highlights", -1.0, 1.0, 0.0),
    ("clarity", -0.6, 1.0, 0.0),
    ("clarity_radius", 0.004, 0.08, 0.02),
    *(
        (f"hsl_{band}_{channel}", -1.0, 1.0, 0.0)
        for band in HSL_BANDS
        for channel in ("hue", "saturation", "luminance")
    ),
    ("split_shadow_x", -0.5, 0.5, 0.0),
    ("split_shadow_y", -0.5, 0.5, 0.0),
    ("split_highlight_x", -0.5, 0.5, 0.0),
    ("split_highlight_y", -0.5, 0.5, 0.0),
    ("split_balance", -1.0, 1.0, 0.0),
)

#: What one unit of each entry is worth to the eye: roughly the smallest change
#: a photographer would call "different". The inversion's L2 pull towards its
#: prior is measured in these units, so that it actually decides between two
#: solutions of equal ΔE -- measured against the full range instead, 30 mired
#: of white balance cost 0.005 and the inversion chose between warm-plus-green
#: and cool-plus-magenta at random (the same pair swung from +42 to +1.5 mired
#: between two runs, and the k-NN then averaged the coin tosses).
_UNIT: dict[str, float] = {
    "wb_mired_shift": 10.0,
    "wb_tint_shift": 5.0,
    "exposure_offset": 0.3,
    "highlight_strength": 0.3,
    "black_point_ev": 1.0,
    "white_point_ev": 0.7,
    "contrast": 0.15,
    "pivot": 0.3,
    "toe": 0.3,
    "shoulder": 0.3,
    "clarity_radius": 0.02,
    "split_balance": 0.3,
}


def _unit(name: str) -> float:
    if name in _UNIT:
        return _UNIT[name]
    if name.startswith("split_"):
        return 0.1
    if name.startswith("hsl_"):
        return 0.2
    return 0.15  # the -1..+1 sliders


NAMES: tuple[str, ...] = tuple(s[0] for s in _SPEC)
BOUNDS: np.ndarray = np.array([(s[1], s[2]) for s in _SPEC], dtype=np.float64)
NEUTRAL: np.ndarray = np.array([s[3] for s in _SPEC], dtype=np.float64)
UNITS: np.ndarray = np.array([_unit(s[0]) for s in _SPEC], dtype=np.float64)
_INDEX = {name: i for i, name in enumerate(NAMES)}


def index(name: str) -> int:
    return _INDEX[name]


def neutral_vector() -> np.ndarray:
    return NEUTRAL.copy()


def clip(vector: np.ndarray) -> np.ndarray:
    """Into the search bounds. A prediction averaged from samples always is;
    one extrapolated by a regression or moved by a delta may not be."""
    return np.clip(vector, BOUNDS[:, 0], BOUNDS[:, 1])


def _mired(kelvin: float) -> float:
    return 1e6 / kelvin


def to_params(
    vector: np.ndarray, context: StyleContext, base: EditParams | None = None
) -> EditParams:
    """The ``EditParams`` a style vector means for one photo.

    ``base`` supplies everything a style does not own -- geometry, masks,
    denoise, sharpening -- and defaults to neutral.
    """
    v = dict(zip(NAMES, (float(x) for x in clip(vector)), strict=True))
    params = (base or EditParams()).model_copy(deep=True)

    shift, tint_shift = v["wb_mired_shift"], v["wb_tint_shift"]
    if abs(shift) < 0.05 and abs(tint_shift) < 0.05:
        params.white_balance.mode = "as_shot"
    else:
        target = _mired(context.as_shot_temperature_k) - shift
        params.white_balance.mode = "custom"
        params.white_balance.temperature_k = float(np.clip(1e6 / max(target, 1.0), 1500, 25000))
        params.white_balance.tint = float(np.clip(context.as_shot_tint + tint_shift, -150, 150))

    params.exposure.ev = float(np.clip(context.exposure_anchor_ev + v["exposure_offset"], -6, 6))
    params.highlight_recovery.strength = v["highlight_strength"]
    tone = params.tone
    # Assigned together: the validator rejects a white point under the black
    # point, and assigning one at a time could pass through such a state.
    params.tone = tone.model_copy(
        update={
            "black_point_ev": v["black_point_ev"],
            "white_point_ev": v["white_point_ev"],
            "contrast": v["contrast"],
            "pivot": v["pivot"],
            "toe": v["toe"],
            "shoulder": v["shoulder"],
        }
    )
    shaping = params.tone_shaping
    shaping.shadows, shaping.highlights = v["shadows"], v["highlights"]
    shaping.whites, shaping.blacks = v["whites"], v["blacks"]

    color = params.color
    color.vibrance, color.saturation = v["vibrance"], v["saturation"]
    hsl: dict[str, HSLBand] = {}
    for band in HSL_BANDS:
        values = {c: v[f"hsl_{band}_{c}"] for c in ("hue", "saturation", "luminance")}
        if any(abs(x) > 1e-4 for x in values.values()):
            hsl[band] = HSLBand(**values)
    color.hsl = hsl
    split = color.split_toning
    for prefix, part in (("shadow", "shadow"), ("highlight", "highlight")):
        x, y = v[f"split_{prefix}_x"], v[f"split_{prefix}_y"]
        strength = min(1.0, math.hypot(x, y))
        hue = math.degrees(math.atan2(y, x)) % 360.0 if strength > 1e-6 else 0.0
        setattr(split, f"{part}_hue", 0.0 if hue >= 360.0 else hue)
        setattr(split, f"{part}_saturation", strength)
    split.balance = v["split_balance"]

    params.local_contrast.shadows = v["local_shadows"]
    params.local_contrast.highlights = v["local_highlights"]
    params.local_contrast.clarity = v["clarity"]
    params.local_contrast.radius = v["clarity_radius"]
    return params


def from_params(params: EditParams, context: StyleContext) -> np.ndarray:
    """The style vector of a set of parameters, read back for one photo.

    The inverse of :func:`to_params` inside the bounds: what the review of
    section 9.2 uses to turn the user's slider corrections into a delta.
    """
    v = dict(zip(NAMES, NEUTRAL, strict=True))
    wb = params.white_balance
    if wb.mode == "custom":
        v["wb_mired_shift"] = _mired(context.as_shot_temperature_k) - _mired(wb.temperature_k)
        v["wb_tint_shift"] = wb.tint - context.as_shot_tint
    v["exposure_offset"] = params.exposure.ev - context.exposure_anchor_ev
    v["highlight_strength"] = params.highlight_recovery.strength
    for name in ("black_point_ev", "white_point_ev", "contrast", "pivot", "toe", "shoulder"):
        v[name] = getattr(params.tone, name)
    for name in ("shadows", "highlights", "whites", "blacks"):
        v[name] = getattr(params.tone_shaping, name)
    v["vibrance"], v["saturation"] = params.color.vibrance, params.color.saturation
    for band in HSL_BANDS:
        entry = params.color.band(band)
        for channel in ("hue", "saturation", "luminance"):
            v[f"hsl_{band}_{channel}"] = getattr(entry, channel)
    split = params.color.split_toning
    for prefix in ("shadow", "highlight"):
        hue = math.radians(getattr(split, f"{prefix}_hue"))
        strength = getattr(split, f"{prefix}_saturation")
        v[f"split_{prefix}_x"] = strength * math.cos(hue)
        v[f"split_{prefix}_y"] = strength * math.sin(hue)
    v["split_balance"] = split.balance
    v["local_shadows"] = params.local_contrast.shadows
    v["local_highlights"] = params.local_contrast.highlights
    v["clarity"] = params.local_contrast.clarity
    v["clarity_radius"] = params.local_contrast.radius
    return np.array([v[name] for name in NAMES], dtype=np.float64)
