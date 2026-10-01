# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The profiles that ship with the program (section 22).

The program has to be useful before anyone has trained anything. Four
profiles are always there, as ``StyleProfile`` rows with ``builtin = true`` and
no model; their ``builtin_rules`` are the JSON of :class:`BuiltinRules`, and
:func:`predict` turns them into a style vector for one photo. From there on
everything -- coherence, confidence, review, feedback -- is the same code path
a learned profile takes (section 22: "nessun percorso speciale").

A rule set is a *base* (entries of the style vector that differ from neutral)
plus a handful of adaptations to the scene, each a documented number:

``auto_exposure``
    fraction of the exposure anchor applied (``style/auto.py``). 1 would put
    every scene's key on middle grey; less keeps a dark scene darker, which
    is what it looked like.
``auto_white_balance``
    fraction of the grey-world white balance applied, the rest staying with
    the camera. The grey world is already confined near the as-shot value.
``range_contrast``
    contrast change per unit of dynamic range below or above
    :data:`TYPICAL_RANGE` (the L* spread of the neutral rendering, 1st to
    99th percentile): flat scenes get a little more, contrasty ones less.
``clip_recovery``
    extra local-highlight recovery per unit of clipped fraction: blown skies
    get more, frames without clipping none.

What each profile does and why, in the words the interface uses, is in
:data:`DESCRIPTIONS`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..analysis.scene import FEATURE_NAMES
from . import vector as sv

__all__ = [
    "BUILTIN_NAMES",
    "BuiltinRules",
    "DESCRIPTIONS",
    "NEUTRAL_AUTO",
    "RULES",
    "predict",
]

NEUTRAL_AUTO = "Neutro automatico"
BUILTIN_NAMES = (NEUTRAL_AUTO, "Naturale", "Ritratto", "Paesaggio")

#: L* spread (p99 - p01, 0..1) of a neutral rendering of an ordinary scene:
#: the median over the user's 89 frames is 0.81 (24 fixtures 0.83, 65 pairs 0.78).
TYPICAL_RANGE = 0.81

_I_P01 = FEATURE_NAMES.index("lum_p01")
_I_P99 = FEATURE_NAMES.index("lum_p99")
_I_CLIP = FEATURE_NAMES.index("clipped_high")


@dataclass(frozen=True)
class BuiltinRules:
    base: dict[str, float] = field(default_factory=dict)
    auto_exposure: float = 0.5
    auto_white_balance: float = 0.4
    range_contrast: float = 0.0
    clip_recovery: float = 0.0

    def as_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> BuiltinRules:
        return cls(
            base={k: float(v) for k, v in (data.get("base") or {}).items()},
            auto_exposure=float(data.get("auto_exposure", 0.5)),
            auto_white_balance=float(data.get("auto_white_balance", 0.4)),
            range_contrast=float(data.get("range_contrast", 0.0)),
            clip_recovery=float(data.get("clip_recovery", 0.0)),
        )


RULES: dict[str, BuiltinRules] = {
    # Section 22: auto white balance, auto exposure on the histogram, standard
    # sigmoid, moderate contrast and saturation, no intended colour cast.
    #
    # Half of the anchor, measured: against the user's 65 Lightroom edits the
    # mean ΔE of this profile is 9.4 at 0.8 of the anchor, 7.9 at 0.6, 7.3 at
    # 0.5 (and the worst photo the lowest, 20.0), 6.9 at 0.4 -- and the user's
    # own exposures follow the anchor with a slope of 0.49. A full
    # normalisation turns a candle-lit church into an office.
    #
    # 0.4 of the grey world: the user shot "As Shot" throughout, and on those
    # edits every tenth of automatic white balance costs about 0.06 ΔE; the
    # section asks for an automatic balance, and 0.4 keeps it within 10 mired
    # of the camera's (see ``auto.WB_LIMIT_MIRED``).
    NEUTRAL_AUTO: BuiltinRules(
        base={"vibrance": 0.10},
        auto_exposure=0.5,
        auto_white_balance=0.4,
        range_contrast=0.3,
    ),
    # Faithful: mostly the camera's white balance, which is usually right
    # about the mood of the light, medium
    # contrast, a touch of vibrance and of local balance.
    "Naturale": BuiltinRules(
        base={
            "contrast": 1.25,
            "vibrance": 0.15,
            "local_highlights": -0.20,
            "local_shadows": 0.15,
            "clarity": 0.05,
            "highlight_strength": 0.8,
        },
        auto_exposure=0.5,
        auto_white_balance=0.3,
        range_contrast=0.4,
        clip_recovery=0.6,
    ),
    # Skin first: the orange and red bands held slightly below neutral
    # saturation and lifted in luminance, so no other control can push skin
    # orange; softer shadows (lower toe, lifted local shadows); contained
    # saturation; clarity *negative*, because clarity on faces ages them.
    "Ritratto": BuiltinRules(
        base={
            "contrast": 1.1,
            "toe": 0.9,
            "local_shadows": 0.25,
            "local_highlights": -0.25,
            "saturation": -0.05,
            "vibrance": 0.05,
            "clarity": -0.10,
            "hsl_orange_saturation": -0.10,
            "hsl_orange_luminance": 0.06,
            "hsl_red_saturation": -0.08,
            "hsl_red_luminance": 0.03,
            "wb_mired_shift": 4.0,
        },
        auto_exposure=0.55,
        auto_white_balance=0.4,
        range_contrast=0.3,
        clip_recovery=0.5,
    ),
    # More contrast and local contrast, selective vibrance on greens, aquas and
    # blues, the strongest highlight recovery of the four (skies).
    "Paesaggio": BuiltinRules(
        base={
            "contrast": 1.35,
            "clarity": 0.25,
            "vibrance": 0.20,
            "hsl_green_saturation": 0.15,
            "hsl_aqua_saturation": 0.10,
            "hsl_blue_saturation": 0.20,
            "hsl_blue_luminance": -0.05,
            "highlight_strength": 1.0,
            "local_highlights": -0.45,
            "local_shadows": 0.20,
            "whites": 0.05,
        },
        auto_exposure=0.5,
        auto_white_balance=0.4,
        range_contrast=0.4,
        clip_recovery=1.0,
    ),
}

DESCRIPTIONS: dict[str, str] = {
    NEUTRAL_AUTO: (
        "Bilanciamento del bianco ed esposizione automatici, curva standard, contrasto e "
        "saturazione moderati. È il punto di partenza e il riferimento di confronto."
    ),
    "Naturale": (
        "Fedele alla scena: bianco quasi quello della fotocamera, contrasto medio, "
        "un po' di vividezza, luci e ombre riequilibrate con misura."
    ),
    "Ritratto": (
        "Pelle protetta (arancio e rosso trattenuti), ombre morbide, saturazione contenuta "
        "e chiarezza leggermente negativa, che sui volti non invecchia."
    ),
    "Paesaggio": (
        "Contrasto e contrasto locale più marcati, vividezza selettiva su verdi e blu, "
        "recupero delle alte luci più deciso per i cieli."
    ),
}


def predict(
    rules: BuiltinRules,
    features: np.ndarray,
    *,
    exposure_anchor_ev: float,
    auto_wb_mired: float,
    auto_wb_tint: float,
) -> np.ndarray:
    """The style vector a built-in profile gives one photo.

    The vector's exposure entry is relative to the anchor (``style/vector.py``),
    so applying a fraction ``g`` of the anchor means an offset of
    ``(g - 1) * anchor``.
    """
    v = sv.neutral_vector()
    for name, value in rules.base.items():
        v[sv.index(name)] = value

    v[sv.index("exposure_offset")] += (rules.auto_exposure - 1.0) * exposure_anchor_ev
    v[sv.index("wb_mired_shift")] += rules.auto_white_balance * auto_wb_mired
    v[sv.index("wb_tint_shift")] += rules.auto_white_balance * auto_wb_tint

    spread = float(features[_I_P99] - features[_I_P01])
    if np.isfinite(spread) and rules.range_contrast:
        v[sv.index("contrast")] += rules.range_contrast * (TYPICAL_RANGE - spread)
    clipped = float(features[_I_CLIP])
    if np.isfinite(clipped) and rules.clip_recovery:
        v[sv.index("local_highlights")] -= rules.clip_recovery * min(clipped, 0.3)
    return sv.clip(v)
