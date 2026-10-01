# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The confidence score of section 9.1: how sure the program is of one photo.

Eight terms -- the seven of section 9.1, with the clipping split into its two
ends -- one per thing that can go wrong, each a **risk** between 0 and 1
with a **weight** the user can change (``Project.confidence_weights``). They
combine as independent doubts::

    confidence = product over terms of (1 - weight * risk)

so one term at full risk with weight 0.5 halves the confidence, two small
doubts compound, and a term at zero changes nothing. Under the project's
threshold (0.55 by default) the photo goes to the individual review queue,
carrying the terms that pulled it down as its reasons.

The terms, and where each number comes from:

``far``
    the nearest training sample is further than the profile's samples are
    from each other (``profile.Calibration``): from nothing at their 90th
    percentile to full risk at twice that;
``ambiguous``
    the neighbours' own edits disagree, measured the same way;
``extrapolation``
    the photo's exposure anchor lies outside the range of the samples'
    anchors -- the model is extrapolating the one parameter it predicts worst
    (0.43 EV of unexplained residual on the user's pairs, phase 6);
``white_balance``
    mixed light, from the spread of ``measure.wb_spread_mired``;
``burnt`` / ``crushed``
    the edit pushes more than 2% of the frame to white, more than 5% to
    black -- the thresholds of section 9.1 -- *beyond what the frame clips
    anyway* at the automatic exposure (``tonal.added_clipping``), and more
    than the user's own edits of the nearest samples add to theirs
    (``reference.py``). A sky that is white in the scene cannot be given back
    by a review; and deep blacks can be the look rather than the mistake;
``geometry``
    the straightening abstained on contradictory lines, found a tilt too
    large to correct, or corrected with low confidence -- unless the user set
    the rotation by hand;
``lens``
    no lensfun profile for the lens: distortion and vignetting uncorrected.

The profiles of rules (section 22) have no samples, so the first three terms
are absent for them rather than zero: the score says only what it can know.

Pure: inputs in, a :class:`Confidence` out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_WEIGHTS",
    "TERMS",
    "Confidence",
    "ConfidenceInputs",
    "Term",
    "score",
    "weights_from",
]

#: Section 9.1: "soglia di escalation regolabile in UI (default 0.55)".
DEFAULT_THRESHOLD = 0.55

TERMS = (
    "far",
    "ambiguous",
    "extrapolation",
    "white_balance",
    "burnt",
    "crushed",
    "geometry",
    "lens",
)

#: A term alone escalates a photo when ``weight * risk`` exceeds 1 - 0.55 =
#: 0.45. At 0.5 and 0.6 a doubt at full risk is enough on its own; the lens at
#: 0.3 never is, because a lens missing from lensfun is missing for every
#: photo shot with it -- the Tamron of the user's fixtures was, with the
#: lensfun data of the wheel -- and would otherwise send the whole project to
#: individual review for one uncorrected vignette.
DEFAULT_WEIGHTS: dict[str, float] = {
    "far": 0.6,
    "ambiguous": 0.5,
    "extrapolation": 0.6,
    "white_balance": 0.5,
    "burnt": 0.5,
    "crushed": 0.5,
    "geometry": 0.5,
    "lens": 0.3,
}

#: Ramps from no risk to full risk, in each term's own units.
#: ``far`` and ``ambiguous``: multiples of the profile's 90th percentile.
_RELATIVE = (1.0, 2.0)
#: Stops outside the samples' anchors. A quarter stop is within the noise of
#: the anchor itself; a full stop is well beyond what any sample showed.
_EXTRAPOLATION_EV = (0.25, 1.0)
#: Mired between the 20th and 80th percentile of the cells. Measured on the
#: user's frames: median 31 (24 fixtures) and 28 (104 event RAWs), 90th
#: percentile 115 and 61. The seven frames over 94 -- where the term alone
#: escalates -- are two indoor frames at 3250 K as shot (DSC06372-3), a church
#: lit by lamps and a stained-glass window (DSC06347), sun against blue shade
#: on snow (DSC05638), and the three frames filled with a log pile
#: (DSC05629-31), where even the greyest pixels are orange
#: wood: not mixed light, but the same ambiguity for an automatic white
#: balance, which cannot tell the colour of the wood from the colour of the
#: light (``style/auto.py`` limits its correction for that very frame).
_WB_MIRED = (40.0, 100.0)
#: Fractions of the frame: section 9.1's 2% burnt and 5% black, or the
#: style's own level when higher, plus the width of the ramp to full risk.
_BURNT = (0.02, 0.06)
_CRUSHED = (0.05, 0.10)
#: ``1 - straighten.confidence`` of a rotation that was applied.
_ROTATION_DOUBT = (0.4, 0.7)


def _ramp(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


@dataclass(slots=True)
class ConfidenceInputs:
    """Everything the score looks at. ``None`` means "not known": no risk."""

    #: From the prediction (``model.Prediction``).
    nearest: float | None = None
    dispersion: float | None = None
    exposure_anchor_ev: float | None = None
    #: From the profile (``profile.Calibration``); ``None`` for rules.
    nearest_p90: float | None = None
    dispersion_p90: float | None = None
    anchor_range: tuple[float, float] | None = None
    #: From ``review/measure.py`` and ``review/tonal.py``: the clipping the
    #: edit adds to the frame's own.
    wb_spread_mired: float | None = None
    burnt: float | None = None
    crushed: float | None = None
    #: How much the user's own edits of the nearest samples add to theirs
    #: (``reference.style_clipping``); ``None`` for a profile of rules.
    burnt_style: float | None = None
    crushed_style: float | None = None
    #: ``Photo.analysis["straighten"]``.
    straighten: dict | None = None
    #: The current rotation is not the automatic one: the user decided.
    rotation_by_user: bool = False
    #: ``True`` / ``False``, or ``None`` when the lens has not been looked up.
    lens_profile: bool | None = None


@dataclass(frozen=True, slots=True)
class Term:
    code: str
    risk: float
    weight: float
    #: The measured value behind the risk, for the sentence the UI shows.
    value: float | None = None

    @property
    def cost(self) -> float:
        return self.weight * self.risk

    def as_json(self) -> dict:
        return {
            "code": self.code,
            "risk": round(self.risk, 3),
            "weight": round(self.weight, 3),
            "value": None if self.value is None else round(self.value, 4),
        }


@dataclass(slots=True)
class Confidence:
    value: float
    terms: list[Term] = field(default_factory=list)

    def reasons(self, minimum_cost: float = 0.1) -> list[Term]:
        """The terms worth telling the user about, the heaviest first."""
        return sorted((t for t in self.terms if t.cost >= minimum_cost), key=lambda t: -t.cost)


def weights_from(stored: dict | None) -> dict[str, float]:
    """The project's weights over the defaults; unknown keys are ignored."""
    weights = dict(DEFAULT_WEIGHTS)
    for key, value in (stored or {}).items():
        if key in weights:
            weights[key] = min(1.0, max(0.0, float(value)))
    return weights


def _geometry(inputs: ConfidenceInputs) -> tuple[float, float | None]:
    level = inputs.straighten
    if not level or inputs.rotation_by_user:
        return 0.0, None
    outcome = level.get("outcome")
    if outcome == "contradictory":
        return 1.0, None
    if outcome == "too_large":
        return 0.8, level.get("measured_deg")
    if outcome == "rotated":
        doubt = 1.0 - float(level.get("confidence", 1.0))
        return _ramp(doubt, *_ROTATION_DOUBT), level.get("rotation_deg")
    return 0.0, None


def _risks(inputs: ConfidenceInputs) -> list[tuple[str, float, float | None]]:
    out: list[tuple[str, float, float | None]] = []
    if inputs.nearest is not None and inputs.nearest_p90:
        ratio = inputs.nearest / inputs.nearest_p90
        out.append(("far", _ramp(ratio, *_RELATIVE), ratio))
    if inputs.dispersion is not None and inputs.dispersion_p90:
        ratio = inputs.dispersion / inputs.dispersion_p90
        out.append(("ambiguous", _ramp(ratio, *_RELATIVE), ratio))
    if inputs.exposure_anchor_ev is not None and inputs.anchor_range is not None:
        low, high = inputs.anchor_range
        outside = max(low - inputs.exposure_anchor_ev, inputs.exposure_anchor_ev - high, 0.0)
        out.append(("extrapolation", _ramp(outside, *_EXTRAPOLATION_EV), outside))
    if inputs.wb_spread_mired is not None:
        spread = inputs.wb_spread_mired
        out.append(("white_balance", _ramp(spread, *_WB_MIRED), spread))
    for code, value, style, (floor, width) in (
        ("burnt", inputs.burnt, inputs.burnt_style, _BURNT),
        ("crushed", inputs.crushed, inputs.crushed_style, _CRUSHED),
    ):
        if value is not None:
            start = max(floor, style or 0.0)
            out.append((code, _ramp(value, start, start + width), value))
    risk, value = _geometry(inputs)
    out.append(("geometry", risk, value))
    if inputs.lens_profile is not None:
        out.append(("lens", 0.0 if inputs.lens_profile else 1.0, None))
    return out


def score(inputs: ConfidenceInputs, weights: dict[str, float] | None = None) -> Confidence:
    """The confidence of one photo, and every term that went into it."""
    weights = weights or DEFAULT_WEIGHTS
    value = 1.0
    terms = []
    for code, risk, measured in _risks(inputs):
        term = Term(code, risk, weights.get(code, DEFAULT_WEIGHTS[code]), measured)
        value *= 1.0 - term.cost
        terms.append(term)
    return Confidence(value=max(0.0, min(1.0, value)), terms=terms)
