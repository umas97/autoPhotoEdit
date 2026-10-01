# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How much of a frame an edit burns or crushes, without rendering it (section 9.1).

The tails of ``review/measure.py`` say at what scene EV each fraction of the
frame sits, at the bright end and at the dark one. An edit maps scene EV to display
code through its exposure, its sigmoid and its shaping curve; the EV at which
that map reaches the white (black) threshold is found on a grid, and the
fraction of the frame beyond it is read off the tail table.

The tone is modelled pointwise: exposure, ``tone``, ``tone_shaping``, the tone
curve, and the local shadows and highlights taken at the pixel's own
brightness. Clarity and highlight recovery move a few pixels either way, and
the white balance and colour move the channels relative to each other; the
estimate is checked against real renders in ``tests/test_review_measure.py``.
Pure: numbers in, numbers out.
"""

from __future__ import annotations

import numpy as np

from ..pipeline.lut import compose
from ..pipeline.ops.local_contrast import local_tone_gain
from ..pipeline.ops.tone import build_shaping_lut, sigmoid_response
from ..pipeline.ops.tone_curve import build_lut
from ..pipeline.params import EditParams
from .measure import TAIL_FRACTIONS

__all__ = ["HIGH_CODE", "LOW_CODE", "added_clipping", "clipped_after", "reference_params"]

#: Display code (Rec.2020, gamma 2.4) past which a channel counts as burnt.
#: 250/255 in sRGB -- the level ``scene.py`` counts as clipped -- would be
#: 0.981, but the chroma inset of the sigmoid lets a saturated channel reach
#: the top a little before the brightness model says it should: measured
#: against real renders of the user's 128 frames under six edits (neutral,
#: +-1 and +2 EV, contrast 1.9, whites/blacks), 0.981 under-counted on every
#: brightening edit (median -0.016 at +2 EV; 10 of 33 frames over 2% missed at
#: +1), 0.96 over-counted (16 false alarms at +1), and 0.97 has no bias
#: (median error under 0.002) and agrees on "over 2%" for 120 to 128 frames of
#: 128 depending on the edit.
HIGH_CODE = 0.97
#: 4/255 in sRGB: 0.0157, 0.0012 linear. Agrees with the renders on "over 5%
#: black" for 111 to 128 frames of 128 depending on the edit; the misses are
#: frames with a large area at 0 in the proxy, whose depth below the first
#: 8-bit level the proxy cannot know (``measure._FIRST_LEVEL_CODE``).
LOW_CODE = 0.0012 ** (1.0 / 2.4)

_EV_GRID = np.linspace(-16.0, 12.0, 1401)
_FRACTIONS = np.asarray(TAIL_FRACTIONS)


def _response(params: EditParams) -> np.ndarray:
    """Display code of every EV of the grid (EV of the *neutral* scene)."""
    code = sigmoid_response(_EV_GRID + params.exposure.ev, params.tone)
    lut = compose(build_shaping_lut(params.tone_shaping), build_lut(params.tone_curve))
    code = np.interp(code, np.linspace(0.0, 1.0, len(lut)), lut.astype(np.float64))
    local = params.local_contrast
    if abs(local.shadows) > 1e-6 or abs(local.highlights) > 1e-6:
        # The gain weighs on the base layer, the brightness of the region. The
        # tails that decide "2% burnt" or "5% black" are regions -- a sky, the
        # dark of a stage -- whose base layer is the pixel itself, so the gain
        # is taken at the pixel's own code. Without it, the lifted shadows of
        # the user's style read as crushed: 18 false alarms in 130 renders of
        # their pairs, predicted and true edits (measured in phase 7).
        gain = local_tone_gain(code.astype(np.float32), local.shadows, local.highlights)
        code = np.clip(code * gain, 0.0, 1.0)
    return code


def _fraction_beyond(threshold_ev: float, table: np.ndarray, *, above: bool) -> float:
    """Fraction of the frame past ``threshold_ev``, interpolated in log-fraction.

    ``table[i]`` is the EV at fraction ``TAIL_FRACTIONS[i]``: decreasing along
    the table for the high tail, increasing for the low one.
    """
    values = table if above else -table
    target = threshold_ev if above else -threshold_ev
    # ``values`` falls as the fraction grows; np.interp wants rising x.
    x, y = values[::-1], np.log(_FRACTIONS[::-1])
    if target >= x[-1]:
        return 0.0 if target > x[-1] + 0.25 else float(_FRACTIONS[0])
    if target <= x[0]:
        return float(_FRACTIONS[-1])
    return float(np.exp(np.interp(target, x, y)))


def clipped_after(tails: dict | None, params: EditParams) -> tuple[float, float] | None:
    """``(burnt, crushed)`` fractions of the frame after ``params``.

    ``None`` without tails (a photo predicted before phase 7).
    """
    if not tails or "high" not in tails or "low" not in tails:
        return None
    code = _response(params)
    high_ev = _EV_GRID[np.argmax(code >= HIGH_CODE)] if np.any(code >= HIGH_CODE) else np.inf
    low_ev = _EV_GRID[np.nonzero(code <= LOW_CODE)[0][-1]] if np.any(code <= LOW_CODE) else -np.inf
    high = np.asarray(tails["high"], dtype=np.float64)
    low = np.asarray(tails["low"], dtype=np.float64)
    burnt = 0.0 if not np.isfinite(high_ev) else _fraction_beyond(float(high_ev), high, above=True)
    crushed = 0.0 if not np.isfinite(low_ev) else _fraction_beyond(float(low_ev), low, above=False)
    return burnt, crushed


def reference_params(exposure_anchor_ev: float) -> EditParams:
    """The frame as the program would develop it with no style at all.

    Neutral tone at the automatic exposure of the *Neutro automatico*
    (section 22): the baseline an edit's clipping is measured against.
    """
    from ..style.builtin import NEUTRAL_AUTO, RULES

    params = EditParams()
    params.exposure.ev = RULES[NEUTRAL_AUTO].auto_exposure * float(exposure_anchor_ev)
    return params


def added_clipping(
    tails: dict | None, params: EditParams, exposure_anchor_ev: float | None
) -> tuple[float, float] | None:
    """``(burnt, crushed)`` that ``params`` adds to the frame's own clipping.

    What the scene burns or crushes whatever the edit -- a white overcast sky,
    the unlit corners of a stage -- is not the edit's doing, and the review
    cannot give it back. Counted in full, it sent to the queue the photos
    whose own delivered JPEGs burn just as much (DSC06403: 26% of the user's
    frame white, 21% estimated for the prediction), and made the clipping
    terms point away from the errors: of the 65 pairs, the photos they
    escalated were *closer* to the user's edit than the others (measured
    in phase 7). ``None`` without tails or without an anchor.
    """
    if exposure_anchor_ev is None:
        return None
    after = clipped_after(tails, params)
    baseline = clipped_after(tails, reference_params(exposure_anchor_ev))
    if after is None or baseline is None:
        return None
    return max(0.0, after[0] - baseline[0]), max(0.0, after[1] - baseline[1])
