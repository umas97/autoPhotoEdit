# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Is this frame technically usable? Focus, motion and exposure (section 7.3).

Everything here reads the camera's embedded preview (``raw/embedded.py``) and
nothing else: a culling pass that decoded RAW files would cost a second a photo
on this machine, and section 7.2 rules that out.

The measures live in ``sharpness.py`` and ``exposure.py``. This module maps them
onto the three default criteria of section 7.3, each a score in [0, 1] where 1
is good, which is the only shape the selection (``select.py``) ever sees. The
calibration of that mapping -- which measured value is "clearly out of focus",
which is "fine" -- is here, with the numbers it was fitted on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .exposure import ExposureMetrics, exposure_metrics
from .sharpness import ANALYSIS_EDGE, SharpnessMetrics, analysis_gray, sharpness_metrics

__all__ = [
    "ANALYSIS_EDGE",
    "ExposureMetrics",
    "SharpnessMetrics",
    "TechnicalScores",
    "analysis_gray",
    "exposure_metrics",
    "sharpness_metrics",
    "smoothstep",
    "technical_scores",
]


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """0 below ``edge0``, 1 above ``edge1``, a smooth ramp between. Either order."""
    if edge0 == edge1:
        return float(x >= edge0)
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class TechnicalScores:
    """The three default criteria, each in [0, 1] where 1 is good."""

    focus: float | None
    motion: float | None
    exposure: float
    #: ``over``, ``under`` or ``None``: which way the exposure is off, for the
    #: reason shown to the user.
    exposure_side: str | None
    sharpness: SharpnessMetrics
    exposure_detail: ExposureMetrics

    def features(self) -> dict:
        """The measurements behind the scores, for the catalogue."""
        return {
            "sharpness": asdict(self.sharpness),
            "exposure": asdict(self.exposure_detail),
            "exposure_side": self.exposure_side,
        }


#: Calibration of the focus score, on the largest directional ratio of the
#: sharpest region. The 24 A7 III previews of ``tests/fixtures`` all measure
#: 0.61 or more -- the lowest being macro shots with nothing but bokeh around
#: the subject, which score 0.92 -- and the same previews blurred by sigma 3 px,
#: a focus missed by a clear margin, all measure 0.41 or less and score under
#: 0.1.
FOCUS_BLURRED, FOCUS_SHARP = 0.36, 0.66

#: Motion is the weakest direction of the tiles with structure in every
#: direction, *when the strongest is still crisp*: that combination is a
#: streak. The weakest direction alone would also be low on a defocused frame,
#: which is already the focus score's business -- counting it twice would
#: give every blurred photo two reasons instead of the right one.
#:
#: Measured on the A7 III previews of ``tests/fixtures`` smeared by a straight
#: streak at 0, 30, 100 and 160 degrees: the weakest ratio is 0.55 or more on
#: every sharp preview, 0.41-0.46 at 7 px, 0.26-0.35 at 15 and 21 px -- about
#: the same at every angle, which the anisotropy (0.23-0.56 on the same
#: frames, and up to 0.54 on defocused ones) is not.
MOTION_STREAKED, MOTION_CRISP = 0.20, 0.50
#: How crisp the strongest direction must be for a weak one to mean motion.
MOTION_ONE_DIRECTION_SHARP = (0.45, 0.62)


def technical_scores(bgr: np.ndarray, gray: np.ndarray | None = None) -> TechnicalScores:
    """Focus, motion and exposure of one preview, each mapped to [0, 1].

    Args:
        bgr: the upright 8-bit BGR preview.
        gray: its :func:`analysis_gray`, when the caller already has it.
    """
    sharp = sharpness_metrics(gray if gray is not None else analysis_gray(bgr))

    small = cv2.resize(
        bgr,
        (max(1, bgr.shape[1] // 4), max(1, bgr.shape[0] // 4)),
        interpolation=cv2.INTER_AREA,
    )
    exposure = exposure_metrics(small)

    focus = motion = None
    if sharp.assessable:
        focus = smoothstep(FOCUS_BLURRED, FOCUS_SHARP, float(sharp.acuity_max))
        smeared = 1.0 - smoothstep(MOTION_STREAKED, MOTION_CRISP, float(sharp.acuity_min))
        crisp_across = smoothstep(*MOTION_ONE_DIRECTION_SHARP, float(sharp.acuity_max))
        motion = 1.0 - smeared * crisp_across

    # Past the recoverable margins of section 7.3, and only there. Highlights:
    # a large blown area -- white, or colour pushed off the top of one channel
    # over most of the frame -- in a frame that is bright overall. A backlit
    # subject under a white sky is blown in the JPEG and normal in its
    # mid-tones: the textbook case where the RAW gives the sky back. The
    # previews of ``tests/fixtures`` clip at most 6 % of their pixels in any
    # channel and 2 % in all three; the same previews pushed 3 stops clip up
    # to 53 %, and a colourful scene pushed 3.5 stops up to 93 %.
    lost = max(
        smoothstep(0.03, 0.30, exposure.blown),
        smoothstep(0.20, 0.60, exposure.clipped),
    )
    # Or, whatever is clipped, mid-tones so far above a normal rendering that
    # the two stops a RAW gives back cannot bring them home. The previews of
    # ``tests/fixtures`` reach +0.8 at most.
    over = max(lost * smoothstep(0.0, 1.5, exposure.ev), smoothstep(2.0, 3.0, exposure.ev))
    # Shadows: mid-tones more than about four stops below a normal rendering.
    under = smoothstep(2.5, 5.5, -exposure.ev)
    side = None
    if max(over, under) > 0.0:
        side = "over" if over >= under else "under"
    return TechnicalScores(
        focus=focus,
        motion=motion,
        exposure=1.0 - max(over, under),
        exposure_side=side,
        sharpness=sharp,
        exposure_detail=exposure,
    )
