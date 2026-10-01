# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How crooked is the camera? The automatic straightening of section 6.4.

Runs on the browsing proxy -- already lens-corrected, so straight things are
straight in it -- and answers with one angle and how sure it is. The rule the
acceptance of phase 5 sets is asymmetric, and so is this module: **never make
a straight horizon crooked.** Missing a tilt costs the user one slider; adding
one costs their trust. Everything below errs on the side of not rotating.

The method, in the order it runs:

1. **Segments.** OpenCV's LSD on a 1024 px grey image. Short segments are
   texture (grass, bark, snow), not structure, and are dropped.
2. **Two families.** A segment within 10 degrees of horizontal votes for a
   horizon, within 10 degrees of vertical for an upright. The deviation of each
   from its axis, counter-clockwise positive, is the tilt it suggests.
3. **Perspective** (``lines.py``). Verticals converge when the camera looks up:
   those left of centre lean one way, those right of it the other, and neither
   says anything about the roll. Each family is fitted as a roll plus a
   convergence towards a vanishing point, and only the roll is kept, with its
   standard error -- which grows when the two cannot be told apart.
4. **Coherence.** What fraction of the family's total length agrees with the
   fit to within half a degree. Real structure agrees; a forest of slightly
   leaning trunks, or snow drifts, does not. Below a threshold the family
   abstains.
5. **Horizon** (``horizon.py``). One long, continuous, straight line that
   separates two regions that look different.
6. **Decision.** Verticals, a horizon, or parallel horizontals may vote;
   converging near-horizontals -- the ground in perspective -- may not. Two
   voters that disagree, or uncertain verticals that flatly contradict a
   horizontal cue, are "linee contraddittorie": no rotation. The rotation is
   applied only if it is at least twice its own standard error and lies
   between 0.15 and 8 degrees; smaller is already straight, larger is more
   likely intentional (or a misreading) than a tilt.

Every constant carries in its comment the measurement that chose it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .horizon import find_horizon
from .lines import FAMILY_WINDOW_DEG, FamilyFit, fit_rolled
from .lines import segments as find_segments

__all__ = [
    "MAX_ROTATION_DEG",
    "MIN_ROTATION_DEG",
    "StraightenResult",
    "estimate",
    "estimate_from_image",
    "estimate_from_segments",
]

#: Long edge the detector works at. LSD's cost grows with the pixels and the
#: angles it measures do not improve beyond this: a 30% long segment at 1024 px
#: is 300 px, which pins its angle to about a tenth of a degree.
_WORK_EDGE = 1024

#: Segments shorter than this fraction of the long edge are dropped. At 4% --
#: 40 px here -- bark, grass and the edges of snow clumps mostly fall below,
#: while window frames, poles and a horizon broken by trees mostly stay above.
_MIN_LENGTH = 0.04

#: Convergence below which a family of near-horizontals counts as parallel:
#: one step of the grid, a vanishing point twenty frames away.
_PARALLEL = 0.05

#: Two confident families further apart than this contradict each other.
_CONTRADICTION_DEG = 0.6

#: The window of section 6.4 within which a rotation is applied.
MIN_ROTATION_DEG = 0.15
MAX_ROTATION_DEG = 8.0


@dataclass(slots=True)
class StraightenResult:
    """What the analysis stores, and why."""

    #: The rotation to apply, as ``GeometryParams.rotation_deg``: 0 when the
    #: detector abstains or the frame is already level.
    rotation_deg: float
    #: The tilt the lines suggest, applied or not. ``None`` when nothing
    #: trustworthy was found.
    measured_deg: float | None
    #: 0..1, for the geometric term of the confidence score (section 9.1).
    confidence: float
    #: ``level`` | ``rotated`` | ``no_lines`` | ``contradictory`` | ``too_large``
    outcome: str
    families: dict[str, FamilyFit] = field(default_factory=dict)

    def as_json(self) -> dict:
        return {
            "rotation_deg": round(self.rotation_deg, 3),
            "measured_deg": None if self.measured_deg is None else round(self.measured_deg, 3),
            "confidence": round(self.confidence, 3),
            "outcome": self.outcome,
            "families": {
                name: {
                    "roll_deg": round(fit.roll_deg, 3),
                    "coherence": round(fit.coherence, 3),
                    "support": round(fit.support, 3),
                    "segments": fit.segments,
                    "convergence": round(fit.convergence, 3),
                    "roll_sigma": round(min(fit.roll_sigma, 99.0), 3),
                }
                for name, fit in self.families.items()
            },
        }


def estimate_from_segments(
    segments: np.ndarray, width: int, height: int, image: np.ndarray | None = None
) -> StraightenResult:
    """The decision of the module docstring, on segments already found.

    ``image`` is the work image the segments came from; without it a horizon
    cannot be checked for the two different worlds it must separate, and so
    cannot decide.
    """
    long_edge = float(max(width, height))
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    length = np.hypot(dx, dy)
    keep = length >= _MIN_LENGTH * long_edge
    segments, dx, dy, length = segments[keep], dx[keep], dy[keep], length[keep]

    # Angle in mathematical orientation (y up), folded into [0, 180).
    angle = np.degrees(np.arctan2(-dy, dx)) % 180.0
    mid_x = ((segments[:, 0] + segments[:, 2]) / 2 - width / 2) / long_edge
    mid_y = ((segments[:, 1] + segments[:, 3]) / 2 - height / 2) / long_edge

    horizontal_dev = np.where(angle > 90.0, angle - 180.0, angle)
    vertical_dev = angle - 90.0
    families: dict[str, FamilyFit] = {}
    is_h = np.abs(horizontal_dev) <= FAMILY_WINDOW_DEG
    is_v = np.abs(vertical_dev) <= FAMILY_WINDOW_DEG
    # y up from here on, as in the angles. A vertical converging upwards has
    # its vanishing point at positive ``along``; a horizontal converging to the
    # right, at positive ``along`` too, and its offset has the opposite sign.
    up = -mid_y
    fit_h = fit_rolled(-horizontal_dev[is_h], up[is_h], mid_x[is_h], length[is_h], long_edge)
    if fit_h is not None:
        fit_h.roll_deg = -fit_h.roll_deg
        families["horizontal"] = fit_h
        horizon = find_horizon(
            segments[is_h], horizontal_dev[is_h], length[is_h], width, long_edge, image
        )
        if horizon is not None:
            families["horizon"] = horizon
    fit_v = fit_rolled(vertical_dev[is_v], mid_x[is_v], up[is_v], length[is_v], long_edge)
    if fit_v is not None:
        families["vertical"] = fit_v

    # Who may decide. Verticals, when coherent: gravity makes them, and the
    # perspective model accounts for a camera pointed up or down. A horizon
    # line, always. The horizontal family as a whole only if its lines are
    # parallel -- a frontal facade -- because converging near-horizontals are
    # the ground plane in perspective (shores, paths, snow edges), whose angles
    # say where the camera points, not how it is rolled.
    voters: list[FamilyFit] = []
    if fit_v is not None and fit_v.confident:
        voters.append(fit_v)
    if "horizon" in families:
        voters.append(families["horizon"])
    elif fit_h is not None and fit_h.confident and abs(fit_h.convergence) <= _PARALLEL:
        voters.append(fit_h)

    if not voters:
        return StraightenResult(0.0, None, 0.2, "no_lines", families)
    if len(voters) == 2 and abs(voters[0].roll_deg - voters[1].roll_deg) > _CONTRADICTION_DEG:
        return StraightenResult(0.0, None, 0.15, "contradictory", families)
    # Verticals too uncertain to decide can still veto: when they say, with
    # whatever confidence they have, that the horizontal cue is off by more
    # than both errors allow, trusting the horizontal cue is a gamble.
    if fit_v is not None and fit_v not in voters and math.isfinite(fit_v.roll_sigma):
        for voter in voters:
            gap = abs(voter.roll_deg - fit_v.roll_deg)
            allowed = 2.0 * math.hypot(voter.roll_sigma, fit_v.roll_sigma) + _CONTRADICTION_DEG
            if fit_v.coherence >= 0.4 and gap > allowed:
                return StraightenResult(0.0, None, 0.15, "contradictory", families)

    # Inverse-variance average: the better-determined family counts more.
    weights = np.array([1.0 / max(fit.roll_sigma, 0.02) ** 2 for fit in voters])
    measured = float(np.average([fit.roll_deg for fit in voters], weights=weights))
    sigma = float(1.0 / math.sqrt(weights.sum()))
    best = max(voters, key=lambda fit: fit.coherence)
    # Confidence grows with coherence, falls with the uncertainty, and gains
    # from a second family agreeing.
    confidence = min(
        1.0, 0.45 + 0.5 * best.coherence + 0.1 * (len(voters) - 1) - 0.5 * sigma
    )
    # A correction is only an improvement if it is larger than its own error:
    # levelling a 0.3 degree tilt with a 0.25 degree uncertainty is as likely
    # to add a tilt as to remove one. Twice the error is the margin.
    if abs(measured) < max(MIN_ROTATION_DEG, 2.0 * sigma):
        return StraightenResult(0.0, measured, confidence, "level", families)
    if abs(measured) > MAX_ROTATION_DEG:
        return StraightenResult(0.0, measured, min(confidence, 0.35), "too_large", families)
    # The lines lean by ``measured``: turning the picture the other way levels it.
    return StraightenResult(-measured, measured, confidence, "rotated", families)


def estimate_from_image(image: np.ndarray) -> StraightenResult:
    """Estimate on an upright, lens-corrected image (uint8 or float, RGB or grey)."""
    if image.ndim == 3:
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        grey = image if image.dtype == np.uint8 else np.clip(image * 255 + 0.5, 0, 255).astype(
            np.uint8
        )
    height, width = grey.shape
    scale = _WORK_EDGE / max(height, width)
    if scale < 1.0:
        grey = cv2.resize(
            grey, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    height, width = grey.shape
    work = image if image.ndim == 3 else grey
    if work.shape[:2] != grey.shape:
        work = cv2.resize(work, (width, height), interpolation=cv2.INTER_AREA)
    return estimate_from_segments(find_segments(grey), width, height, work)


estimate = estimate_from_image


