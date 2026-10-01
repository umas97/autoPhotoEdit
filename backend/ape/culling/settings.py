# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The user's culling settings, with the defaults they start from (section 7.4).

Every value here is visible and adjustable in the interface: the weights, the
criteria switched on, the aggressiveness and the threshold it sets. None of it
is hidden in the selection code, which is what lets the interface show the
formula exactly as it is applied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..db.enums import CullingMode

__all__ = [
    "CRITERIA",
    "DEFAULT_CRITERIA",
    "DEFAULT_WEIGHTS",
    "CullingSettings",
    "target_count",
    "threshold",
]

#: Every criterion of section 7.3, in the order the interface lists them.
CRITERIA = ("sharpness", "motion", "exposure", "burst", "faces", "aesthetic")

#: Section 7.3: the four light criteria on, the two that need a model off.
DEFAULT_CRITERIA = {
    "sharpness": True,
    "motion": True,
    "exposure": True,
    "burst": True,
    "faces": False,
    "aesthetic": False,
}

#: Relative weights of the scored criteria. Focus weighs most because it is
#: the one defect nothing downstream can repair; exposure next because a RAW
#: repairs most of it; motion last because, when it is bad, the focus score
#: usually drops with it and the two would otherwise count the same defect
#: twice.
DEFAULT_WEIGHTS = {
    "sharpness": 0.40,
    "motion": 0.25,
    "exposure": 0.35,
    "faces": 0.30,
    "aesthetic": 0.20,
}

_SCORED = ("sharpness", "motion", "exposure", "faces", "aesthetic")

#: The technical threshold at aggressiveness 0 and 1. A score below it is a
#: photo compromised beyond recovery. At the default of 0.5 the threshold is
#: 0.37: on the calibration set of ``technical.py`` every sharp preview scores
#: above 0.9 and every one blurred by sigma 3 px below 0.1, so the default sits
#: far from both and the slider has room to move either way.
_THRESHOLD_LENIENT = 0.12
_THRESHOLD_STRICT = 0.62


def threshold(aggressiveness: float) -> float:
    """The score under which a criterion counts as a technical failure."""
    a = min(1.0, max(0.0, float(aggressiveness)))
    return _THRESHOLD_LENIENT + (_THRESHOLD_STRICT - _THRESHOLD_LENIENT) * a


@dataclass(frozen=True)
class CullingSettings:
    mode: CullingMode = CullingMode.CONSERVATIVE
    #: Percent (0-100) in ``target_percent``, a count in ``target_count``.
    target: float | None = None
    aggressiveness: float = 0.5
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    criteria: Mapping[str, bool] = field(default_factory=lambda: dict(DEFAULT_CRITERIA))

    @classmethod
    def from_values(
        cls,
        mode: CullingMode | str | None,
        target: float | None,
        aggressiveness: float | None,
        weights: Mapping[str, float] | None,
        criteria: Mapping[str, bool] | None,
    ) -> CullingSettings:
        """Fill in what a project row leaves empty, and drop what is unknown."""
        merged_weights = dict(DEFAULT_WEIGHTS)
        for key, value in (weights or {}).items():
            if key in merged_weights and value is not None:
                merged_weights[key] = max(0.0, float(value))
        merged_criteria = dict(DEFAULT_CRITERIA)
        for key, value in (criteria or {}).items():
            if key in merged_criteria:
                merged_criteria[key] = bool(value)
        return cls(
            mode=CullingMode(mode or CullingMode.CONSERVATIVE),
            target=None if target is None else float(target),
            aggressiveness=0.5 if aggressiveness is None else float(aggressiveness),
            weights=merged_weights,
            criteria=merged_criteria,
        )


def target_count(settings: CullingSettings, population: int) -> int | None:
    """How many photos a target mode asks for, out of ``population``."""
    if settings.target is None:
        return None
    if settings.mode is CullingMode.TARGET_PERCENT:
        return max(0, math.ceil(population * min(100.0, max(0.0, settings.target)) / 100.0))
    if settings.mode is CullingMode.TARGET_COUNT:
        return max(0, int(settings.target))
    return None
