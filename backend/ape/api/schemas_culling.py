# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The shapes of the culling API (section 7).

Kept apart from ``schemas.py`` for the four-hundred-line limit of section 26,
and because culling has two very different traffic patterns: one large
document when the screen opens -- every photo, every group -- and many small
ones while the user works -- only what changed. The second kind has to stay
small, because it answers a slider being dragged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..db.models import CullingMode, MergeDecision, MergeKind, PhotoStatus

__all__ = [
    "CullDecisionIn",
    "CullDecisionsRequest",
    "CullPhoto",
    "CullSelection",
    "CullSettings",
    "CullSettingsUpdate",
    "CullStartRequest",
    "CullSummary",
    "CullView",
    "FeatureAvailability",
]


class CullSettings(BaseModel):
    mode: CullingMode
    target: float | None
    aggressiveness: float
    weights: dict[str, float]
    criteria: dict[str, bool]
    #: The score under which a criterion is a technical failure, at this
    #: aggressiveness. Shown so that the formula is visible (section 7.4).
    threshold: float


class CullSettingsUpdate(BaseModel):
    """Every field optional: what is absent stays as it is."""

    mode: CullingMode | None = None
    target: float | None = Field(default=None, ge=0)
    aggressiveness: float | None = Field(default=None, ge=0.0, le=1.0)
    weights: dict[str, float] | None = None
    criteria: dict[str, bool] | None = None


class FeatureAvailability(BaseModel):
    available: bool
    #: A code from ``models_registry.Unavailable``; the interface words it.
    reason: str | None = None
    licence: str | None = None
    notice: str | None = None
    size_mb: float | None = None


class CullSummary(BaseModel):
    total: int
    selected: int
    culled: int
    analysed: int
    pending: int
    failed: int
    #: In a target mode: how many photos the target asks for.
    target_count: int | None = None


class CullDecision(BaseModel):
    """The outcome for one photo. What a slider move sends back, per photo."""

    id: int
    culled: bool
    reasons: list[str]
    score: float | None
    rank: int
    #: Each active criterion's score, the inputs of ``score``.
    criteria: dict[str, float]
    decided_by: str | None


class CullPhoto(CullDecision):
    filename: str
    shot_at: datetime | None
    status: PhotoStatus
    missing: bool
    error: str | None
    burst_group_id: int | None
    merge_group_id: int | None
    #: Where to zoom in the comparison view: the camera's focus point when it
    #: recorded one, the centre of the sharpest region otherwise.
    focus_point: tuple[float, float] | None
    #: Aspect of the preview, so the grid can reserve the right box.
    width: int | None
    height: int | None
    iso: int | None
    aperture: float | None
    shutter: float | None
    focal_length: float | None
    #: Stored scores, before the burst comparison. ``motion`` is 1 for no blur.
    sharpness: float | None
    motion: float | None
    exposure: float | None
    exposure_side: str | None


class CullMerge(BaseModel):
    id: int
    kind: MergeKind
    decision: MergeDecision
    confidence: float | None
    reasons: dict
    members: list[int]
    reference: int | None


class CullSelection(BaseModel):
    summary: CullSummary
    settings: CullSettings
    decisions: list[CullDecision]


class CullView(BaseModel):
    summary: CullSummary
    settings: CullSettings
    availability: dict[str, FeatureAvailability]
    photos: list[CullPhoto]
    #: Burst id to its members, proposed frame first.
    bursts: dict[int, list[int]]
    merges: list[CullMerge]
    #: The editing has been started: kept photos are being developed.
    confirmed: bool


class CullStartRequest(BaseModel):
    #: Analyse again photos that already have a current analysis.
    force: bool = False


class CullDecisionIn(BaseModel):
    photo_id: int
    decision: Literal["keep", "discard", "auto"]


class CullDecisionsRequest(BaseModel):
    decisions: list[CullDecisionIn] = Field(min_length=1, max_length=5000)
