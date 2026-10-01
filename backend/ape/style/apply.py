# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""From per-photo predictions to the project's edits (sections 8.4 and 23).

The ``predict`` jobs store a style vector on each photo. What turns them into
``EditVersion`` rows runs here, in the server, on numbers only, and lazily --
at the first look after new predictions, and never while predictions of the
project are still queued -- for the same reason as the clustering: coherence
moves a photo towards its scene, and half a scene is the wrong target.

**The user's edits are never overwritten.** A prediction replaces the *style*
fields of a photo (white balance, exposure, tone, colour, local contrast),
and only when those fields are still what the program last put there -- the
neutral first version, or the previous prediction. Geometry, crop, denoise,
sharpening and masks always stay as they are. A photo whose style the user
touched keeps it, and is reported as such. Each write is a new version with
source ``predicted`` and the old one as parent (section 23).
"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    EditVersion,
    EditVersionSource,
    Job,
    JobKind,
    JobState,
    Photo,
    PhotoStatus,
    Project,
)
from ..db.workset import editing_set
from ..pipeline.params import EditParams
from . import vector as sv
from .coherence import CoherenceItem, regularise

__all__ = [
    "STYLE_FIELDS",
    "apply_predictions",
    "pending_predictions",
    "status_counts",
    "style_signature",
]

_log = logging.getLogger(__name__)

#: The parts of ``EditParams`` a style owns (section 8.2). Everything else is
#: the photo's own.
STYLE_FIELDS = (
    "white_balance",
    "exposure",
    "highlight_recovery",
    "tone",
    "tone_shaping",
    "color",
    "local_contrast",
)

#: A re-application that moves no parameter by more than this, in the units of
#: ``vector.UNITS``, writes nothing: new photos joining a scene nudge its
#: median, and a new version on every photo for a nudge is history noise.
_MIN_CHANGE = 0.05

_FORWARD = frozenset(
    {PhotoStatus.IMPORTED, PhotoStatus.CULLED, PhotoStatus.ANALYZED, PhotoStatus.PREDICTED}
)


def style_signature(params: EditParams) -> str:
    """A digest of the style fields: equal digests, equal style."""
    payload = {name: getattr(params, name).model_dump(mode="json") for name in STYLE_FIELDS}
    text = json.dumps(payload, sort_keys=True)
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()


_NEUTRAL_SIGNATURE: str | None = None


def _neutral_signature() -> str:
    global _NEUTRAL_SIGNATURE
    if _NEUTRAL_SIGNATURE is None:
        _NEUTRAL_SIGNATURE = style_signature(EditParams())
    return _NEUTRAL_SIGNATURE


def _editing_set(project_id: int):
    return editing_set(project_id)


def pending_predictions(session: Session, project_id: int) -> int:
    return int(
        session.scalar(
            select(func.count(Job.id)).where(
                Job.project_id == project_id,
                Job.kind.in_((JobKind.PREDICT, JobKind.ANALYZE, JobKind.PROXY)),
                Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
            )
        )
        or 0
    )


def _current(session: Session, photo_id: int) -> EditVersion | None:
    return session.scalars(
        select(EditVersion).where(
            EditVersion.photo_id == photo_id, EditVersion.is_current.is_(True)
        )
    ).first()


def apply_predictions(session: Session, project: Project, *, force: bool = False) -> dict:
    """Write the coherent predictions of the project as versions, where allowed.

    Returns counts for the interface: ``written``, ``kept_user_edit``,
    ``unchanged``, ``waiting``.
    """
    from ..analysis.service import ensure_clustered
    from ..api.routes_photos import add_version

    counts = {"written": 0, "kept_user_edit": 0, "unchanged": 0, "waiting": 0}
    profile_id = project.style_profile_id
    if profile_id is None:
        return counts
    waiting = pending_predictions(session, project.id)
    counts["waiting"] = waiting
    if waiting:
        return counts
    photos = [
        p
        for p in session.scalars(select(Photo).where(*_editing_set(project.id))).all()
        if p.prediction and p.prediction.get("profile_id") == profile_id
    ]
    lam = float(project.coherence_lambda)
    fresh = any(
        not p.prediction.get("applied") or p.prediction.get("lambda") != lam for p in photos
    )
    if not photos or not (fresh or force):
        return counts
    ensure_clustered(session, project.id)

    items = []
    for photo in photos:
        context = photo.prediction["context"]
        items.append(
            CoherenceItem(
                photo_id=photo.id,
                vector=np.asarray(photo.prediction["vector"], dtype=np.float64),
                as_shot_temperature_k=float(context["as_shot_temperature_k"]),
                as_shot_tint=float(context["as_shot_tint"]),
                cluster_id=photo.cluster_id,
                shot_at=photo.shot_at,
                focal_length=photo.focal_length,
            )
        )
    coherent = regularise(items, lam)

    for photo in photos:
        prediction = dict(photo.prediction)
        vector = coherent[photo.id]
        context = sv.StyleContext(**prediction["context"])
        current = _current(session, photo.id)
        current_params = EditParams.from_dict(current.params) if current else EditParams()
        signature = style_signature(current_params)
        # A photo the user approved or rejected in the review (section 9.2)
        # keeps what they decided on, even if it is still our prediction.
        decided = bool((photo.review or {}).get("decision"))
        ours = not decided and signature in (
            _neutral_signature(),
            prediction.get("applied_signature"),
        )
        previous = prediction.get("applied_vector")
        if not ours:
            counts["kept_user_edit"] += 1
            prediction["kept_user_edit"] = True
        elif (
            previous is not None
            and np.max(np.abs(np.asarray(previous) - vector) / sv.UNITS) < _MIN_CHANGE
            and signature == prediction.get("applied_signature")
        ):
            counts["unchanged"] += 1
        else:
            params = sv.to_params(vector, context, base=current_params)
            add_version(session, photo, params, EditVersionSource.PREDICTED)
            prediction["applied_signature"] = style_signature(params)
            prediction["applied_vector"] = [round(float(x), 5) for x in vector]
            prediction.pop("kept_user_edit", None)
            counts["written"] += 1
            if photo.status in _FORWARD:
                photo.status = PhotoStatus.PREDICTED
        prediction["applied"] = True
        prediction["lambda"] = lam
        photo.prediction = prediction
    session.flush()
    _log.info("progetto %d: stile applicato %s", project.id, counts)
    if counts["written"]:
        from .. import snapshots

        snapshots.take_auto(session, project, snapshots.AUTO_PREDICTION)
    return counts


def status_counts(session: Session, project: Project) -> dict:
    """How many photos carry the project's style, and how many kept the user's own."""
    styled = kept = 0
    if project.style_profile_id is not None:
        for (prediction,) in session.execute(
            select(Photo.prediction).where(*_editing_set(project.id), Photo.prediction.is_not(None))
        ):
            if not prediction or prediction.get("profile_id") != project.style_profile_id:
                continue
            if prediction.get("kept_user_edit"):
                kept += 1
            elif prediction.get("applied_signature"):
                styled += 1
    return {"styled": styled, "kept_user_edit": kept}
