# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The review of section 9.2, as the catalogue sees it.

Everything here runs in the server on numbers already stored: the predictions
and their coherence (``style/apply.py``), the scenes (``analysis/service.py``),
and the pixel measurements the ``predict`` job took (``review/measure.py``).
It is lazy like the clustering -- :func:`refresh` runs at the first look after
something changed and is a no-op otherwise -- because the confidence depends
on the current edit, and the edit changes under the user's hand.

**Who decides.** A photo's ``status`` is derived, not set: approved if the
user approved it, in the individual queue if the user rejected it or its scene,
or if its confidence is under the project's threshold, "predicted" otherwise.
The user's part lives in ``Photo.review`` and is never overwritten by a
recomputation -- moving the threshold moves photos in and out of the queue,
never out of "approved".
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import EditVersion, Photo, PhotoStatus, Project, StyleProfile
from ..pipeline.params import EditParams
from ..style import apply as style_apply
from ..style.predict import AUTO_VERSION
from . import developed, tonal
from .confidence import ConfidenceInputs, score, weights_from
from .reference import sample_clipping, style_clipping

__all__ = [
    "current_versions",
    "editing_photos",
    "inputs_for",
    "overview",
    "refresh",
    "status_for",
]

_log = logging.getLogger(__name__)

#: Statuses the review never moves a photo out of: the export's, and a failure.
_FROZEN = frozenset({PhotoStatus.EXPORTED, PhotoStatus.FAILED})

#: Bumped when the confidence changes meaning: stored scores are recomputed.
#: 2: clipping relative to the style of the nearest samples.
#: 3: clipping added by the edit to the frame's own (``tonal.added_clipping``).
CONFIDENCE_VERSION = 3


def editing_photos(session: Session, project: Project) -> list[Photo]:
    """The photos under review: in the editing flow, predicted with the project's style."""
    if project.style_profile_id is None:
        return []
    photos = session.scalars(
        select(Photo)
        .where(*style_apply._editing_set(project.id))
        .order_by(Photo.shot_at.is_(None), Photo.shot_at, Photo.filename, Photo.id)
    ).all()
    return [
        p
        for p in photos
        if p.prediction
        and p.prediction.get("profile_id") == project.style_profile_id
        and p.prediction.get("applied")
    ]


def current_versions(session: Session, photo_ids: list[int]) -> dict[int, EditVersion]:
    if not photo_ids:
        return {}
    rows = session.scalars(
        select(EditVersion).where(
            EditVersion.photo_id.in_(photo_ids), EditVersion.is_current.is_(True)
        )
    ).all()
    return {row.photo_id: row for row in rows}


def _automatic_rotation(photo: Photo) -> float:
    return float(((photo.analysis or {}).get("straighten") or {}).get("rotation_deg") or 0.0)


def inputs_for(
    photo: Photo,
    params: EditParams,
    calibration,
    clipping: dict[int, tuple[float, float]] | None = None,
) -> ConfidenceInputs:
    """What the score of section 9.1 looks at, for one photo and one edit.

    ``clipping`` is ``reference.sample_clipping`` of a learned profile.
    """
    prediction = photo.prediction or {}
    analysis = photo.analysis or {}
    measured = analysis.get("auto") or {}
    anchor = (prediction.get("context") or {}).get("exposure_anchor_ev")
    clipped = tonal.added_clipping(measured.get("tails"), params, anchor)
    inputs = ConfidenceInputs(
        wb_spread_mired=measured.get("wb_spread_mired"),
        burnt=None if clipped is None else clipped[0],
        crushed=None if clipped is None else clipped[1],
        straighten=analysis.get("straighten"),
        rotation_by_user=abs(params.geometry.rotation_deg - _automatic_rotation(photo)) > 0.01,
        lens_profile=None if "lens" not in analysis else analysis["lens"] is not None,
    )
    if calibration is not None and prediction.get("method") == "model":
        inputs.nearest = prediction.get("nearest")
        inputs.dispersion = prediction.get("dispersion")
        inputs.exposure_anchor_ev = anchor
        inputs.nearest_p90 = calibration.nearest_p90
        inputs.dispersion_p90 = calibration.dispersion_p90
        inputs.anchor_range = (calibration.anchor_min, calibration.anchor_max)
        style = style_clipping(prediction.get("neighbours") or [], clipping or {})
        if style is not None:
            inputs.burnt_style, inputs.crushed_style = style
    return inputs


def status_for(photo: Photo, threshold: float) -> PhotoStatus:
    """Where a reviewed photo belongs, from the user's decision and its score."""
    if photo.status in _FROZEN:
        return photo.status
    review = photo.review or {}
    if review.get("decision") == "approved":
        return PhotoStatus.APPROVED
    if review.get("decision") == "rejected" or review.get("queued"):
        return PhotoStatus.NEEDS_REVIEW
    if photo.confidence is not None and photo.confidence < threshold:
        return PhotoStatus.NEEDS_REVIEW
    return PhotoStatus.PREDICTED


def _key(version_id: int | None, profile: StyleProfile, weights: dict) -> str:
    text = json.dumps(
        [CONFIDENCE_VERSION, version_id, str(profile.updated_at), weights], sort_keys=True
    )
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def refresh(session: Session, project: Project) -> dict[str, int]:
    """Apply finished predictions, score what changed, place every photo.

    Returns counts for the log and the tests: ``scored``, ``remeasure``
    (photos predicted before phase 7, sent back to the ``predict`` job for
    the measurements the score needs), ``developing`` (developed renders not
    yet on disk, ``review/developed.py``).
    """
    from ..analysis.service import ensure_clustered
    from ..jobs.handlers_style import enqueue_prediction
    from ..style.profile import load

    counts = {"scored": 0, "remeasure": 0, "developing": 0}
    if project.style_profile_id is None:
        return counts
    style_apply.apply_predictions(session, project)
    if style_apply.pending_predictions(session, project.id):
        return counts
    ensure_clustered(session, project.id)
    profile_row = session.get(StyleProfile, project.style_profile_id)
    if profile_row is None:
        return counts
    try:
        calibration = load(session, profile_row.id).calibration
    except LookupError:
        return counts
    weights = weights_from(project.confidence_weights)
    clipping = sample_clipping(session, profile_row) if calibration is not None else {}
    photos = editing_photos(session, project)
    versions = current_versions(session, [p.id for p in photos])
    threshold = float(project.confidence_threshold)
    for photo in photos:
        prediction = dict(photo.prediction)
        measured = (photo.analysis or {}).get("auto") or {}
        # Asked once per version: a prediction that fails (a proxy gone from
        # the cache) must not be queued again at every look.
        if measured.get("version") != AUTO_VERSION and prediction.get("remeasure") != AUTO_VERSION:
            counts["remeasure"] += int(enqueue_prediction(session, photo, project.style_profile_id))
            prediction["remeasure"] = AUTO_VERSION
            photo.prediction = prediction
        version = versions.get(photo.id)
        key = _key(version.id if version else None, profile_row, weights)
        if prediction.get("confidence_key") != key:
            params = EditParams.from_dict(version.params) if version else EditParams()
            result = score(inputs_for(photo, params, calibration, clipping), weights)
            photo.confidence = round(result.value, 4)
            photo.escalation_reasons = [t.as_json() for t in result.reasons()]
            prediction["confidence_key"] = key
            photo.prediction = prediction
            counts["scored"] += 1
        status = status_for(photo, threshold)
        if photo.status != status:
            photo.status = status
    counts["developing"] = developed.enqueue_missing(session, photos, versions)
    if photos and project.status.value in ("new", "analyzing"):
        from ..db.models import ProjectStatus

        project.status = ProjectStatus.REVIEWING
    session.flush()
    if counts["scored"]:
        _log.info("progetto %d: confidenza %s", project.id, counts)
    return counts


def _reasons(photo: Photo) -> list[str]:
    codes = [r["code"] for r in (photo.escalation_reasons or [])]
    review = photo.review or {}
    if review.get("decision") == "rejected":
        codes.insert(0, "rejected")
    elif review.get("queued"):
        codes.insert(0, "scene_rejected")
    return codes


def _scene_state(members: list[Photo]) -> str:
    """``approved`` when every photo outside the queue is; ``rejected`` when the
    user sent the scene to the queue; ``pending`` otherwise."""
    reviews = [m.review or {} for m in members]
    if any(r.get("queued") and r.get("by") == "scene" for r in reviews):
        return "rejected"
    outside = [m for m in members if m.status is not PhotoStatus.NEEDS_REVIEW]
    if outside and all(m.status is PhotoStatus.APPROVED for m in outside):
        return "approved"
    if not outside and all(m.status is PhotoStatus.APPROVED for m in members):
        return "approved"
    return "pending"


def overview(session: Session, project: Project) -> dict[str, Any]:
    """The review screen: scenes, queue, counts, settings. Refreshes first."""
    from ..api.photo_out import proxy_on_disk, proxy_rev
    from .confidence import DEFAULT_WEIGHTS
    from .feedback import proposed_count

    refreshed = refresh(session, project)
    photos = editing_photos(session, project)
    versions = current_versions(session, [p.id for p in photos])
    profile = (
        session.get(StyleProfile, project.style_profile_id) if project.style_profile_id else None
    )
    scenes: dict[int, list[Photo]] = {}
    for photo in photos:
        if photo.cluster_id is not None:
            scenes.setdefault(photo.cluster_id, []).append(photo)
    scene_rows = []
    for cluster, members in sorted(scenes.items()):
        members.sort(key=lambda p: (p.cluster_rank is None, p.cluster_rank or 0))
        scene_rows.append(
            {
                "cluster": cluster,
                "representative": members[0].id,
                "photos": [m.id for m in members],
                "state": _scene_state(members),
                "queued": sum(m.status is PhotoStatus.NEEDS_REVIEW for m in members),
                "approved": sum(m.status is PhotoStatus.APPROVED for m in members),
                "min_confidence": min(
                    (m.confidence for m in members if m.confidence is not None), default=None
                ),
            }
        )
    queue = sorted(
        (p for p in photos if p.status is PhotoStatus.NEEDS_REVIEW),
        key=lambda p: (p.confidence if p.confidence is not None else 1.0, p.id),
    )
    return {
        "profile": None
        if profile is None
        else {"id": profile.id, "name": profile.name, "builtin": bool(profile.builtin)},
        "pending": style_apply.pending_predictions(session, project.id),
        "developing": refreshed["developing"],
        "threshold": float(project.confidence_threshold),
        "weights": weights_from(project.confidence_weights),
        "default_weights": DEFAULT_WEIGHTS,
        "counts": {
            "total": len(photos),
            "approved": sum(p.status is PhotoStatus.APPROVED for p in photos),
            "queue": len(queue),
            "scenes": len(scene_rows),
            "scenes_done": sum(s["state"] != "pending" for s in scene_rows),
        },
        "scenes": scene_rows,
        "queue": [
            {
                "photo_id": p.id,
                "confidence": p.confidence,
                "reasons": _reasons(p),
                "cluster": p.cluster_id,
            }
            for p in queue
        ],
        "photos": [
            {
                "id": p.id,
                "filename": p.filename,
                "status": p.status.value,
                "confidence": p.confidence,
                "cluster_id": p.cluster_id,
                "has_proxy": proxy_on_disk(p.proxy_path),
                "proxy_rev": proxy_rev(p.proxy_path),
                "developed": developed.existing_version(p.id, versions.get(p.id)),
            }
            for p in photos
        ],
        "feedback": {"proposed": proposed_count(session, project)},
    }
