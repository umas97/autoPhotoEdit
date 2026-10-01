# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the user does in the review (section 9.2), written to the catalogue.

**A scene** is reviewed on its representative, the medoid of the cluster. A
correction of the representative's sliders is propagated to every other photo
of the scene as a *delta of the style vector* (``style/vector.py``), not as
absolute values: a scene whose photos differ by half a stop still does after
"a bit warmer, a bit brighter". Only the style is propagated -- geometry,
crop, masks, noise and sharpening are each photo's own. A photo the user
decided on individually is left alone.

* approving a scene approves every photo of it that is not in the individual
  queue (their doubts are their own, and they are resolved one by one);
  several scenes can be approved at once from the grid of representatives;
* rejecting a scene sends all its photos to the individual queue, with their
  edits unchanged, to be chosen among the variants there (the user's choice,
  2026-09-24).

**A photo** of the queue is approved as it is on screen, or rejected: its
style goes back to the *Neutro automatico* of section 22 and it stays in the
queue, marked to be fixed by hand (the user's choice, 2026-09-24).

Every edit is a new ``EditVersion`` (section 23). Every operation returns an
**undo** record -- which version was current and what the review said,
photo by photo -- that :func:`undo` turns back into the previous state, as new
versions: the history only grows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.base import utcnow
from ..db.models import EditVersion, EditVersionSource, Photo, PhotoStatus, Project, StyleProfile
from ..pipeline.params import EditParams
from ..style import vector as sv
from . import feedback
from .service import current_versions, editing_photos, refresh, status_for

__all__ = [
    "ReviewError",
    "apply_to_scene",
    "approve_photo",
    "approve_scene",
    "approve_scenes",
    "reject_photo",
    "reject_scene",
    "reset_photo",
    "undo",
]

#: A correction smaller than this in every entry (``sv.UNITS``) is not one.
_MIN_DELTA = 1e-3


class ReviewError(ValueError):
    """A review action that cannot be carried out, in words for the UI."""


def _context(photo: Photo) -> sv.StyleContext | None:
    context = (photo.prediction or {}).get("context")
    return sv.StyleContext(**context) if context else None


def _params(version: EditVersion | None) -> EditParams:
    return EditParams.from_dict(version.params) if version else EditParams()


def _snapshot(photos: list[Photo], versions: dict[int, EditVersion]) -> list[dict[str, Any]]:
    return [
        {
            "photo_id": p.id,
            "version_id": versions[p.id].id if p.id in versions else None,
            "review": p.review,
        }
        for p in photos
    ]


def _write(session: Session, photo: Photo, params: EditParams, source: EditVersionSource) -> None:
    from ..api.routes_photos import add_version

    add_version(session, photo, params, source)


def _members(session: Session, project: Project, cluster: int) -> list[Photo]:
    members = [p for p in editing_photos(session, project) if p.cluster_id == cluster]
    if not members:
        raise ReviewError(f"la scena {cluster} non esiste o non ha foto da rivedere")
    members.sort(key=lambda p: (p.cluster_rank is None, p.cluster_rank or 0, p.id))
    return members


def _place(session: Session, project: Project, photos: list[Photo]) -> None:
    """Rescore and re-place: the edits changed, and so may the confidence."""
    session.flush()
    refresh(session, project)
    threshold = float(project.confidence_threshold)
    for photo in photos:
        photo.status = status_for(photo, threshold)
    _maybe_done(session, project)


def _maybe_done(session: Session, project: Project) -> None:
    """Section 23.2: the last approval is a passage of the workflow, snapshotted."""
    from .. import snapshots

    session.flush()
    photos = editing_photos(session, project)
    if photos and all(p.status is PhotoStatus.APPROVED for p in photos):
        snapshots.take_auto(session, project, snapshots.AUTO_REVIEW)


def _base(
    session: Session, photo: Photo, versions: dict[int, EditVersion], base_version_id: int | None
) -> EditVersion | None:
    """The version the correction on screen started from.

    The editor saves by itself (the user's choice, 2026-09-29), so by the time
    "Applica alla scena" is pressed the representative's current version may
    already *be* the correction: the delta has to be measured from where the
    user started, which only the screen knows. Without it -- an old client, a
    script -- the current version is the start, as before.
    """
    if base_version_id is not None:
        base = session.get(EditVersion, base_version_id)
        if base is None or base.photo_id != photo.id:
            raise ReviewError("la versione di partenza non è di questa foto")
        return base
    return versions.get(photo.id)


def _undo_to(undo_record: list[dict[str, Any]], photo: Photo, base: EditVersion | None) -> None:
    """Undoing the action puts the photo back where the correction started."""
    for entry in undo_record:
        if entry["photo_id"] == photo.id:
            entry["version_id"] = base.id if base is not None else None


def apply_to_scene(
    session: Session,
    project: Project,
    cluster: int,
    params: EditParams,
    base_version_id: int | None = None,
) -> dict[str, Any]:
    """Give the representative ``params`` and the rest of the scene the same delta.

    The delta is ``params`` minus the representative's version
    ``base_version_id`` (default: its current one).
    """
    members = _members(session, project, cluster)
    versions = current_versions(session, [m.id for m in members])
    undo_record = _snapshot(members, versions)
    representative, others = members[0], members[1:]
    context = _context(representative)
    base = _base(session, representative, versions, base_version_id)
    _undo_to(undo_record, representative, base)
    before = _params(base)
    if before == params:
        return {"propagated": 0, "undo": undo_record}
    if _params(versions.get(representative.id)) != params:
        _write(session, representative, params, EditVersionSource.USER_EDITED)
    propagated = 0
    if context is not None:
        delta = sv.from_params(params, context) - sv.from_params(before, context)
        if np.max(np.abs(delta) / sv.UNITS) >= _MIN_DELTA:
            for photo in others:
                if (photo.review or {}).get("by") == "photo":
                    continue  # decided one by one: the scene does not speak for it
                own = _context(photo)
                if own is None:
                    continue
                current = _params(versions.get(photo.id))
                moved = sv.clip(sv.from_params(current, own) + delta)
                _write(
                    session,
                    photo,
                    sv.to_params(moved, own, base=current),
                    EditVersionSource.CLUSTER_APPLIED,
                )
                propagated += 1
    _place(session, project, members)
    return {"propagated": propagated, "undo": undo_record}


def approve_scene(
    session: Session,
    project: Project,
    cluster: int,
    params: EditParams | None = None,
    base_version_id: int | None = None,
) -> dict[str, Any]:
    """Approve a scene, after applying a correction of its representative if given."""
    members = _members(session, project, cluster)
    versions = current_versions(session, [m.id for m in members])
    undo_record = _snapshot(members, versions)
    representative = members[0]
    propagated = 0
    if params is not None:
        base = _base(session, representative, versions, base_version_id)
        _undo_to(undo_record, representative, base)
        propagated = apply_to_scene(session, project, cluster, params, base_version_id)[
            "propagated"
        ]
    stamp = utcnow().isoformat()
    approved = 0
    for photo in members:
        review = photo.review or {}
        if review.get("by") == "photo":
            continue
        in_queue = photo.confidence is not None and photo.confidence < float(
            project.confidence_threshold
        )
        # The representative is what the user looked at: approved whatever its
        # score. The others only if nothing about them is in doubt.
        if photo is representative or not in_queue:
            photo.review = {
                "decision": "approved",
                "by": "scene",
                "at": stamp,
                # What the user looked at: its correction is a training pair.
                "representative": photo is representative,
            }
            approved += 1
        elif review.get("queued"):
            photo.review = None  # the scene is no longer rejected
    _place(session, project, members)
    feedback.sync(session, project, representative)
    return {"approved": approved, "propagated": propagated, "undo": undo_record}


def approve_scenes(session: Session, project: Project, clusters: list[int]) -> dict[str, Any]:
    """Approve several scenes as they are: the grid's "approve the selected".

    Each scene is approved exactly as :func:`approve_scene` would, and the
    undo records are joined, so that one Ctrl+Z takes the whole choice back.
    """
    undo_record: list[dict[str, Any]] = []
    approved = 0
    for cluster in dict.fromkeys(clusters):
        result = approve_scene(session, project, cluster)
        undo_record += result["undo"]
        approved += result["approved"]
    return {"approved": approved, "scenes": len(dict.fromkeys(clusters)), "undo": undo_record}


def reject_scene(session: Session, project: Project, cluster: int) -> dict[str, Any]:
    """Send every photo of the scene to the individual queue, edits unchanged."""
    members = _members(session, project, cluster)
    undo_record = _snapshot(members, current_versions(session, [m.id for m in members]))
    queued = 0
    for photo in members:
        if (photo.review or {}).get("by") == "photo":
            continue
        photo.review = {"queued": True, "by": "scene", "at": utcnow().isoformat()}
        queued += 1
        feedback.sync(session, project, photo)
    _place(session, project, members)
    return {"queued": queued, "undo": undo_record}


def _photo(session: Session, project: Project, photo_id: int) -> Photo:
    photo = session.get(Photo, photo_id)
    if photo is None or photo.project_id != project.id:
        raise ReviewError(f"la foto {photo_id} non è in questo progetto")
    if photo.prediction is None or photo.prediction.get("profile_id") != project.style_profile_id:
        raise ReviewError("questa foto non ha ancora lo stile del progetto")
    return photo


def approve_photo(
    session: Session, project: Project, photo_id: int, params: EditParams | None = None
) -> dict[str, Any]:
    """Approve one photo as it is on screen (``params``, when the user changed it)."""
    photo = _photo(session, project, photo_id)
    versions = current_versions(session, [photo.id])
    undo_record = _snapshot([photo], versions)
    if params is not None and _params(versions.get(photo.id)) != params:
        _write(session, photo, params, EditVersionSource.USER_EDITED)
    photo.review = {"decision": "approved", "by": "photo", "at": utcnow().isoformat()}
    _place(session, project, [photo])
    feedback.sync(session, project, photo)
    return {"undo": undo_record}


def reject_photo(session: Session, project: Project, photo_id: int) -> dict[str, Any]:
    """Style back to the *Neutro automatico*; the photo stays in the queue."""
    from ..style.builtin import NEUTRAL_AUTO
    from ..style.predict import describe_photo, params_for
    from ..style.profile import ensure_builtins, load

    photo = _photo(session, project, photo_id)
    versions = current_versions(session, [photo.id])
    undo_record = _snapshot([photo], versions)
    ensure_builtins(session)
    neutral_id = session.scalars(
        select(StyleProfile.id).where(
            StyleProfile.builtin.is_(True), StyleProfile.name.startswith(NEUTRAL_AUTO)
        )
    ).first()
    current = _params(versions.get(photo.id))
    if neutral_id is not None:
        try:
            scene, _measured = describe_photo(
                photo.scene_features, photo.embedding, photo.analysis, photo.proxy_path
            )
            params = params_for(load(session, neutral_id), scene, current)
        except (LookupError, ValueError) as exc:
            raise ReviewError(f"non riesco a calcolare il Neutro automatico: {exc}") from exc
        if params != current:
            _write(session, photo, params, EditVersionSource.USER_EDITED)
    photo.review = {"decision": "rejected", "by": "photo", "at": utcnow().isoformat()}
    _place(session, project, [photo])
    feedback.sync(session, project, photo)
    return {"undo": undo_record}


def reset_photo(session: Session, project: Project, photo_id: int) -> dict[str, Any]:
    """Forget the user's decision on one photo; its edit stays as it is."""
    photo = _photo(session, project, photo_id)
    undo_record = _snapshot([photo], current_versions(session, [photo.id]))
    photo.review = None
    _place(session, project, [photo])
    feedback.sync(session, project, photo)
    return {"undo": undo_record}


def undo(session: Session, project: Project, record: list[dict[str, Any]]) -> int:
    """Put back the versions and decisions an operation's undo record names.

    A version is put back by restoring it as a *new* version (section 23), so
    that undoing an undo stays possible. Returns how many photos changed.
    """
    from ..api.routes_photos import add_version

    photos = []
    ids = [int(entry["photo_id"]) for entry in record]
    versions = current_versions(session, ids)
    for entry in record:
        photo = session.get(Photo, int(entry["photo_id"]))
        if photo is None or photo.project_id != project.id:
            continue
        target_id = entry.get("version_id")
        current = versions.get(photo.id)
        if target_id is not None and (current is None or current.id != target_id):
            target = session.get(EditVersion, int(target_id))
            if target is not None and target.photo_id == photo.id:
                add_version(
                    session,
                    photo,
                    EditParams.from_dict(target.params),
                    EditVersionSource.REVERTED,
                )
        photo.review = entry.get("review")
        photos.append(photo)
    _place(session, project, photos)
    for photo in photos:
        feedback.sync(session, project, photo)
    return len(photos)
