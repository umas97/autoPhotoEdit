# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the user decides during culling, and how culling ends.

Two rules from section 7 are enforced here and nowhere else.

**A decision by the user is never overwritten by anything automatic** (test
10). It is stored as ``cull_decided_by = user``, which the selection treats as
fixed; handing a photo back to the automatic selection is an explicit act of
its own -- ``auto`` -- and it is what undo sends.

**Nothing is developed without the user's consent.** "Procedi con l'editing" is
the only way from culling into development (section 7.6), and it queues the
proxies of the selected photos and nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import CullDecidedBy, JobKind, Photo, Project, ProjectStatus
from .service import CullingState, _photos_query, apply_selection

__all__ = ["confirm_selection", "set_decisions"]


def set_decisions(
    session: Session, project: Project, decisions: Sequence[tuple[int, str]]
) -> CullingState:
    """Record the user's own decisions, then reselect around them.

    ``keep`` and ``discard`` freeze the photo (test 10); ``auto`` hands it back
    to the automatic selection, which is what undoing a decision means.

    A photo kept after the editing has started gets its proxy queued at once:
    section 7.5 promises that a discarded photo can come back at any moment,
    and back means usable, not merely visible.
    """
    from ..jobs.queue import enqueue

    ids = [photo_id for photo_id, _ in decisions]
    photos = {
        p.id: p
        for p in session.scalars(
            select(Photo).where(Photo.id.in_(ids), Photo.project_id == project.id)
        ).all()
    }
    started = project.status not in (
        ProjectStatus.NEW, ProjectStatus.IMPORTING, ProjectStatus.CULLING
    )
    for photo_id, decision in decisions:
        photo = photos.get(photo_id)
        if photo is None:
            raise LookupError(f"foto {photo_id} non appartiene al progetto")
        if decision == "keep":
            photo.culled, photo.cull_reasons = False, []
            photo.cull_decided_by = CullDecidedBy.USER
            if started and not photo.proxy_path:
                enqueue(
                    session, JobKind.PROXY, {"photo_id": photo.id},
                    project_id=project.id, dedupe_key=f"proxy:{photo.id}",
                )
        elif decision == "discard":
            photo.culled, photo.cull_reasons = True, ["user"]
            photo.cull_decided_by = CullDecidedBy.USER
        elif decision == "auto":
            photo.cull_decided_by = CullDecidedBy.AUTO if photo.culling_features else None
        else:
            raise ValueError(f"decisione sconosciuta: {decision}")
    session.flush()
    return apply_selection(session, project)


def confirm_selection(session: Session, project: Project) -> int:
    """"Procedi con l'editing": queue the proxies of the selected photos.

    Returns how many jobs were added. This is the one door from culling into
    development, and it is opened only by the user (section 7.6).
    """
    from ..jobs.queue import enqueue

    queued = 0
    for photo in session.scalars(_photos_query(project.id).where(Photo.culled.is_(False))):
        if photo.proxy_path:
            continue
        job = enqueue(
            session, JobKind.PROXY, {"photo_id": photo.id},
            project_id=project.id, dedupe_key=f"proxy:{photo.id}",
        )
        queued += int(job is not None)
    # Forward only: confirming again from a later phase -- after new photos
    # were imported and culled -- must not send the project back.
    if project.status in (ProjectStatus.NEW, ProjectStatus.IMPORTING, ProjectStatus.CULLING):
        project.status = ProjectStatus.ANALYZING
    return queued
