# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""When detection runs, and in which process (section 25.2).

Detection is lazy, like the grouping of culling: it runs at the first look
after the photos it needs have been measured, never while measurements are
still arriving -- a bracketing seen half-analysed would be recorded, for good,
as a shorter one. The mark is the same as culling's: a photo whose features
were written since the last look has no ``burst_rank``.

* In a project that is being culled, culling's own grouping runs detection
  first and then the bursts (``culling/service.py``).
* In a project that went straight to editing, :func:`ensure_detected` does the
  detection alone, when the merges screen or the project overview is opened.

Either way, what needs no pixels -- bracketings, focus stacks -- runs here, in
the server, and the panorama search is queued for a worker.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import Job, JobKind, JobState, Photo, PhotoKind, Project
from .proposals import propose_in_server, protected_ids

__all__ = ["ensure_detected", "order_of", "queue_panorama_search"]

_log = logging.getLogger(__name__)

#: Jobs that write what detection reads.
_FEEDERS = (JobKind.PROXY, JobKind.CULL, JobKind.DETECT_MERGES)


def order_of(photo: Photo) -> tuple:
    """The order two frames of the same second were taken in."""
    camera = (photo.culling_features or {}).get("camera") or {}
    shot_number = camera.get("shot_number")
    return (shot_number if shot_number is not None else -1, photo.filename)


def queue_panorama_search(session: Session, project: Project) -> None:
    from ..jobs.queue import enqueue

    enqueue(
        session, JobKind.DETECT_MERGES, {"project_id": project.id}, project_id=project.id,
        dedupe_key=f"detect_merges:{project.id}",
    )


def _pending(session: Session, project_id: int) -> int:
    return int(
        session.scalar(
            select(func.count(Job.id)).where(
                Job.project_id == project_id,
                Job.kind.in_(_FEEDERS),
                Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
            )
        )
        or 0
    )


def ensure_detected(session: Session, project: Project) -> bool:
    """Run detection if photos were measured since the last time. Returns whether it ran."""
    if not project.merge_detection_enabled:
        return False
    if project.culling_enabled:
        from ..culling.service import ensure_grouped

        return ensure_grouped(session, project)
    if _pending(session, project.id):
        return False
    base = (
        Photo.project_id == project.id,
        Photo.kind == PhotoKind.RAW,
        Photo.missing.is_(False),
        Photo.culling_features.is_not(None),
    )
    stale = session.scalar(select(func.count(Photo.id)).where(*base, Photo.burst_rank.is_(None)))
    if not stale:
        return False
    photos = session.scalars(select(Photo).where(*base)).all()
    protected = protected_ids(session, project.id)
    created = propose_in_server(session, project, photos, protected, order=order_of)
    if created:
        _log.info("progetto %d: %d fusioni proposte", project.id, created)
    for photo in photos:
        # No bursts without culling; the rank is only the "looked at" mark.
        photo.burst_rank = 0
    queue_panorama_search(session, project)
    session.flush()
    return True
