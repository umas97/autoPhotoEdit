# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Problems panel (section 19): what failed, in words, and "Riprova".

Two kinds of problem, because the catalogue has two kinds of failure:

* a **photo** in state ``failed`` -- its file could not be read, or decoded, or
  exported. The reason shown is the photo's own message (written for this
  audience by the handler); the technical details are the traceback of the job
  that failed on it, when there is one;
* a **job** that failed on something that is not a failed photo -- a training
  pair, a segmentation, a developed thumbnail. Shown with its stage and its
  error, retried as a job;
* a **merge** in state ``failed`` (section 25.6) -- frames that do not align, a
  panorama that does not stitch. Not a job failure: the job ended normally and
  recorded the reason on the group. "Riprova" queues the merge again; changing
  its parameters first happens on the merges screen.

A failure never stops the batch (the queue goes on with the next job), so this
is the one place where the failures meet. Paths in the details have the home
directory replaced by ``~``: the details are there to be copied into an issue.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .db.enums import JobKind, JobState, PhotoStatus
from .db.models import Job, MergeDecision, MergeGroup, Photo, Project
from .jobs import queue as jobq

__all__ = ["STAGES", "anonymise", "count", "overview", "plain", "retry"]

#: Where in the program a job kind is, in the words of the interface.
STAGES: dict[JobKind, str] = {
    JobKind.PROXY: "anteprima",
    JobKind.CULL: "analisi di cernita",
    JobKind.ANALYZE: "analisi",
    JobKind.PREDICT: "stile",
    JobKind.STYLE_PAIR: "coppia di addestramento",
    JobKind.RENDER_PREVIEW: "miniatura sviluppata",
    JobKind.EXPORT: "export",
    JobKind.SEGMENT: "segmentazione",
}

#: Exception classes whose message is already a sentence for the user: the
#: class name in front of it is for the log, not for the panel.
_SENTENCES = ("ValueError", "LookupError", "RuntimeError", "KeyError", "OSError")

#: Failures whose message says nothing to a person, and what to say instead.
_KNOWN = {
    "MemoryError": "memoria esaurita durante l'elaborazione",
    "FileNotFoundError": "file non trovato",
    "PermissionError": "permesso negato",
    "IsADirectoryError": "al posto del file c'è una cartella",
}

_PREFIX = re.compile(r"^(\w+(?:Error|Exception))(?:: ?(.*))?$", re.S)
#: LibRaw's messages arrive as Python bytes, ``b'...'``: the quotes are Python's.
_BYTES = re.compile(r"\bb'([^']*)'")


def anonymise(text: str | None) -> str | None:
    """The home directory as ``~``: a traceback names the user's folders."""
    if text is None:
        return None
    return text.replace(str(Path.home()), "~")


def plain(message: str | None) -> str:
    """A job's or a photo's error, in the words of the panel."""
    if not message:
        return "errore sconosciuto"
    match = _PREFIX.match(message.strip())
    if not match:
        return anonymise(message.strip()) or ""
    kind, rest = match.groups()
    rest = rest or ""
    if kind in _KNOWN:
        rest = f"{_KNOWN[kind]}: {rest}" if rest else _KNOWN[kind]
    elif kind not in _SENTENCES:
        return anonymise(message.strip()) or ""
    return anonymise(_BYTES.sub(r"\1", rest.strip())) or "errore sconosciuto"


def _photo_of(job: Job) -> int | None:
    value = (job.payload or {}).get("photo_id")
    return int(value) if isinstance(value, int) else None


def _failed_jobs(session: Session, project_id: int | None) -> list[Job]:
    statement = select(Job).where(Job.state == JobState.FAILED).order_by(Job.id.desc())
    if project_id is not None:
        statement = statement.where(Job.project_id == project_id)
    return list(session.scalars(statement).all())


def _failed_merges(session: Session, project_id: int | None) -> list[MergeGroup]:
    statement = select(MergeGroup).where(MergeGroup.decision == MergeDecision.FAILED)
    if project_id is not None:
        statement = statement.where(MergeGroup.project_id == project_id)
    return list(session.scalars(statement.order_by(MergeGroup.id.desc())).all())


def count(session: Session, project_id: int | None = None) -> int:
    """How many problems the panel lists: for the badge that is always visible."""
    photos = select(Photo.id).where(Photo.status == PhotoStatus.FAILED)
    jobs = select(func.count(Job.id)).where(Job.state == JobState.FAILED)
    if project_id is not None:
        photos = photos.where(Photo.project_id == project_id)
        jobs = jobs.where(Job.project_id == project_id)
    photo_id = func.json_extract(Job.payload, "$.photo_id")
    # A failed job on a failed photo is that photo's problem, counted once.
    jobs = jobs.where(or_(photo_id.is_(None), photo_id.not_in(photos)))
    failed_photos = session.scalar(select(func.count()).select_from(photos.subquery()))
    merges = select(func.count(MergeGroup.id)).where(MergeGroup.decision == MergeDecision.FAILED)
    if project_id is not None:
        merges = merges.where(MergeGroup.project_id == project_id)
    return (
        int(failed_photos or 0) + int(session.scalar(jobs) or 0)
        + int(session.scalar(merges) or 0)
    )


def overview(session: Session, project_id: int | None = None) -> dict[str, Any]:
    """Every problem, newest first: ``photos``, ``jobs`` and their ``count``."""
    statement = select(Photo).where(Photo.status == PhotoStatus.FAILED).order_by(Photo.id)
    if project_id is not None:
        statement = statement.where(Photo.project_id == project_id)
    photos = session.scalars(statement).all()
    names = dict(session.execute(select(Project.id, Project.name)).all())
    jobs = _failed_jobs(session, project_id)
    by_photo: dict[int, Job] = {}
    for job in jobs:  # newest first: the first one seen is the last failure
        photo_id = _photo_of(job)
        if photo_id is not None:
            by_photo.setdefault(photo_id, job)

    failed_ids = {p.id for p in photos}
    photo_items = []
    for photo in photos:
        job = by_photo.get(photo.id)
        photo_items.append(
            {
                "photo_id": photo.id,
                "project_id": photo.project_id,
                "project": names.get(photo.project_id),
                "filename": photo.filename,
                "reason": plain(photo.error or (job.error if job else None)),
                "stage": STAGES.get(job.kind) if job else None,
                "job_id": job.id if job else None,
                "details": anonymise((job.traceback or job.error) if job else photo.error),
                "at": (job.finished_at.isoformat() if job and job.finished_at else None),
            }
        )
    job_items = [
        {
            "job_id": job.id,
            "project_id": job.project_id,
            "project": names.get(job.project_id) if job.project_id else None,
            "photo_id": _photo_of(job),
            "stage": STAGES.get(job.kind, job.kind.value),
            "reason": plain(job.error),
            "details": anonymise(job.traceback or job.error),
            "at": job.finished_at.isoformat() if job.finished_at else None,
        }
        for job in jobs
        if _photo_of(job) not in failed_ids
    ]
    merge_items = [
        {
            "group_id": group.id,
            "project_id": group.project_id,
            "project": names.get(group.project_id),
            "kind": group.kind.value,
            "frames": len(group.members),
            "reason": plain(group.error),
        }
        for group in _failed_merges(session, project_id)
    ]
    return {
        "photos": photo_items,
        "jobs": job_items,
        "merges": merge_items,
        "count": len(photo_items) + len(job_items) + len(merge_items),
    }


def retry(
    session: Session,
    *,
    photo_ids: list[int] | None = None,
    job_ids: list[int] | None = None,
    merge_ids: list[int] | None = None,
    project_id: int | None = None,
) -> int:
    """Put failed work back in the queue; ``None`` for both means everything listed.

    A failed photo is retried through its failed jobs. One with none -- failed
    during an export, whose items have their own "Riprova" -- gets its proxy
    rebuilt, which is where a photo's development starts again.
    """
    everything = photo_ids is None and job_ids is None and merge_ids is None
    listed = overview(session, project_id)
    if everything:
        photo_ids = [p["photo_id"] for p in listed["photos"]]
        job_ids = [j["job_id"] for j in listed["jobs"]]
        merge_ids = [m["group_id"] for m in listed["merges"]]
    retried = 0
    wanted = set(photo_ids or [])
    jobs = _failed_jobs(session, project_id)
    for photo_id in wanted:
        photo = session.get(Photo, photo_id)
        if photo is None or photo.status is not PhotoStatus.FAILED:
            continue
        own = [job for job in jobs if _photo_of(job) == photo_id]
        if own:
            retried += sum(jobq.retry_job(session, job.id) for job in own)
        else:
            job = jobq.enqueue(
                session,
                JobKind.PROXY,
                {"photo_id": photo_id, "force": True},
                project_id=photo.project_id,
                dedupe_key=f"proxy:{photo_id}",
            )
            retried += int(job is not None)
    for job_id in job_ids or []:
        retried += int(jobq.retry_job(session, job_id))
    from .merge import virtual

    for group_id in merge_ids or []:
        group = session.get(MergeGroup, group_id)
        if group is not None and group.decision is MergeDecision.FAILED:
            virtual.accept(session, group)
            retried += 1
    session.flush()
    return retried
