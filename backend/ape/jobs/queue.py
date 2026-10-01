# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The work queue, which is a table (docs/SPEC.md sections 11 and 12).

A queue in memory disappears when the process does. This one is rows in the
catalogue, so killing the program in the middle of a thousand-photo batch loses
nothing: the jobs that had not started are still ``queued``, the ones that were
running go back to ``queued`` at the next startup, and the ones that finished
stay finished. That is test 6, and it is the reason the queue is here rather
than in ``multiprocessing``.

**Claiming is one statement.** Sixteen workers ask for work at the same time;
what stops two of them taking the same job is a single ``UPDATE ... WHERE id =
(SELECT ... LIMIT 1) RETURNING``, which SQLite executes atomically. No
application-level locking, no "select then update" window.

**Enqueueing is idempotent.** A job carries a ``dedupe_key`` naming the work
rather than the request -- ``proxy:1234`` -- and a partial unique index refuses
a second live job with the same key. Asking twice for the same proxy is
therefore free, which matters because the UI asks whenever a thumbnail scrolls
into view.

**Nothing is deleted on failure.** A failed job keeps its error message for the
Problems panel of section 19, and ``retry`` makes a new attempt out of it
rather than a new job, so ``attempts`` stays meaningful.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db.base import sql_timestamp
from ..db.models import Job, JobKind, JobState, utcnow
from .wake import ring_after_commit

__all__ = [
    "JobRecord",
    "cancel_job",
    "cancel_project_jobs",
    "claim_job",
    "complete_job",
    "counts_by_state",
    "enqueue",
    "fail_job",
    "next_queued",
    "requeue_running",
    "retry_job",
    "update_progress",
]

_log = logging.getLogger(__name__)

#: Attempts before a job is left ``failed`` for good. Three covers the
#: transient causes -- a file briefly locked, a moment of memory pressure --
#: without hiding a real defect behind a retry loop.
MAX_ATTEMPTS = 3

#: Default priorities. Lower runs first; interactive work overtakes batches.
PRIORITY = {
    JobKind.RENDER_PREVIEW: 10,
    #: The user clicked "Cielo" and is looking at the photo, waiting.
    JobKind.SEGMENT: 15,
    #: The user released the eraser and is looking at the photo, waiting.
    JobKind.RETOUCH_FILL: 15,
    #: One photo of a "Copia punti": asked for, but not looked at yet.
    JobKind.RETOUCH_COPY: 30,
    #: A merge preview: the user pressed "Anteprima" on a group and waits.
    JobKind.MERGE_PREVIEW: 12,
    #: An accepted merge. Ahead of the proxies, whose photos exist already:
    #: this one's photo does not exist until it has run.
    JobKind.MERGE: 45,
    JobKind.DETECT_MERGES: 55,
    JobKind.PROXY: 50,
    JobKind.PREDICT: 60,
    #: Training runs behind the project's own work: a proxy the user is
    #: waiting to see matters more than a profile that will be ready later.
    JobKind.STYLE_PAIR: 80,
    JobKind.ANALYZE: 70,
    JobKind.EXPORT: 90,
}


@dataclass(frozen=True)
class JobRecord:
    """A claimed job, detached from the session that produced it.

    Workers run in other processes and must not hold an ORM object across a
    long computation, so what leaves the queue is a plain frozen value.
    """

    id: int
    kind: JobKind
    payload: dict[str, Any]
    project_id: int | None
    attempts: int


def enqueue(
    session: Session,
    kind: JobKind,
    payload: dict[str, Any] | None = None,
    *,
    project_id: int | None = None,
    dedupe_key: str | None = None,
    priority: int | None = None,
) -> Job | None:
    """Add a job, unless an identical one is already waiting or running.

    Returns:
        The new :class:`Job`, or ``None`` when an equivalent live job existed.
    """
    if dedupe_key is not None:
        existing = session.scalars(
            select(Job).where(
                Job.kind == kind,
                Job.dedupe_key == dedupe_key,
                Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
            )
        ).first()
        if existing is not None:
            return None

    job = Job(
        project_id=project_id,
        kind=kind,
        payload=payload or {},
        dedupe_key=dedupe_key,
        priority=PRIORITY.get(kind, 100) if priority is None else priority,
    )
    session.add(job)
    try:
        session.flush()
    except IntegrityError:
        # Another process enqueued the same work between the check and the
        # insert. That is exactly what the unique index is there to catch.
        session.rollback()
        return None
    ring_after_commit(session)
    return job


def claim_job(
    session: Session,
    kinds: Sequence[JobKind] | None = None,
    *,
    project_id: int | None = None,
    worker_pid: int | None = None,
    limits: dict[JobKind, int] | None = None,
) -> JobRecord | None:
    """Take the next job atomically, or return ``None`` if there is none.

    The whole claim is one UPDATE with a subquery, so two workers racing for the
    last job cannot both get it: the second one's subquery no longer sees it.

    ``limits`` caps how many jobs of a kind may run at once, counted inside the
    same statement -- so the cap holds under the same atomicity. It exists for
    the full-resolution export, of which fourteen at once would not fit in the
    memory of the machine of section 3 (``WorkerPool.export_limit``).
    """
    conditions = ["state = 'queued'"]
    parameters: dict[str, Any] = {
        "pid": worker_pid if worker_pid is not None else os.getpid(),
        # Raw SQL bypasses the column type, so the timestamp is formatted here
        # the way UTCDateTime would have stored it.
        "now": sql_timestamp(),
    }
    if kinds:
        names = []
        for index, kind in enumerate(kinds):
            key = f"kind{index}"
            names.append(f":{key}")
            parameters[key] = kind.value
        conditions.append(f"kind IN ({', '.join(names)})")
    if project_id is not None:
        conditions.append("project_id = :project_id")
        parameters["project_id"] = project_id
    for index, (kind, cap) in enumerate((limits or {}).items()):
        conditions.append(
            f"NOT (kind = :capped{index} AND (SELECT COUNT(*) FROM job AS running"
            f" WHERE running.state = 'running' AND running.kind = :capped{index})"
            f" >= :cap{index})"
        )
        parameters[f"capped{index}"] = kind.value
        parameters[f"cap{index}"] = int(cap)

    statement = text(
        f"""
        UPDATE job
           SET state = 'running',
               claimed_by = :pid,
               started_at = :now,
               attempts = attempts + 1
         WHERE id = (
                 SELECT id FROM job
                  WHERE {" AND ".join(conditions)}
                  ORDER BY priority, id
                  LIMIT 1
               )
        RETURNING id, kind, payload, project_id, attempts
        """
    )
    row = session.execute(statement, parameters).first()
    session.commit()
    if row is None:
        return None

    import json

    payload = row.payload
    if isinstance(payload, str):
        payload = json.loads(payload)
    return JobRecord(
        id=int(row.id),
        kind=JobKind(row.kind),
        payload=payload or {},
        project_id=row.project_id,
        attempts=int(row.attempts),
    )


def update_progress(session: Session, job_id: int, progress: float) -> None:
    """Record how far along a running job is, in [0, 1]."""
    session.execute(
        update(Job)
        .where(Job.id == job_id, Job.state == JobState.RUNNING)
        .values(progress=min(1.0, max(0.0, float(progress))))
    )


def complete_job(session: Session, job_id: int, *, progress: float = 1.0) -> None:
    session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(
            state=JobState.DONE,
            progress=progress,
            error=None,
            finished_at=utcnow(),
            claimed_by=None,
        )
    )


def fail_job(
    session: Session, job_id: int, message: str, *, retry: bool = True, details: str | None = None
) -> JobState:
    """Mark a job failed, or put it back in the queue if attempts remain.

    Returns:
        The state the job ended up in, so the caller can log the difference
        between "will try again" and "gave up".
    """
    job = session.get(Job, job_id)
    if job is None:
        return JobState.FAILED

    # The message is what the Problems panel shows; truncate rather than let a
    # traceback of a thousand lines into the catalogue.
    job.error = message[:2000]
    # "Dettagli tecnici" of the Problems panel: the last lines are the ones
    # that say where, so a long traceback keeps its end.
    job.traceback = details[-20_000:] if details else None
    if retry and job.attempts < MAX_ATTEMPTS:
        job.state = JobState.QUEUED
        job.claimed_by = None
        job.started_at = None
        job.progress = 0.0
    else:
        job.state = JobState.FAILED
        job.finished_at = utcnow()
        job.claimed_by = None
    return job.state


def retry_job(session: Session, job_id: int) -> bool:
    """Put a failed job back in the queue, on the user's say-so."""
    job = session.get(Job, job_id)
    if job is None or job.state not in (JobState.FAILED, JobState.CANCELLED):
        return False
    job.state = JobState.QUEUED
    job.error = None
    job.traceback = None
    job.progress = 0.0
    job.finished_at = None
    job.claimed_by = None
    ring_after_commit(session)
    return True


def cancel_job(session: Session, job_id: int) -> bool:
    """Cancel a job that has not started. A running one is left to finish.

    Interrupting a running job mid-write is how half-written files happen; the
    worker checks for cancellation between photos instead.
    """
    job = session.get(Job, job_id)
    if job is None or job.state is not JobState.QUEUED:
        return False
    job.state = JobState.CANCELLED
    job.finished_at = utcnow()
    return True


def cancel_project_jobs(session: Session, project_id: int) -> int:
    """Cancel everything queued for a project. Returns how many."""
    result = session.execute(
        update(Job)
        .where(Job.project_id == project_id, Job.state == JobState.QUEUED)
        .values(state=JobState.CANCELLED, finished_at=utcnow())
    )
    return int(result.rowcount or 0)


def requeue_running(session: Session, *, worker_pid: int | None = None) -> int:
    """Put running jobs back in the queue. Called at startup, and after a crash.

    ``worker_pid`` restricts it to the jobs one dead worker held; without it,
    every running job goes back, which is what a fresh start wants.
    """
    statement = update(Job).where(Job.state == JobState.RUNNING)
    if worker_pid is not None:
        statement = statement.where(Job.claimed_by == worker_pid)
    result = session.execute(
        statement.values(state=JobState.QUEUED, claimed_by=None, started_at=None, progress=0.0)
    )
    if result.rowcount:
        ring_after_commit(session)
    return int(result.rowcount or 0)


def next_queued(session: Session, kinds: Sequence[JobKind] | None = None) -> Job | None:
    """Peek at the next job without claiming it. For the UI and for tests."""
    statement = select(Job).where(Job.state == JobState.QUEUED)
    if kinds:
        statement = statement.where(Job.kind.in_(list(kinds)))
    return session.scalars(statement.order_by(Job.priority, Job.id).limit(1)).first()


def counts_by_state(session: Session, project_id: int | None = None) -> dict[str, int]:
    """``{state: count}``, for the progress bar and the Problems panel."""
    statement = select(Job.state, func.count(Job.id)).group_by(Job.state)
    if project_id is not None:
        statement = statement.where(Job.project_id == project_id)
    counts = {state.value: 0 for state in JobState}
    for state, count in session.execute(statement):
        counts[state.value if hasattr(state, "value") else str(state)] = int(count)
    return counts
