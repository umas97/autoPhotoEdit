# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The queue, seen from outside: what is running, what failed, what to retry.

This is also the Problems panel of section 19. A failed job keeps its message,
so the panel is a query rather than a log parser, and "retry" is a state change
on a row rather than a new request the user has to reconstruct.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Job, JobKind, JobState
from ..jobs import queue as jobq
from .deps import get_session
from .schemas import JobOut, JobSummary

__all__ = ["router", "summarise"]

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def summarise(session: Session, project_id: int | None = None) -> JobSummary:
    """Counts by state plus a single fraction for the progress bar."""
    counts = jobq.counts_by_state(session, project_id)
    active = counts[JobState.QUEUED.value] + counts[JobState.RUNNING.value]
    total = sum(counts.values())
    finished = total - active
    return JobSummary(
        counts=counts,
        active=active,
        total=total,
        progress=1.0 if total == 0 else finished / total,
    )


@router.get("", response_model=list[JobOut])
def list_jobs(
    session: Session = Depends(get_session),
    project_id: int | None = None,
    state: JobState | None = None,
    kind: JobKind | None = None,
    limit: int = Query(200, ge=1, le=2000),
) -> list[JobOut]:
    statement = select(Job).order_by(Job.id.desc()).limit(limit)
    if project_id is not None:
        statement = statement.where(Job.project_id == project_id)
    if state is not None:
        statement = statement.where(Job.state == state)
    if kind is not None:
        statement = statement.where(Job.kind == kind)
    return [JobOut.model_validate(job) for job in session.scalars(statement).all()]


@router.get("/summary", response_model=JobSummary)
def job_summary(
    session: Session = Depends(get_session), project_id: int | None = None
) -> JobSummary:
    return summarise(session, project_id)


@router.post("/{job_id}/retry", response_model=JobOut)
def retry(job_id: int, session: Session = Depends(get_session)) -> JobOut:
    if not jobq.retry_job(session, job_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "solo un job fallito o annullato può essere ripreso"
        )
    session.flush()
    return JobOut.model_validate(session.get(Job, job_id))


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel(job_id: int, session: Session = Depends(get_session)) -> JobOut:
    """Cancel a job that has not started.

    A running job is left alone deliberately: stopping it in the middle of
    writing a file is how a truncated proxy gets into the cache. The worker
    stops between jobs, not inside one.
    """
    if not jobq.cancel_job(session, job_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "il job è già partito o già concluso; i job in corso terminano da soli",
        )
    session.flush()
    return JobOut.model_validate(session.get(Job, job_id))


@router.post("/cancel-project/{project_id}", response_model=dict)
def cancel_project(project_id: int, session: Session = Depends(get_session)) -> dict:
    cancelled = jobq.cancel_project_jobs(session, project_id)
    return {"cancelled": cancelled}
