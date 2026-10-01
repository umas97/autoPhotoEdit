# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A running export batch: pause, resume, cancel, retry, and how far it got.

**Pause is a property of the batch.** The queued jobs are cancelled and the
items stay queued; resuming puts the same jobs back. A job already running
finishes its photo -- interrupting a write is how half files happen.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    ExportBatch,
    ExportBatchState,
    ExportItem,
    ExportItemState,
    Job,
    JobKind,
    JobState,
    Photo,
    PhotoStatus,
    utcnow,
)
from ..jobs import queue as jobq

__all__ = ["batch_summary", "cancel", "finish_if_done", "pause", "resume", "retry_failed"]


def _items(session: Session, batch: ExportBatch, *states: ExportItemState) -> list[ExportItem]:
    return list(
        session.scalars(
            select(ExportItem).where(ExportItem.batch_id == batch.id, ExportItem.state.in_(states))
        ).all()
    )


def pause(session: Session, batch: ExportBatch) -> int:
    """Stop handing out the batch's photos. Returns how many were held back."""
    if batch.state is not ExportBatchState.RUNNING:
        return 0
    batch.state = ExportBatchState.PAUSED
    held = 0
    for item in _items(session, batch, ExportItemState.QUEUED):
        if item.job_id is not None and jobq.cancel_job(session, item.job_id):
            held += 1
    return held


def _requeue(session: Session, batch: ExportBatch, item: ExportItem) -> None:
    if item.job_id is not None and jobq.retry_job(session, item.job_id):
        return
    job = session.get(Job, item.job_id) if item.job_id is not None else None
    if job is not None and job.state in (JobState.QUEUED, JobState.RUNNING):
        return  # still live: it was running when the batch was paused
    fresh = jobq.enqueue(
        session,
        JobKind.EXPORT,
        {"item_id": item.id},
        project_id=batch.project_id,
        dedupe_key=f"export:{batch.id}:{item.id}",
    )
    if fresh is not None:
        item.job_id = fresh.id


def resume(session: Session, batch: ExportBatch) -> int:
    """Put a paused batch's photos back in the queue. Returns how many."""
    if batch.state is not ExportBatchState.PAUSED:
        return 0
    batch.state = ExportBatchState.RUNNING
    items = _items(session, batch, ExportItemState.QUEUED)
    for item in items:
        _requeue(session, batch, item)
    return len(items)


def cancel(session: Session, batch: ExportBatch) -> int:
    """Give up on the rest of a batch. Files already written stay where they are."""
    if batch.state in (ExportBatchState.DONE, ExportBatchState.CANCELLED):
        return 0
    batch.state = ExportBatchState.CANCELLED
    batch.finished_at = utcnow()
    items = _items(session, batch, ExportItemState.QUEUED)
    for item in items:
        if item.job_id is not None:
            jobq.cancel_job(session, item.job_id)
        item.state = ExportItemState.CANCELLED
    return len(items)


def retry_failed(
    session: Session, batch: ExportBatch, item_ids: Iterable[int] | None = None
) -> int:
    """Queue the failed photos of a batch again ("Riprova", section 19)."""
    wanted = set(item_ids) if item_ids is not None else None
    failed = [
        item
        for item in _items(session, batch, ExportItemState.FAILED)
        if wanted is None or item.id in wanted
    ]
    for item in failed:
        item.state = ExportItemState.QUEUED
        item.error = None
        item.finished_at = None
        photo = session.get(Photo, item.photo_id)
        if photo is not None and photo.status is PhotoStatus.FAILED:
            # The user says the cause is gone (a file put back, a card
            # reinserted): let the photo be tried again.
            photo.status = _status_before_failure(photo)
            photo.error = None
        _requeue(session, batch, item)
    if failed and batch.state in (ExportBatchState.DONE, ExportBatchState.CANCELLED):
        batch.state = ExportBatchState.RUNNING
        batch.finished_at = None
    return len(failed)


def _status_before_failure(photo: Photo) -> PhotoStatus:
    """Where a photo goes back to when its failure is lifted: the furthest
    stage its data says it reached. The review then places it again."""
    if photo.prediction:
        return PhotoStatus.PREDICTED
    if photo.analysis:
        return PhotoStatus.ANALYZED
    return PhotoStatus.IMPORTED


def finish_if_done(session: Session, batch: ExportBatch) -> bool:
    """Close the batch when nothing of it is left to run."""
    if batch.state is not ExportBatchState.RUNNING:
        return False
    remaining = session.scalar(
        select(func.count(ExportItem.id)).where(
            ExportItem.batch_id == batch.id, ExportItem.state == ExportItemState.QUEUED
        )
    )
    if remaining:
        return False
    batch.state = ExportBatchState.DONE
    batch.finished_at = utcnow()
    return True


def batch_summary(session: Session, batch: ExportBatch, *, with_items: bool = False) -> dict:
    counts = {state.value: 0 for state in ExportItemState}
    for state, count in session.execute(
        select(ExportItem.state, func.count(ExportItem.id))
        .where(ExportItem.batch_id == batch.id)
        .group_by(ExportItem.state)
    ):
        counts[state.value if hasattr(state, "value") else str(state)] = int(count)
    outcomes = dict(
        session.execute(
            select(ExportItem.outcome, func.count(ExportItem.id))
            .where(ExportItem.batch_id == batch.id, ExportItem.outcome.is_not(None))
            .group_by(ExportItem.outcome)
        ).all()
    )
    summary: dict[str, Any] = {
        "id": batch.id,
        "state": batch.state.value,
        "total": batch.total,
        "counts": counts,
        "outcomes": {str(k): int(v) for k, v in outcomes.items()},
        "output_dir": batch.settings.get("output_dir"),
        "created_at": batch.created_at,
        "finished_at": batch.finished_at,
    }
    if with_items:
        rows = session.execute(
            select(ExportItem, Photo.filename)
            .join(Photo, Photo.id == ExportItem.photo_id)
            .where(
                ExportItem.batch_id == batch.id,
                ExportItem.state.in_((ExportItemState.FAILED, ExportItemState.SKIPPED)),
            )
            .order_by(ExportItem.position)
        ).all()
        summary["problems"] = [
            {
                "item_id": item.id,
                "photo_id": item.photo_id,
                "filename": filename,
                "state": item.state.value,
                "error": item.error,
            }
            for item, filename in rows
        ]
    return summary
