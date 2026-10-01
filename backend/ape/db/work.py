# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The job queue, which is a table (sections 11 and 12).

A queue in memory disappears when the process does. This one is rows, so killing
the program in the middle of a thousand-photo batch loses nothing -- which is
test 6. ``jobs/queue.py`` holds the operations; this is only the shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON as SAJSON
from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime, enum_column, utcnow
from .enums import JobKind, JobState

if TYPE_CHECKING:  # the relationship is resolved by name at mapper configuration
    from .catalog import Project

__all__ = ["Job"]


class Job(Base):
    """A unit of background work. The queue is this table (section 12).

    ``payload`` carries everything a worker needs, so a job is runnable after a
    restart without reconstructing any in-memory state -- which is what makes
    the resume of test 6 possible at all.
    """

    __tablename__ = "job"
    __table_args__ = (
        # The queue's hot query: oldest queued job of a given kind. Ordering by
        # id inside the index is what keeps the pickup an index scan of one row.
        Index("ix_job_pickup", "state", "priority", "id"),
        Index("ix_job_project_state", "project_id", "state"),
        # Idempotence of enqueueing: at most one *live* job per (kind, key).
        # Partial on purpose -- once a job has finished, the same work may
        # legitimately be asked for again. A NULL key opts out, since SQLite
        # counts NULLs as distinct in a unique index.
        Index(
            "uq_job_live_dedupe",
            "kind",
            "dedupe_key",
            unique=True,
            sqlite_where=text("state IN ('queued', 'running')"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("project.id", ondelete="CASCADE"), default=None
    )
    kind: Mapped[JobKind] = mapped_column(enum_column(JobKind, "job_kind"))
    payload: Mapped[dict] = mapped_column(SAJSON, default=dict)
    state: Mapped[JobState] = mapped_column(
        enum_column(JobState, "job_state"), default=JobState.QUEUED
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    #: The traceback of the last failure, for "Dettagli tecnici" (section 19).
    traceback: Mapped[str | None] = mapped_column(Text, default=None)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    #: Lower runs first. Interactive work (a preview the user is waiting for)
    #: jumps ahead of a thousand-photo batch.
    priority: Mapped[int] = mapped_column(Integer, default=100)
    #: Identifies the work, not the request: enqueueing the same proxy twice
    #: must not produce two jobs. NULL opts out.
    dedupe_key: Mapped[str | None] = mapped_column(String(200), default=None)
    #: PID of the worker that claimed it, for the diagnostics of section 19.
    claimed_by: Mapped[int | None] = mapped_column(Integer, default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)

    project: Mapped[Project | None] = relationship(back_populates="jobs")
