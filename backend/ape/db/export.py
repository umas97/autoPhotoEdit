# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export batches: what the user asked to export, and what happened to each photo.

Not in the sketch of section 11, which has a ``Job`` of kind ``export`` and
nothing else. A job is a unit of work for the pool; it has nowhere to keep what
section 16 asks to know afterwards -- the name the photo was given, whether it
was renamed or skipped, which version of its edit was developed, where its
sidecars went. And "pause and resume" (phase 8) is a property of the whole
batch, not of any job in it.

So a batch is one row with the settings frozen at the moment the user pressed
"Esporta", and one item per photo, each pointing at the job that renders it.
Pausing cancels the items' queued jobs and resuming re-queues them, so the
queue itself needs no notion of a pause.

A photo's ``status`` is *not* moved to ``exported``: exporting in the middle of
the review, to look at a few files, must not empty the review's queue. Whether
a photo has been exported, and whether it changed since, is read from its
items.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import JSON as SAJSON
from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime, enum_column, utcnow
from .enums import ExportConflict

__all__ = ["ExportBatch", "ExportBatchState", "ExportItem", "ExportItemState"]


class ExportBatchState(enum.StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    CANCELLED = "cancelled"


class ExportItemState(enum.StrEnum):
    QUEUED = "queued"
    DONE = "done"
    #: The file existed and the policy was "salta": not a failure (section 16.2).
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExportBatch(Base):
    """One press of "Esporta": the settings, frozen, and its items."""

    __tablename__ = "export_batch"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("project.id", ondelete="CASCADE"), index=True
    )
    #: ``export/settings.py::ExportSettings``, as it was when the batch started.
    settings: Mapped[dict] = mapped_column(SAJSON)
    state: Mapped[ExportBatchState] = mapped_column(
        enum_column(ExportBatchState, "export_batch_state"), default=ExportBatchState.RUNNING
    )
    total: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)

    items: Mapped[list[ExportItem]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class ExportItem(Base):
    """One photo of a batch."""

    __tablename__ = "export_item"
    __table_args__ = (
        Index("ix_export_item_batch_state", "batch_id", "state"),
        Index("ix_export_item_photo", "photo_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("export_batch.id", ondelete="CASCADE"))
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id", ondelete="CASCADE"))
    #: The edit developed: the version current when the batch started. An edit
    #: made while the batch runs belongs to the next export, not half of this one.
    version_id: Mapped[int | None] = mapped_column(
        ForeignKey("edit_version.id", ondelete="SET NULL"), default=None
    )
    #: 1-based order in the batch: the ``{counter}`` of section 16.1.
    position: Mapped[int] = mapped_column(Integer)
    #: The rendered file name, before any collision rename.
    name: Mapped[str] = mapped_column(Text)
    #: The user's answer for this one file, when it collided and they chose
    #: without "applica a tutte". ``None``: the batch's policy.
    on_conflict: Mapped[ExportConflict | None] = mapped_column(
        enum_column(ExportConflict, "export_item_conflict"), default=None
    )
    state: Mapped[ExportItemState] = mapped_column(
        enum_column(ExportItemState, "export_item_state"), default=ExportItemState.QUEUED
    )
    #: ``written`` | ``renamed`` | ``overwritten`` | ``skipped`` (``naming.WriteOutcome``).
    outcome: Mapped[str | None] = mapped_column(String(16), default=None)
    path: Mapped[str | None] = mapped_column(Text, default=None)
    #: ``[{"kind": "darktable" | "adobe", "path": ..., "outcome": ...}]``.
    sidecars: Mapped[list | None] = mapped_column(SAJSON(none_as_null=True), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("job.id", ondelete="SET NULL"), default=None
    )
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)

    batch: Mapped[ExportBatch] = relationship(back_populates="items")
