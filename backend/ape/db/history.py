# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Snapshots: a named point in a project's history (section 23).

A snapshot records *which version was current* for every photo, not the pixels
and not the parameters. A thousand photos cost a few kilobytes, so taking one at
every phase change is free, and rolling back to one is a matter of moving
pointers -- which is also why a rollback is itself undoable.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON as SAJSON
from sqlalchemy import (
    Boolean,
    ForeignKey,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime, enum_column, utcnow
from .enums import PhotoStatus, SnapshotKind

__all__ = ["ProjectSnapshot", "SnapshotEntry"]


class ProjectSnapshot(Base):
    """A named point in time: which version was current for every photo."""

    __tablename__ = "project_snapshot"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("project.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[SnapshotKind] = mapped_column(
        enum_column(SnapshotKind, "snapshot_kind"), default=SnapshotKind.MANUAL
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    entries: Mapped[list[SnapshotEntry]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan"
    )


class SnapshotEntry(Base):
    """One photo's state inside a snapshot. A pointer, never pixels."""

    __tablename__ = "snapshot_entry"
    __table_args__ = (UniqueConstraint("snapshot_id", "photo_id", name="uq_snapshot_photo"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("project_snapshot.id", ondelete="CASCADE"), index=True
    )
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id", ondelete="CASCADE"))
    edit_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("edit_version.id", ondelete="SET NULL"), default=None
    )
    culled: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[PhotoStatus] = mapped_column(
        enum_column(PhotoStatus, "snapshot_photo_status"), default=PhotoStatus.IMPORTED
    )
    #: What the user decided, beyond the version: ``review`` (the photo's review
    #: record) and ``cull_decided_by``. ``None`` in snapshots of schema < 8.
    decisions: Mapped[dict | None] = mapped_column(SAJSON(none_as_null=True), default=None)

    snapshot: Mapped[ProjectSnapshot] = relationship(back_populates="entries")


