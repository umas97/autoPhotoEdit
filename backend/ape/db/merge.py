# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-shot merges: the groups of section 25 and their members.

A group is a *proposal*. Detection fills this table; nothing here has produced a
pixel. ``decision = rejected`` is permanent for that group, which is what stops
the program suggesting the same bracketing again at every reopening.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON as SAJSON
from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime, enum_column, utcnow
from .enums import MergeDecision, MergeKind, MergeRole

__all__ = ["MergeGroup", "MergeMember"]


class MergeGroup(Base):
    """A candidate multi-shot merge (section 25). A proposal, never an action."""

    __tablename__ = "merge_group"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("project.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[MergeKind] = mapped_column(enum_column(MergeKind, "merge_kind"))
    decision: Mapped[MergeDecision] = mapped_column(
        enum_column(MergeDecision, "merge_decision"), default=MergeDecision.PROPOSED
    )
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    detect_reasons: Mapped[list | None] = mapped_column(SAJSON, default=None)
    params: Mapped[dict | None] = mapped_column(SAJSON, default=None)
    result_photo_id: Mapped[int | None] = mapped_column(Integer, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    #: What the last preview and the last full merge measured: alignment
    #: residuals, measured gains, coverage. Shown next to the preview so the
    #: user can judge a merge before paying for it.
    report: Mapped[dict | None] = mapped_column(SAJSON, default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)

    members: Mapped[list[MergeMember]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )


class MergeMember(Base):
    __tablename__ = "merge_member"
    __table_args__ = (UniqueConstraint("group_id", "photo_id", name="uq_merge_member"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    group_id: Mapped[int] = mapped_column(
        ForeignKey("merge_group.id", ondelete="CASCADE"), index=True
    )
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer, default=0)
    ev_offset: Mapped[float | None] = mapped_column(Float, default=None)
    role: Mapped[MergeRole] = mapped_column(
        enum_column(MergeRole, "merge_role"), default=MergeRole.MEMBER
    )

    group: Mapped[MergeGroup] = relationship(back_populates="members")


