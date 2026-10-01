# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The photographs and their edits: the heart of the schema of section 11.

Three rules run through these tables and explain most of their oddities.

**Nothing about a photo is ever deleted.** A file that disappears from the
source folder is marked ``missing``, not removed; an edit that is undone becomes
a new :class:`EditVersion` whose parent is the one it undid. The history is a
tree (section 23) and the catalogue is append-mostly, because the user's work is
the one thing here that cannot be recomputed.

**A decision by the user outranks anything automatic.** ``Photo.cull_decided_by``
is the clearest case: a recomputation of the scores may rewrite
``culling_score`` but must leave ``culled`` alone when the user set it.

**The source folder is never written to** (section 2). Nothing here points at a
file the program creates next to a RAW: proxies, intermediates and exports all
live under the paths of ``config.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON as SAJSON
from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime, enum_column, utcnow
from .enums import (
    CropDecision,
    CullDecidedBy,
    CullingMode,
    EditVersionSource,
    ExportConflict,
    PhotoKind,
    PhotoStatus,
    ProjectStatus,
)

if TYPE_CHECKING:  # the relationship is resolved by name at mapper configuration
    from .work import Job

__all__ = ["CropProposal", "EditVersion", "Photo", "Project"]


class Project(Base):
    """A folder of RAW files plus every choice the user made about it."""

    __tablename__ = "project"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    source_dir: Mapped[str] = mapped_column(Text)
    output_dir: Mapped[str | None] = mapped_column(Text, default=None)
    style_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("style_profile.id", ondelete="SET NULL"), default=None
    )
    #: Weight of the coherence regulariser of section 8.4 (default 0.35 there).
    coherence_lambda: Mapped[float] = mapped_column(Float, default=0.35)
    #: Below this confidence a photo goes to individual review (section 9.1:
    #: default 0.55; catalogues before schema 5 were created with 0.6, which no
    #: screen could change, and are moved to 0.55 by the migration).
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.55)
    #: The user's weights of the confidence terms (``review/confidence.py``);
    #: a term not here has its default.
    confidence_weights: Mapped[dict] = mapped_column(
        SAJSON, default=dict, server_default=text("'{}'")
    )

    culling_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    culling_mode: Mapped[CullingMode] = mapped_column(
        enum_column(CullingMode, "culling_mode"), default=CullingMode.CONSERVATIVE
    )
    culling_target: Mapped[float | None] = mapped_column(Float, default=None)
    culling_aggressiveness: Mapped[float] = mapped_column(Float, default=0.5)
    culling_weights: Mapped[dict] = mapped_column(SAJSON, default=dict)
    culling_criteria: Mapped[dict] = mapped_column(SAJSON, default=dict)

    export_template: Mapped[str] = mapped_column(Text, default="{basename}.{ext}")
    export_on_conflict: Mapped[ExportConflict] = mapped_column(
        enum_column(ExportConflict, "export_on_conflict"), default=ExportConflict.ASK
    )
    export_strip_gps: Mapped[bool] = mapped_column(Boolean, default=False)
    #: The rest of the Export screen -- format, quality, colour space, size,
    #: output sharpening, sidecars -- as ``export/settings.py`` validates it.
    #: One column rather than eight: they are only ever read together, by the
    #: screen and by a new batch. Not in the sketch of section 11.
    export_settings: Mapped[dict] = mapped_column(
        SAJSON, default=dict, server_default=text("'{}'")
    )

    #: The source folder is not reachable: offer to re-point it (section 15).
    source_missing: Mapped[bool] = mapped_column(Boolean, default=False)
    merge_detection_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    #: Two crop proposals rejected in a row: stop proposing here (section 6.4).
    #: Not in the sketch of section 11; the rule needs somewhere to live, and
    #: the user can switch proposals back on.
    crop_proposals_paused: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("0")
    )

    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, onupdate=utcnow
    )
    status: Mapped[ProjectStatus] = mapped_column(
        enum_column(ProjectStatus, "project_status"), default=ProjectStatus.NEW
    )

    photos: Mapped[list[Photo]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    jobs: Mapped[list[Job]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Photo(Base):
    """One shot. Either a RAW on disk or, from phase 11, a merge of several."""

    __tablename__ = "photo"
    __table_args__ = (
        # Deduplication by content (section 15): one Photo per distinct file,
        # whatever it is called. Merged photos have no hash of their own, and
        # SQLite allows any number of NULLs in a unique index.
        UniqueConstraint("project_id", "hash", name="uq_photo_project_hash"),
        Index("ix_photo_project_status", "project_id", "status"),
        Index("ix_photo_project_path", "project_id", "path"),
        Index("ix_photo_burst_group", "burst_group_id"),
        Index("ix_photo_cluster", "cluster_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("project.id", ondelete="CASCADE"), index=True
    )
    #: Absolute path of the RAW. NULL for a merged photo, which has no file of
    #: its own -- its pixels live in ``intermediate_path`` (section 25.1).
    path: Mapped[str | None] = mapped_column(Text, default=None)
    filename: Mapped[str] = mapped_column(Text)
    #: SHA-256 of the file contents. NULL for merged photos.
    hash: Mapped[str | None] = mapped_column(String(64), default=None)

    camera: Mapped[str | None] = mapped_column(String(120), default=None)
    lens: Mapped[str | None] = mapped_column(String(160), default=None)
    iso: Mapped[int | None] = mapped_column(Integer, default=None)
    aperture: Mapped[float | None] = mapped_column(Float, default=None)
    shutter: Mapped[float | None] = mapped_column(Float, default=None)
    focal_length: Mapped[float | None] = mapped_column(Float, default=None)
    shot_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)
    orientation: Mapped[int | None] = mapped_column(Integer, default=None)

    proxy_path: Mapped[str | None] = mapped_column(Text, default=None)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    scene_features: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    cluster_id: Mapped[int | None] = mapped_column(Integer, default=None)
    #: Distance order inside the cluster; 0 is the medoid, the representative
    #: the review of section 9.2 shows.
    cluster_rank: Mapped[int | None] = mapped_column(Integer, default=None)
    #: What the analysis of phase 5 found and why: lens profile, straightening
    #: outcome, as-shot white balance, versions of the features. Not in the
    #: sketch of section 11, which has the columns for the results but none for
    #: the reasons the interface has to show.
    analysis: Mapped[dict | None] = mapped_column(SAJSON(none_as_null=True), default=None)
    #: The style prediction of phase 6 before coherence: profile, style vector,
    #: context, and which samples it came from (the transparency of section 8.3).
    prediction: Mapped[dict | None] = mapped_column(SAJSON(none_as_null=True), default=None)
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    escalation_reasons: Mapped[list | None] = mapped_column(SAJSON, default=None)
    #: The user's decision in the review of section 9.2, which no
    #: recomputation of the confidence overrules -- the rule of
    #: ``cull_decided_by``: ``{"decision": "approved" | "rejected", "by":
    #: "scene" | "photo", "at": ...}``, or ``{"queued": true, "by": "scene"}``
    #: for the photos of a scene the user rejected. Not in the sketch of
    #: section 11, whose ``status`` says where a photo is but not who put it
    #: there.
    review: Mapped[dict | None] = mapped_column(SAJSON(none_as_null=True), default=None)
    status: Mapped[PhotoStatus] = mapped_column(
        enum_column(PhotoStatus, "photo_status"), default=PhotoStatus.IMPORTED
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    #: Set when ``status`` is ``failed``. Not in the schema sketch of section 11,
    #: but test 11 asks for "a readable message" on a failed photo and there was
    #: nowhere to put one.
    error: Mapped[str | None] = mapped_column(Text, default=None)

    sharpness: Mapped[float | None] = mapped_column(Float, default=None)
    motion_blur: Mapped[float | None] = mapped_column(Float, default=None)
    exposure_score: Mapped[float | None] = mapped_column(Float, default=None)
    face_score: Mapped[float | None] = mapped_column(Float, default=None)
    aesthetic_score: Mapped[float | None] = mapped_column(Float, default=None)
    culling_score: Mapped[float | None] = mapped_column(Float, default=None)
    burst_group_id: Mapped[int | None] = mapped_column(Integer, default=None)
    burst_rank: Mapped[int | None] = mapped_column(Integer, default=None)
    #: Everything the culling analysis measured on the embedded preview --
    #: signature for bursts, EXIF for brackets, the measures behind the scores
    #: -- so that regrouping and reselecting never need the file again
    #: (section 7.3). Not in the sketch of section 11, which lists the scores
    #: but not what they are computed from.
    culling_features: Mapped[dict | None] = mapped_column(
        # ``none_as_null``: "not analysed" is SQL NULL, which is what the
        # queries filter on, and not the JSON document ``null``.
        SAJSON(none_as_null=True), default=None
    )
    culled: Mapped[bool] = mapped_column(Boolean, default=False)
    cull_reasons: Mapped[list | None] = mapped_column(SAJSON, default=None)
    #: ``user`` freezes the decision against every automatic recomputation.
    cull_decided_by: Mapped[CullDecidedBy | None] = mapped_column(
        enum_column(CullDecidedBy, "cull_decided_by"), default=None
    )

    #: A JPEG the camera wrote next to the RAW. Not a Photo of its own (section 15).
    sidecar_jpeg_path: Mapped[str | None] = mapped_column(Text, default=None)
    #: Other paths whose content is byte-identical to this one.
    duplicate_paths: Mapped[list | None] = mapped_column(SAJSON, default=None)
    #: The file is gone from the source folder. The row stays.
    missing: Mapped[bool] = mapped_column(Boolean, default=False)

    kind: Mapped[PhotoKind] = mapped_column(
        enum_column(PhotoKind, "photo_kind"), default=PhotoKind.RAW
    )
    merge_group_id: Mapped[int | None] = mapped_column(
        ForeignKey("merge_group.id", ondelete="SET NULL"), default=None
    )
    #: Linear scene-referred buffer of a merged photo, regenerable from its members.
    intermediate_path: Mapped[str | None] = mapped_column(Text, default=None)
    #: Out of the working set because another photo stands in for it: a frame
    #: of an accepted merge, or a merged photo whose merge was undone (kept, with
    #: its history, for when the user accepts the merge again). Style, review
    #: and export see only the photos where this is false (section 25.1).
    superseded: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    project: Mapped[Project] = relationship(back_populates="photos")
    versions: Mapped[list[EditVersion]] = relationship(
        back_populates="photo",
        cascade="all, delete-orphan",
        foreign_keys="EditVersion.photo_id",
    )


class EditVersion(Base):
    """One state of a photo's ``EditParams``. The history is a tree (section 23)."""

    __tablename__ = "edit_version"
    __table_args__ = (
        # "Exactly one current version per photo" is an invariant test 16 checks
        # explicitly, so it is enforced by the database rather than by care.
        Index(
            "uq_edit_version_current",
            "photo_id",
            unique=True,
            sqlite_where=text("is_current = 1"),
        ),
        Index("ix_edit_version_photo", "photo_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id", ondelete="CASCADE"))
    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("edit_version.id", ondelete="SET NULL"), default=None
    )
    params: Mapped[dict] = mapped_column(SAJSON)
    #: ``EditParams.version`` at the time of writing, for the migrations of params.py.
    params_version: Mapped[int] = mapped_column(Integer, default=1)
    source: Mapped[EditVersionSource] = mapped_column(
        enum_column(EditVersionSource, "edit_version_source"), default=EditVersionSource.PREDICTED
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)

    photo: Mapped[Photo] = relationship(back_populates="versions", foreign_keys=[photo_id])


class CropProposal(Base):
    __tablename__ = "crop_proposal"

    id: Mapped[int] = mapped_column(primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photo.id", ondelete="CASCADE"), index=True)
    rect: Mapped[dict] = mapped_column(SAJSON)
    aspect: Mapped[str | None] = mapped_column(String(16), default=None)
    score: Mapped[float | None] = mapped_column(Float, default=None)
    decision: Mapped[CropDecision] = mapped_column(
        enum_column(CropDecision, "crop_decision"), default=CropDecision.PENDING
    )
    #: When the user decided, to find "two rejections in a row" (section 6.4).
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)


