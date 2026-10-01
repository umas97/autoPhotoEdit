# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Learned looks, their training pairs, and the two global tables.

``StyleProfile`` covers both halves of section 8: a profile learned from the
user's own edits, which carries a model, and a built-in of section 22, which
carries rules instead and cannot be overwritten. The CHECK constraint is what
keeps the two from being confused.

``LensProfileOverride`` and ``Setting`` are deliberately not per-project: a lens
the user has taught us about, and a preference like the copyright line, belong
to the person rather than to the job they are doing today.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON as SAJSON
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, UTCDateTime, enum_column, utcnow
from .enums import StyleSampleStatus

__all__ = ["LensProfileOverride", "Setting", "StyleProfile", "StyleSample"]


class StyleProfile(Base):
    """A learned look, or one of the built-in ones of section 22."""

    __tablename__ = "style_profile"
    __table_args__ = (
        # A built-in profile has no trained model: it predicts from rules.
        CheckConstraint(
            "builtin = 0 OR (model IS NULL AND pca IS NULL)",
            name="ck_builtin_has_no_model",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    model: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    pca: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    norm_stats: Mapped[dict | None] = mapped_column(SAJSON, default=None)
    embedding_centroid: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    n_pairs: Mapped[int] = mapped_column(Integer, default=0)
    #: Shipped with the program: not deletable, not overwritable. Editing one
    #: creates a user copy instead.
    builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    builtin_rules: Mapped[dict | None] = mapped_column(SAJSON, default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utcnow, onupdate=utcnow
    )
    #: When ``model`` was last fitted; ``None`` while it has never been.
    trained_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    #: Where the pairs came from, for the "add more" button (not a constraint).
    raw_dir: Mapped[str | None] = mapped_column(Text, default=None)
    reference_dir: Mapped[str | None] = mapped_column(Text, default=None)

    samples: Mapped[list[StyleSample]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )


class StyleSample(Base):
    """One RAW + edited-reference pair a profile was trained on."""

    __tablename__ = "style_sample"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("style_profile.id", ondelete="CASCADE"), index=True
    )
    #: Kept for retraining and for the UI. The file may well be gone: a profile
    #: must import and predict without it (test 15).
    raw_path: Mapped[str | None] = mapped_column(Text, default=None)
    reference_path: Mapped[str | None] = mapped_column(Text, default=None)
    params: Mapped[dict] = mapped_column(SAJSON)
    scene_features: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    residual_loss: Mapped[float | None] = mapped_column(Float, default=None)
    #: Left out of training: by the user, or because it is unreproducible.
    excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[StyleSampleStatus] = mapped_column(
        enum_column(StyleSampleStatus, "style_sample_status"),
        default=StyleSampleStatus.PENDING,
    )
    error: Mapped[str | None] = mapped_column(Text, default=None)
    #: ``name`` / ``xmp`` / ``time`` / ``manual``: how the pair was made.
    pairing: Mapped[str | None] = mapped_column(String(16), default=None)
    #: The style vector (``style/vector.py`` order) and the photo context it
    #: is relative to -- what the model trains on. ``params`` is the same
    #: thing as full ``EditParams``, for people and for the exporters.
    vector: Mapped[list | None] = mapped_column(SAJSON(none_as_null=True), default=None)
    context: Mapped[dict | None] = mapped_column(SAJSON(none_as_null=True), default=None)
    #: Mean ΔE2000 of the inversion's render against the reference.
    delta_e: Mapped[float | None] = mapped_column(Float, default=None)
    #: The 512 px reference a portable profile carries (section 20.1).
    thumbnail: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    #: A correction from the review of a project (section 9.2.4): the project
    #: and photo it came from. ``None`` for a pair of files.
    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("project.id", ondelete="SET NULL"), default=None, index=True
    )
    photo_id: Mapped[int | None] = mapped_column(
        ForeignKey("photo.id", ondelete="SET NULL"), default=None
    )

    profile: Mapped[StyleProfile] = relationship(back_populates="samples")


class LensProfileOverride(Base):
    """User-taught mapping from an EXIF lens name to a lensfun entry. Global."""

    __tablename__ = "lens_profile_override"

    lens_model: Mapped[str] = mapped_column(String(160), primary_key=True)
    lensfun_maker: Mapped[str] = mapped_column(String(120))
    lensfun_model: Mapped[str] = mapped_column(String(160))


class Setting(Base):
    """Global preferences, shared by every project."""

    __tablename__ = "setting"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[dict | list | str | int | float | bool | None] = mapped_column(SAJSON)


