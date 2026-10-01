# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every enumerated value in the catalogue, in one place.

They are stored as strings (see ``base.enum_column``) and they are the
vocabulary the API speaks too, so ``schemas.py`` imports them rather than
redeclaring them: a state that exists in the database and not in the interface
is a state the user can reach and not see.
"""

from __future__ import annotations

import enum

__all__ = [
    "CropDecision",
    "CullDecidedBy",
    "CullingMode",
    "EditVersionSource",
    "ExportConflict",
    "JobKind",
    "JobState",
    "MergeDecision",
    "MergeKind",
    "MergeRole",
    "PhotoKind",
    "PhotoStatus",
    "ProjectStatus",
    "SnapshotKind",
    "StyleSampleStatus",
]


class ProjectStatus(enum.StrEnum):
    NEW = "new"
    IMPORTING = "importing"
    CULLING = "culling"
    ANALYZING = "analyzing"
    REVIEWING = "reviewing"
    EXPORTING = "exporting"
    DONE = "done"


class CullingMode(enum.StrEnum):
    CONSERVATIVE = "conservative"
    TARGET_PERCENT = "target_percent"
    TARGET_COUNT = "target_count"


class ExportConflict(enum.StrEnum):
    ASK = "ask"
    RENAME = "rename"
    OVERWRITE = "overwrite"
    SKIP = "skip"


class PhotoStatus(enum.StrEnum):
    IMPORTED = "imported"
    CULLED = "culled"
    ANALYZED = "analyzed"
    PREDICTED = "predicted"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    EXPORTED = "exported"
    FAILED = "failed"


class PhotoKind(enum.StrEnum):
    RAW = "raw"
    MERGED = "merged"


class CullDecidedBy(enum.StrEnum):
    AUTO = "auto"
    USER = "user"


class EditVersionSource(enum.StrEnum):
    PREDICTED = "predicted"
    USER_EDITED = "user_edited"
    CLUSTER_APPLIED = "cluster_applied"
    REVERTED = "reverted"
    SNAPSHOT_RESTORED = "snapshot_restored"


class CropDecision(enum.StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"


class MergeKind(enum.StrEnum):
    HDR = "hdr"
    PANORAMA = "panorama"
    FOCUS_STACK = "focus_stack"


class MergeDecision(enum.StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class MergeRole(enum.StrEnum):
    MEMBER = "member"
    REFERENCE = "reference"


class StyleSampleStatus(enum.StrEnum):
    """Where a training pair is on its way into a profile (section 8.2)."""

    PENDING = "pending"  # queued for inversion
    READY = "ready"  # inverted, used for training
    #: Inverted, but the pipeline could not reproduce the edit closely enough
    #: (section 8.2.5): kept, shown, not trained on.
    UNREPRODUCIBLE = "unreproducible"
    FAILED = "failed"  # unreadable, or the reference is not that shot
    #: A correction the user made in the review of a project (section 9.2.4),
    #: waiting for their explicit go-ahead before it joins the training set.
    PROPOSED = "proposed"


class SnapshotKind(enum.StrEnum):
    MANUAL = "manual"
    AUTO = "auto"


class JobKind(enum.StrEnum):
    PROXY = "proxy"
    #: The culling analysis of one photo, on its embedded preview (section 7.2).
    #: Not in the sketch of section 11, which predates the decision that
    #: culling never decodes a RAW and so cannot share the ``analyze`` job.
    CULL = "cull"
    ANALYZE = "analyze"
    PREDICT = "predict"
    #: Prepare and invert one training pair of a style profile (section 8.2).
    STYLE_PAIR = "style_pair"
    RENDER_PREVIEW = "render_preview"
    EXPORT = "export"
    #: A subject for a mask, found by a segmentation model on request (6.3).
    SEGMENT = "segment"
    #: A multi-shot merge at full resolution, once the user accepted it (25).
    MERGE = "merge"
    #: Its quick preview at 1024 px, which the user waits for (section 25.6).
    MERGE_PREVIEW = "merge_preview"
    #: The panorama search of one project, on the thumbnails (section 25.2).
    DETECT_MERGES = "detect_merges"
    #: The magic eraser's fills of one photo, computed on the proxy
    #: (docs/SPEC_rimozione.md 4.4). Only ever from a gesture, never in a batch.
    RETOUCH_FILL = "retouch_fill"
    #: "Copia punti sulle foto selezionate": the spots of one photo onto
    #: another, their sources found again there.
    RETOUCH_COPY = "retouch_copy"


class JobState(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


