# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The catalogue of docs/SPEC.md section 11, as one import.

The tables live in five modules, grouped the way the specification groups them
-- photographs and edits, merges, styles, history, jobs -- because section 26
caps a source file at about four hundred lines and the schema is bigger than
that. Nothing else in the codebase needs to know: ``from ..db.models import
Photo`` works, and which file ``Photo`` is declared in stays a detail of this
package.

Importing this module is also what registers every mapper, so anything that
creates tables or resolves a relationship by name must import it rather than one
of the parts.
"""

from __future__ import annotations

from .base import Base, enum_column, utcnow
from .catalog import CropProposal, EditVersion, Photo, Project
from .enums import (
    CropDecision,
    CullDecidedBy,
    CullingMode,
    EditVersionSource,
    ExportConflict,
    JobKind,
    JobState,
    MergeDecision,
    MergeKind,
    MergeRole,
    PhotoKind,
    PhotoStatus,
    ProjectStatus,
    SnapshotKind,
    StyleSampleStatus,
)
from .export import ExportBatch, ExportBatchState, ExportItem, ExportItemState
from .history import ProjectSnapshot, SnapshotEntry
from .merge import MergeGroup, MergeMember
from .style import LensProfileOverride, Setting, StyleProfile, StyleSample
from .work import Job

__all__ = [
    "Base",
    "CropDecision",
    "CropProposal",
    "CullDecidedBy",
    "CullingMode",
    "EditVersion",
    "EditVersionSource",
    "ExportBatch",
    "ExportBatchState",
    "ExportConflict",
    "ExportItem",
    "ExportItemState",
    "Job",
    "JobKind",
    "JobState",
    "LensProfileOverride",
    "MergeDecision",
    "MergeGroup",
    "MergeKind",
    "MergeMember",
    "MergeRole",
    "Photo",
    "PhotoKind",
    "PhotoStatus",
    "Project",
    "ProjectSnapshot",
    "ProjectStatus",
    "Setting",
    "SnapshotEntry",
    "SnapshotKind",
    "StyleSampleStatus",
    "StyleProfile",
    "StyleSample",
    "enum_column",
    "utcnow",
]
