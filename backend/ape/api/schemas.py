# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The shapes the HTTP API speaks in.

Separate from the ORM on purpose. The catalogue holds columns the interface has
no business seeing -- embeddings, hashes, absolute paths outside the project --
and the interface needs fields the catalogue does not store, like how many jobs
are still queued for a project. Mapping between the two here, once, is what
keeps a change of schema from becoming a change of API.

These models are also the source of the OpenAPI document that ``frontend/src/
lib/api.ts`` is generated from (section 5), so every field name here becomes a
field name in TypeScript.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..db.models import (
    CullingMode,
    ExportConflict,
    JobKind,
    JobState,
    PhotoKind,
    PhotoStatus,
    ProjectStatus,
)

__all__ = [
    "ImportRequest",
    "ImportResponse",
    "JobOut",
    "JobSummary",
    "PhotoDetail",
    "PhotoOut",
    "PhotoPage",
    "ProjectCreate",
    "ProjectOut",
    "ProjectUpdate",
    "RemapRequest",
    "VersionOut",
]


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    source_dir: str
    output_dir: str | None = None
    #: Import the folder as part of creating the project. The usual flow.
    import_now: bool = True


class ProjectUpdate(BaseModel):
    """Every field optional: a PATCH changes what it names and nothing else."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    output_dir: str | None = None
    style_profile_id: int | None = None
    coherence_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    culling_enabled: bool | None = None
    culling_mode: CullingMode | None = None
    culling_target: float | None = None
    culling_aggressiveness: float | None = Field(default=None, ge=0.0, le=1.0)
    culling_weights: dict[str, Any] | None = None
    culling_criteria: dict[str, Any] | None = None
    export_template: str | None = None
    export_on_conflict: ExportConflict | None = None
    export_strip_gps: bool | None = None
    merge_detection_enabled: bool | None = None
    status: ProjectStatus | None = None


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source_dir: str
    output_dir: str | None
    status: ProjectStatus
    style_profile_id: int | None
    coherence_lambda: float
    confidence_threshold: float
    culling_enabled: bool
    culling_mode: CullingMode
    culling_target: float | None
    culling_aggressiveness: float
    culling_weights: dict[str, Any]
    culling_criteria: dict[str, Any]
    export_template: str
    export_on_conflict: ExportConflict
    export_strip_gps: bool
    source_missing: bool
    merge_detection_enabled: bool
    created_at: datetime
    updated_at: datetime
    #: Counts the UI needs on the projects screen and would otherwise ask for
    #: with one request per project.
    photo_count: int = 0
    missing_count: int = 0
    pending_jobs: int = 0


class ImportRequest(BaseModel):
    #: Defaults to the project's own source folder.
    folder: str | None = None
    #: Absence from this folder marks a photo missing. Off when adding a second
    #: folder to a project.
    mark_missing: bool = True
    #: Queue the proxies straight away. Off only for scripted imports.
    build_proxies: bool = True
    #: The explicit choice of section 7.1. ``True``: cull first -- the embedded
    #: previews are analysed and no proxy is built until the user confirms.
    #: ``False``: straight to editing. ``None`` keeps the project's own choice,
    #: which is what a re-import wants.
    culling: bool | None = None


class ImportResponse(BaseModel):
    imported: int
    already_present: int
    duplicates: int
    restored: int
    marked_missing: int
    sidecars: int
    subdirectories_ignored: int
    rejected_formats: dict[str, int]
    other_files: int
    failed: list[tuple[str, str]]
    queued_jobs: int
    #: The same facts as a block of Italian, ready to show. The interface shows
    #: this rather than assembling its own sentences from the numbers.
    summary: str


class RemapRequest(BaseModel):
    folder: str


class RemapResponse(BaseModel):
    reattached: int
    by_filename: int
    still_missing: int


class MergeBadge(BaseModel):
    kind: str
    sources: int


class PhotoOut(BaseModel):
    """What the grid needs. Deliberately small: a thousand of these travel at once."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    status: PhotoStatus
    kind: PhotoKind
    missing: bool
    culled: bool
    width: int | None
    height: int | None
    orientation: int | None
    shot_at: datetime | None
    camera: str | None
    lens: str | None
    iso: int | None
    aperture: float | None
    shutter: float | None
    focal_length: float | None
    confidence: float | None
    culling_score: float | None
    burst_group_id: int | None
    cluster_id: int | None
    cluster_rank: int | None = None
    has_proxy: bool = False
    #: Part of the proxy's URL, so a regenerated proxy is fetched again.
    proxy_rev: str | None = None
    #: The camera's embedded preview can be served (``/thumb``): true of every
    #: photo with a file, since it is extracted on demand when not cached.
    has_thumb: bool = False
    error: str | None = None
    #: A frame an accepted merge stands in for (shown only on request).
    superseded: bool = False
    merge_group_id: int | None = None
    #: For a merged photo, the badge of section 25.6: kind and sources.
    merge: MergeBadge | None = None


class VersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    parent_version_id: int | None
    params_version: int
    source: str
    created_at: datetime
    is_current: bool


class CropProposalOut(BaseModel):
    """A crop the analysis suggests. Never applied without the user (section 6.4)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    rect: dict[str, float]
    aspect: str | None
    score: float | None
    decision: str


class PhotoDetail(PhotoOut):
    """One photo, opened. Carries the parameters and the history."""

    path: str | None
    sidecar_jpeg_path: str | None
    duplicate_paths: list[str] | None
    params: dict[str, Any] | None = None
    current_version_id: int | None = None
    versions: list[VersionOut] = Field(default_factory=list)
    #: What the analysis found: lens profile, straightening, as-shot light.
    analysis: dict[str, Any] | None = None
    #: The pending proposal, if any and if the project still wants them.
    crop_proposal: CropProposalOut | None = None
    crop_proposals_paused: bool = False


class PhotoPage(BaseModel):
    items: list[PhotoOut]
    total: int
    offset: int
    limit: int


class ParamsUpdate(BaseModel):
    """A new state of the edit. Validated against ``EditParams`` in the route."""

    params: dict[str, Any]
    #: Where the change came from, for the history of section 23.
    source: str = "user_edited"


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int | None
    kind: JobKind
    state: JobState
    progress: float
    error: str | None
    attempts: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobSummary(BaseModel):
    """Job counts by state, plus what the progress bar needs to draw itself."""

    counts: dict[str, int]
    active: int
    total: int
    #: 0..1 over the jobs of this project that are not yet finished.
    progress: float
