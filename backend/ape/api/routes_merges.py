# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The merges screen over HTTP (section 25.6).

Reading the list is also what runs detection in a project that skipped
culling (``merge/service.py``), the way opening the culling screen groups
bursts: lazily, once the photos it needs have been measured.

Every change of decision is the user's click. Nothing here merges on its own:
"Anteprima" queues a 1024 px preview, "Accetta" queues the full merge, and the
merged photo appears when that job has run.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import (
    Job,
    JobKind,
    JobState,
    MergeDecision,
    MergeGroup,
    MergeKind,
    MergeRole,
    Photo,
    Project,
)
from ..merge import virtual
from ..merge.service import ensure_detected
from .deps import get_project, get_session
from .photo_out import proxy_on_disk, proxy_rev

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["merges"])

_PREVIEW_FILE = re.compile(r"^[0-9a-f]{64}(\.jpg|-coverage\.png)$")

#: Order of the sections of the screen: what needs an answer first.
_ORDER = {
    MergeDecision.PROPOSED: 0,
    MergeDecision.FAILED: 1,
    MergeDecision.ACCEPTED: 2,
    MergeDecision.REJECTED: 3,
}


class GroupIn(BaseModel):
    kind: MergeKind
    photo_ids: list[int] = Field(min_length=2, max_length=64)
    reference_id: int | None = None


class GroupEdit(BaseModel):
    photo_ids: list[int] | None = Field(default=None, min_length=2, max_length=64)
    reference_id: int | None = None
    options: dict[str, Any] | None = None


def _jobs(session: Session, groups: list[MergeGroup]) -> dict[tuple[str, int], Job]:
    kinds = (JobKind.MERGE, JobKind.MERGE_PREVIEW)
    keys = [f"{kind.value}:{g.id}" for g in groups for kind in kinds]
    if not keys:
        return {}
    rows = session.scalars(
        select(Job).where(
            Job.dedupe_key.in_(keys), Job.state.in_((JobState.QUEUED, JobState.RUNNING))
        )
    ).all()
    found = {}
    for job in rows:
        kind, _, group_id = (job.dedupe_key or "").partition(":")
        found[(kind, int(group_id))] = job
    return found


def _job_state(job: Job | None) -> dict | None:
    if job is None:
        return None
    return {"state": job.state.value, "progress": round(float(job.progress or 0.0), 2)}


def _member(photo: Photo | None, member) -> dict:
    return {
        "photo_id": member.photo_id,
        "position": member.position,
        "ev_offset": member.ev_offset,
        "reference": member.role is MergeRole.REFERENCE,
        "filename": photo.filename if photo else None,
        "missing": bool(photo is None or photo.missing),
        "has_thumb": bool(photo and photo.path and not photo.missing),
        "has_proxy": bool(photo and proxy_on_disk(photo.proxy_path)),
        "proxy_rev": proxy_rev(photo.proxy_path) if photo else None,
        "shot_at": photo.shot_at.isoformat() if photo and photo.shot_at else None,
    }


def _group_out(session: Session, group: MergeGroup, jobs: dict) -> dict:
    members = sorted(group.members, key=lambda m: m.position)
    photos = {p.id: p for p in session.scalars(
        select(Photo).where(Photo.id.in_([m.photo_id for m in members]))
    )}
    digest = virtual.recipe_for(session, group).digest()
    report = group.report or {}
    preview = report.get("preview") or {}
    preview_job = jobs.get((JobKind.MERGE_PREVIEW.value, group.id))
    if preview_job is not None:
        preview_state = preview_job.state.value
    elif preview.get("digest") == digest:
        preview_state = "error" if preview.get("error") else "ready"
    else:
        preview_state = "stale" if preview else "none"
    current = preview if preview.get("digest") == digest else {}
    result = session.get(Photo, group.result_photo_id) if group.result_photo_id else None
    return {
        "id": group.id,
        "kind": group.kind.value,
        "decision": group.decision.value,
        "confidence": group.confidence,
        "reasons": (group.detect_reasons or [{}])[0],
        "options": (group.params or {}).get("options") or {},
        "members": [_member(photos.get(m.photo_id), m) for m in members],
        "preview": {
            "state": preview_state,
            "url": f"/api/merges/previews/{current['file']}" if current.get("file") else None,
            "coverage_url": (
                f"/api/merges/previews/{digest}-coverage.png" if current.get("coverage") else None
            ),
            "error": current.get("error"),
            "report": {k: v for k, v in current.items() if k not in ("file", "digest", "error")},
        },
        "merge_job": _job_state(jobs.get((JobKind.MERGE.value, group.id))),
        "full_report": report.get("full"),
        "error": group.error,
        "result_photo_id": result.id if result is not None and not result.superseded else None,
    }


@router.get("/projects/{project_id}/merges")
def list_merges(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    ensure_detected(session, project)
    groups = session.scalars(
        select(MergeGroup).where(MergeGroup.project_id == project.id)
    ).all()
    groups = sorted(groups, key=lambda g: (_ORDER[g.decision], g.id))
    jobs = _jobs(session, list(groups))
    searching = session.scalar(
        select(Job.id).where(
            Job.project_id == project.id,
            Job.kind == JobKind.DETECT_MERGES,
            Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
        )
    )
    return {
        "enabled": project.merge_detection_enabled,
        "searching": searching is not None,
        "groups": [_group_out(session, g, jobs) for g in groups],
    }


def _group(session: Session, group_id: int) -> MergeGroup:
    group = session.get(MergeGroup, group_id)
    if group is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"fusione {group_id} inesistente")
    return group


def _answer(session: Session, group: MergeGroup) -> dict:
    session.flush()
    return _group_out(session, group, _jobs(session, [group]))


@router.post("/projects/{project_id}/merges", status_code=status.HTTP_201_CREATED)
def create_merge(
    body: GroupIn, project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    """A group made by hand from photos selected in the grid (section 25.2)."""
    try:
        group = virtual.create_group(session, project, body.kind, body.photo_ids, body.reference_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    virtual.request_preview(session, group)
    return _answer(session, group)


@router.patch("/merges/{group_id}")
def edit_merge(group_id: int, body: GroupEdit, session: Session = Depends(get_session)) -> dict:
    """"Modifica gruppo": members, reference, options (section 25.6)."""
    group = _group(session, group_id)
    try:
        virtual.edit_group(session, group, body.photo_ids, body.reference_id, body.options)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _answer(session, group)


@router.post("/merges/{group_id}/preview")
def preview_merge(group_id: int, session: Session = Depends(get_session)) -> dict:
    group = _group(session, group_id)
    if group.decision is MergeDecision.REJECTED:
        raise HTTPException(status.HTTP_409_CONFLICT, "questa fusione è stata rifiutata")
    virtual.request_preview(session, group)
    return _answer(session, group)


@router.post("/merges/{group_id}/accept")
def accept_merge(group_id: int, session: Session = Depends(get_session)) -> dict:
    """Queue the full merge. Also "Riprova" on a failed group."""
    group = _group(session, group_id)
    if group.decision is MergeDecision.ACCEPTED and group.result_photo_id is not None:
        return _answer(session, group)
    virtual.accept(session, group)
    return _answer(session, group)


@router.post("/merges/{group_id}/reject")
def reject_merge(group_id: int, session: Session = Depends(get_session)) -> dict:
    group = _group(session, group_id)
    virtual.reject(session, group)
    return _answer(session, group)


@router.post("/merges/{group_id}/undo")
def undo_merge(group_id: int, session: Session = Depends(get_session)) -> dict:
    """Section 25.6: the merged photo leaves, the frames return, the group is
    proposed again. A rejection is undone the same way."""
    group = _group(session, group_id)
    if group.decision is MergeDecision.REJECTED:
        group.decision = MergeDecision.PROPOSED
    else:
        virtual.undo(session, group)
    return _answer(session, group)


@router.get("/merges/previews/{name}")
def read_preview(name: str) -> Response:
    if not _PREVIEW_FILE.match(name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "anteprima inesistente")
    path = get_settings().merge_preview_dir / name
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "anteprima non più in cache")
    return Response(
        content=path.read_bytes(),
        media_type="image/png" if name.endswith(".png") else "image/jpeg",
        # Named by the recipe's digest: a name always means the same bytes.
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )
