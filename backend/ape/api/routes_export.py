# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Export screen (section 10.6): settings, plan, batches.

``GET`` of a project's export is the screen itself: the settings, the plan they
make -- how many photos, the live preview of the file name of section 16.1, the
files already in the destination -- and the batches that ran. Changing a
setting returns the new plan, so the screen never shows a preview computed from
settings it no longer has.

Starting a batch answers ``409`` with the colliding photos when the policy is
"ask" and they have no answer yet; the screen shows the dialog of section 16.2
and posts again with the answers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import ExportBatch, ExportBatchState, ExportConflict, Project
from ..db.session import read_setting
from ..export import batch as batches
from ..export import service
from ..export.naming import NameContext, TemplateError, render_name
from ..export.settings import ExportSettings, load_settings, store_settings
from .deps import get_project, get_session

__all__ = ["router"]

router = APIRouter(tags=["export"])

#: Batches listed on the screen: the running one and the few before it.
_RECENT = 5


class ConflictAnswer(BaseModel):
    photo_id: int
    policy: ExportConflict


class StartRequest(BaseModel):
    settings: dict[str, Any] | None = None
    answers: list[ConflictAnswer] = []
    apply_to_all: ExportConflict | None = None


class RetryRequest(BaseModel):
    item_ids: list[int] | None = None


def _merged(project: Project, changes: dict[str, Any] | None) -> ExportSettings:
    current = load_settings(project).model_dump(mode="json")
    try:
        return ExportSettings.model_validate(current | (changes or {}))
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


def _plan(session: Session, project: Project, settings: ExportSettings) -> dict[str, Any]:
    try:
        return {"ok": True, **service.plan(session, project, settings).describe()}
    except service.PlanError as exc:
        return {"ok": False, "error": str(exc), "count": 0, "names": [], "conflicts": []}


def _batches(session: Session, project: Project) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(ExportBatch)
        .where(ExportBatch.project_id == project.id)
        .order_by(ExportBatch.id.desc())
        .limit(_RECENT)
    ).all()
    return [batches.batch_summary(session, b, with_items=True) for b in rows]


def _screen(session: Session, project: Project, settings: ExportSettings) -> dict[str, Any]:
    return {
        "settings": settings.model_dump(mode="json"),
        "plan": _plan(session, project, settings),
        "batches": _batches(session, project),
        "artist": read_setting(session, "artist") or "",
        "copyright": read_setting(session, "copyright") or "",
        "source_dir": project.source_dir,
    }


@router.get("/api/projects/{project_id}/export")
def read_export(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict[str, Any]:
    return _screen(session, project, load_settings(project))


@router.patch("/api/projects/{project_id}/export")
def update_export(
    changes: dict[str, Any],
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Change some settings; the answer is the whole screen, with the new plan."""
    settings = _merged(project, changes)
    store_settings(project, settings)
    session.flush()
    return _screen(session, project, settings)


@router.get("/api/projects/{project_id}/export/name")
def preview_name(
    template: str,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """The name the template gives the first photo, while the user is typing it."""
    settings = load_settings(project)
    photos = service.selection(session, project, settings)
    if not photos:
        return {"name": None, "error": "nessuna foto nel progetto"}
    first = photos[0]
    try:
        name = render_name(
            template,
            NameContext(
                filename=first.filename,
                ext=settings.extension,
                counter=1,
                project=project.name,
                camera=first.camera,
                lens=first.lens,
                iso=first.iso,
                focal_length=first.focal_length,
                shot_at=first.shot_at,
            ),
        )
    except TemplateError as exc:
        return {"name": None, "error": str(exc), "filename": first.filename}
    return {"name": name, "error": None, "filename": first.filename}


@router.post("/api/projects/{project_id}/export", status_code=status.HTTP_201_CREATED)
def start_export(
    payload: StartRequest,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Start a batch.

    Raises:
        HTTPException 400: the plan refuses (no destination, source folder,
            template, nothing to export).
        HTTPException 409: files exist and the policy is "ask"; ``detail``
            carries the photos that need an answer.
    """
    settings = _merged(project, payload.settings)
    try:
        batch = service.start(
            session,
            project,
            settings,
            decisions={a.photo_id: a.policy for a in payload.answers},
            apply_to_all=payload.apply_to_all,
        )
    except service.ConflictsPending as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, {"message": str(exc), "conflicts": exc.conflicts}
        ) from exc
    except service.PlanError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    session.flush()
    return batches.batch_summary(session, batch)


def _batch(session: Session, batch_id: int) -> ExportBatch:
    batch = session.get(ExportBatch, batch_id)
    if batch is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"export {batch_id} inesistente")
    return batch


@router.get("/api/exports/active")
def active_batches(session: Session = Depends(get_session)) -> dict[str, Any]:
    """Batches running or paused, for the job bar and the close dialog."""
    rows = session.scalars(
        select(ExportBatch).where(
            ExportBatch.state.in_((ExportBatchState.RUNNING, ExportBatchState.PAUSED))
        )
    ).all()
    return {"batches": [batches.batch_summary(session, b) for b in rows]}


@router.get("/api/exports/{batch_id}")
def read_batch(batch_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    return batches.batch_summary(session, _batch(session, batch_id), with_items=True)


@router.post("/api/exports/{batch_id}/pause")
def pause_batch(batch_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    batch = _batch(session, batch_id)
    batches.pause(session, batch)
    session.flush()
    return batches.batch_summary(session, batch, with_items=True)


@router.post("/api/exports/{batch_id}/resume")
def resume_batch(batch_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    batch = _batch(session, batch_id)
    batches.resume(session, batch)
    session.flush()
    return batches.batch_summary(session, batch, with_items=True)


@router.post("/api/exports/{batch_id}/cancel")
def cancel_batch(batch_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    batch = _batch(session, batch_id)
    batches.cancel(session, batch)
    session.flush()
    return batches.batch_summary(session, batch, with_items=True)


@router.post("/api/exports/{batch_id}/retry")
def retry_batch(
    batch_id: int, payload: RetryRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    batch = _batch(session, batch_id)
    batches.retry_failed(session, batch, payload.item_ids)
    session.flush()
    return batches.batch_summary(session, batch, with_items=True)
