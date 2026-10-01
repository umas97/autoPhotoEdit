# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Projects: create, list, import, re-point a moved source folder.

The import endpoint is the one with weight behind it. It does the scan and the
catalogue work synchronously -- reading a mebibyte of each file is fast, and the
user is waiting for the count -- then queues one proxy job per photo and
returns. Everything slow happens in the pool afterwards, with progress on the
WebSocket, which is what keeps a thousand-photo import from being a request that
times out.

Deleting a project deletes rows and the project's cache. It never deletes
photographs: section 2 puts the source folder out of reach of the whole
program, and the export folder is the user's, not ours.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import cache
from ..culling.service import enqueue_culling
from ..db.models import Job, JobState, Photo, Project
from ..importer import import_folder, remap_source, scan_folder
from ..jobs.handlers import enqueue_proxies
from ..safety import register_protected_root
from .deps import get_project, get_session
from .schemas import (
    ImportRequest,
    ImportResponse,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
    RemapRequest,
    RemapResponse,
)

__all__ = ["router"]

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _counts(session: Session, project: Project) -> dict[str, int]:
    photos = int(
        session.scalar(select(func.count(Photo.id)).where(Photo.project_id == project.id)) or 0
    )
    missing = int(
        session.scalar(
            select(func.count(Photo.id)).where(
                Photo.project_id == project.id, Photo.missing.is_(True)
            )
        )
        or 0
    )
    pending = int(
        session.scalar(
            select(func.count(Job.id)).where(
                Job.project_id == project.id,
                Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
            )
        )
        or 0
    )
    return {"photo_count": photos, "missing_count": missing, "pending_jobs": pending}


def _as_out(session: Session, project: Project) -> ProjectOut:
    return ProjectOut(**ProjectOut.model_validate(project).model_dump() | _counts(session, project))


@router.get("", response_model=list[ProjectOut])
def list_projects(session: Session = Depends(get_session)) -> list[ProjectOut]:
    projects = session.scalars(select(Project).order_by(Project.updated_at.desc())).all()
    return [_as_out(session, project) for project in projects]


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def create_project(
    payload: ProjectCreate, session: Session = Depends(get_session)
) -> ProjectOut:
    """Create a project and, unless asked not to, import its folder at once.

    Raises:
        HTTPException 400: the source folder does not exist. A project pointing
            at nothing is not a state worth persisting -- unlike a project whose
            folder disappears *later*, which is ``source_missing``.
    """
    source = Path(payload.source_dir).expanduser()
    if not source.is_dir():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"la cartella {source} non esiste o non è una cartella"
        )
    # Arm the guard before the project can own anything (section 2.3).
    register_protected_root(source)

    project = Project(
        name=payload.name,
        source_dir=str(source.resolve()),
        output_dir=str(Path(payload.output_dir).expanduser()) if payload.output_dir else None,
    )
    session.add(project)
    session.flush()

    if payload.import_now:
        import_folder(session, project)
        enqueue_proxies(session, project.id)
    session.flush()
    return _as_out(session, project)


@router.get("/{project_id}", response_model=ProjectOut)
def read_project(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> ProjectOut:
    # Opening a project re-arms the guard: the process may have restarted since
    # the project was created.
    register_protected_root(project.source_dir)
    project.source_missing = not Path(project.source_dir).is_dir()
    return _as_out(session, project)


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(
    payload: ProjectUpdate,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> ProjectOut:
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(project, field, value)
    session.flush()
    return _as_out(session, project)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    background: BackgroundTasks,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> None:
    """Remove a project from the catalogue, and its files from the cache.

    No photograph is touched, and neither are the hand-painted masks: only what
    ``cache.py`` would regenerate.
    """
    files = cache.project_files(session, project.id)
    session.delete(project)
    # Committed here rather than by ``get_session``, which does it after the
    # response has gone: the list the home asks for the moment it reads the 204
    # would otherwise still contain the project.
    session.commit()
    # Thousands of unlinks are not worth keeping the dialog open for.
    background.add_task(cache.remove_project_files, files)


@router.post("/{project_id}/import", response_model=ImportResponse)
def import_into_project(
    payload: ImportRequest,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> ImportResponse:
    """Scan the folder and add what is new (section 15).

    Running this twice in a row is the same as running it once: that is test 12,
    and it is the property the whole importer is built around.
    """
    folder = Path(payload.folder).expanduser() if payload.folder else Path(project.source_dir)
    try:
        summary = import_folder(
            session, project, folder, mark_missing=payload.mark_missing
        )
    except FileNotFoundError as exc:
        project.source_missing = True
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    if payload.culling is not None:
        project.culling_enabled = payload.culling
    if project.culling_enabled:
        # Culling first: the previews are analysed, and nothing is developed
        # until the user confirms a selection (section 7.6).
        queued = enqueue_culling(session, project)
    else:
        queued = enqueue_proxies(session, project.id) if payload.build_proxies else 0
    session.flush()
    return ImportResponse(
        imported=summary.imported,
        already_present=summary.already_present,
        duplicates=summary.duplicates,
        restored=summary.restored,
        marked_missing=summary.marked_missing,
        sidecars=summary.sidecars,
        subdirectories_ignored=summary.subdirectories_ignored,
        rejected_formats=summary.rejected_formats,
        other_files=summary.other_files,
        failed=summary.failed,
        queued_jobs=queued,
        summary=summary.describe(),
    )


@router.get("/{project_id}/scan", response_model=dict)
def preview_scan(project: Project = Depends(get_project), folder: str | None = None) -> dict:
    """What an import *would* find. Reads the folder, writes nothing."""
    target = Path(folder).expanduser() if folder else Path(project.source_dir)
    try:
        scan = scan_folder(target)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return {
        "folder": str(target),
        "raws": len(scan.raws),
        "sidecars": len(scan.sidecars),
        "subdirectories": [p.name for p in scan.subdirectories],
        "rejected_formats": scan.rejected,
        "other_files": scan.other_files,
    }


@router.post("/{project_id}/remap", response_model=RemapResponse)
def remap_project_source(
    payload: RemapRequest,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> RemapResponse:
    """Point the project at the same photos in a new folder (section 15)."""
    try:
        summary = remap_source(session, project, payload.folder)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return RemapResponse(
        reattached=summary.reattached,
        by_filename=summary.by_filename,
        still_missing=summary.still_missing,
    )
