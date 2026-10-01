# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Project snapshots over HTTP (section 23.2): list, take, roll back, delete.

The logic is ``snapshots.py``; this module only translates its refusals into
status codes. A rollback answers with the snapshot of the state it replaced:
rolling back to that one is the undo.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import snapshots
from ..db.models import Project
from .deps import get_project, get_session

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["snapshots"])


class SnapshotIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


def _refuse(exc: snapshots.SnapshotError) -> HTTPException:
    missing = "inesistente" in str(exc)
    return HTTPException(
        status.HTTP_404_NOT_FOUND if missing else status.HTTP_400_BAD_REQUEST, str(exc)
    )


@router.get("/projects/{project_id}/snapshots")
def list_snapshots(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> list[dict]:
    return snapshots.describe(session, project)


@router.post("/projects/{project_id}/snapshots", status_code=status.HTTP_201_CREATED)
def take_snapshot(
    body: SnapshotIn,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    try:
        snapshot = snapshots.take(session, project, body.name)
    except snapshots.SnapshotError as exc:
        raise _refuse(exc) from exc
    return {"id": snapshot.id, "name": snapshot.name}


@router.post("/projects/{project_id}/snapshots/{snapshot_id}/restore")
def restore_snapshot(
    snapshot_id: int,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    try:
        return snapshots.restore(session, project, snapshot_id)
    except snapshots.SnapshotError as exc:
        raise _refuse(exc) from exc


@router.delete("/projects/{project_id}/snapshots/{snapshot_id}", status_code=204)
def delete_snapshot(
    snapshot_id: int,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> None:
    try:
        snapshots.remove(session, project, snapshot_id)
    except snapshots.SnapshotError as exc:
        raise _refuse(exc) from exc
