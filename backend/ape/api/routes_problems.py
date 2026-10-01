# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The Problems panel and the diagnostics bundle over HTTP (section 19).

The bundle is served as a download, never written by the server: where it
goes is the browser's save dialog, like the ``.apestyle`` export. Its plan --
every file, what it holds and its size, with a preview -- is a separate read,
so the interface can show it *before* the zip exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import diagnostics, problems
from .deps import get_session

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["problems"])

#: A log file is up to 5 MB: the plan shows its end, the zip has it all.
_PREVIEW_LINES = 60
_PREVIEW_CHARS = 20_000


class RetryIn(BaseModel):
    #: Both ``None``: everything the panel lists (of ``project_id``, if given).
    photo_ids: list[int] | None = Field(default=None, max_length=10_000)
    job_ids: list[int] | None = Field(default=None, max_length=10_000)
    merge_ids: list[int] | None = Field(default=None, max_length=10_000)
    project_id: int | None = None


@router.get("/problems")
def read_problems(project_id: int | None = None, session: Session = Depends(get_session)) -> dict:
    return problems.overview(session, project_id)


@router.post("/problems/retry")
def retry_problems(body: RetryIn, session: Session = Depends(get_session)) -> dict:
    retried = problems.retry(
        session, photo_ids=body.photo_ids, job_ids=body.job_ids, merge_ids=body.merge_ids,
        project_id=body.project_id,
    )
    return {"retried": retried}


def _preview(entry: diagnostics.Entry) -> str:
    content = entry["content"]
    if entry["name"].startswith("log/"):
        return "\n".join(content.splitlines()[-_PREVIEW_LINES:])
    return content[:_PREVIEW_CHARS]


@router.get("/diagnostics/plan")
def diagnostics_plan(session: Session = Depends(get_session)) -> list[dict]:
    """What the bundle will contain, file by file, before it exists."""
    return [
        {
            "name": entry["name"],
            "description": entry["description"],
            "size": entry["size"],
            "preview": _preview(entry),
        }
        for entry in diagnostics.plan(session)
    ]


@router.get("/diagnostics/bundle.zip")
def diagnostics_bundle(session: Session = Depends(get_session)) -> Response:
    name, payload = diagnostics.build(session)
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "Cache-Control": "no-store",
        },
    )
