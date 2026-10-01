# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 7 over HTTP: the review of a project (section 9).

``GET /api/projects/{id}/review`` is the screen: scenes, the individual queue,
counts, the threshold and the weights. Like the project's style, reading it is
what brings it up to date (``review/service.py``) -- predictions applied,
confidence rescored where the edit changed, every photo placed.

Every action answers with an ``undo`` record, which the interface keeps and
posts back to ``/review/undo`` on Ctrl+Z: undoing is itself a set of new
versions (section 23), never a deletion.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db.models import Photo, Project
from ..pipeline.params import EditParams
from ..review import decisions, detail, developed, feedback, service
from ..review.confidence import TERMS
from .deps import get_photo, get_project, get_session

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["review"])


class ReviewSettingsIn(BaseModel):
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    weights: dict[str, float] | None = None


class ParamsIn(BaseModel):
    #: The parameters on screen, when the user changed them; ``None`` keeps the
    #: current version.
    params: dict[str, Any] | None = None
    #: The version of the representative the correction started from: the
    #: editor saves by itself, so the current one may already be the correction.
    base_version_id: int | None = None


class ScenesIn(BaseModel):
    clusters: list[int] = Field(min_length=1, max_length=2000)


class UndoIn(BaseModel):
    undo: list[dict[str, Any]]


class IncorporateIn(BaseModel):
    name: str | None = Field(default=None, max_length=200)


def _params(body: ParamsIn | None) -> EditParams | None:
    if body is None or body.params is None:
        return None
    try:
        return EditParams.model_validate(body.params)
    except Exception as exc:  # pydantic's own message names the field
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"parametri non validi: {exc}") from exc


def _run(action, *args) -> dict:
    try:
        return action(*args)
    except (decisions.ReviewError, feedback.FeedbackError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


def _require_style(project: Project) -> None:
    if project.style_profile_id is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "scegli prima uno stile per il progetto")


@router.get("/projects/{project_id}/review")
def read_review(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    return service.overview(session, project)


@router.put("/projects/{project_id}/review/settings")
def review_settings(
    body: ReviewSettingsIn,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    """Threshold and weights of the confidence (section 9.1: both regolabili in UI)."""
    if body.threshold is not None:
        project.confidence_threshold = body.threshold
    if body.weights is not None:
        unknown = set(body.weights) - set(TERMS)
        if unknown:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"pesi sconosciuti: {', '.join(sorted(unknown))}"
            )
        project.confidence_weights = {
            k: min(1.0, max(0.0, float(v))) for k, v in body.weights.items()
        }
    session.flush()
    return service.overview(session, project)


@router.post("/projects/{project_id}/review/scenes/{cluster}/apply")
def apply_scene(
    cluster: int,
    body: ParamsIn,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    """The representative's correction, propagated to its scene as a delta."""
    _require_style(project)
    params = _params(body)
    if params is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "nessuna correzione da applicare")
    return _run(decisions.apply_to_scene, session, project, cluster, params, body.base_version_id)


@router.post("/projects/{project_id}/review/scenes/{cluster}/approve")
def approve_scene(
    cluster: int,
    body: ParamsIn | None = None,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    _require_style(project)
    base = body.base_version_id if body is not None else None
    return _run(decisions.approve_scene, session, project, cluster, _params(body), base)


@router.post("/projects/{project_id}/review/scenes/approve")
def approve_scenes(
    body: ScenesIn,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    """Approve several scenes as they are, from the grid: one action, one undo."""
    _require_style(project)
    return _run(decisions.approve_scenes, session, project, body.clusters)


@router.post("/projects/{project_id}/review/scenes/{cluster}/reject")
def reject_scene(
    cluster: int, project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    _require_style(project)
    return _run(decisions.reject_scene, session, project, cluster)


@router.post("/projects/{project_id}/review/photos/{photo_id}/approve")
def approve_photo(
    photo_id: int,
    body: ParamsIn | None = None,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    _require_style(project)
    return _run(decisions.approve_photo, session, project, photo_id, _params(body))


@router.post("/projects/{project_id}/review/photos/{photo_id}/reject")
def reject_photo(
    photo_id: int, project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    _require_style(project)
    return _run(decisions.reject_photo, session, project, photo_id)


@router.post("/projects/{project_id}/review/photos/{photo_id}/reset")
def reset_photo(
    photo_id: int, project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    _require_style(project)
    return _run(decisions.reset_photo, session, project, photo_id)


@router.post("/projects/{project_id}/review/undo")
def undo(
    body: UndoIn, project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    return {"restored": decisions.undo(session, project, body.undo)}


@router.get("/photos/{photo_id}/review")
def photo_review(
    photo: Photo = Depends(get_photo), session: Session = Depends(get_session)
) -> dict:
    return detail.photo_review(session, photo)


@router.get("/projects/{project_id}/feedback")
def read_feedback(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    return feedback.summary(session, project)


@router.post("/projects/{project_id}/feedback/incorporate")
def incorporate(
    body: IncorporateIn,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    """The explicit confirmation of section 9.2.4: the corrections join a profile."""
    from ..style.service import summary

    target = _run(feedback.incorporate, session, project, body.name)
    return summary(session, target)


@router.delete("/projects/{project_id}/feedback")
def discard_feedback(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    return {"discarded": feedback.discard(session, project)}


@router.get("/photos/{photo_id}/developed/{version_id}")
def read_developed(photo_id: int, version_id: int) -> Response:
    """A photo's version, developed small (``review/developed.py``).

    The URL names the version, so it always means the same bytes. A 404 is the
    normal state until the render job has run: the interface shows the
    neutral proxy meanwhile.
    """
    path = developed.developed_path(photo_id, version_id)
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "sviluppo non ancora pronto") from exc
    return Response(
        content=payload,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )
