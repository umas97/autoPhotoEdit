# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 5 over HTTP: scenes, lenses, crop proposals, and the two downloads.

The downloads -- the CLIP model (section 17) and the lensfun data (section
19) -- are the only network calls in the program, and both start here, from a
button the user pressed. Neither is ever started by anything else.

Crop proposals are answered here too. Applying one writes a new version of the
photo's parameters, exactly like moving a slider would (section 23); rejecting
two in a row pauses proposals in the project, until the user asks for them
again (section 6.4).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import lensdb, models_registry
from ..analysis import service
from ..config import get_settings
from ..db.models import (
    CropDecision,
    CropProposal,
    EditVersionSource,
    JobKind,
    LensProfileOverride,
    Photo,
    Project,
)
from ..db.session import session_scope
from ..jobs.queue import enqueue
from ..pipeline.params import CropRect
from .deps import get_project, get_session, preview_cache
from .routes_photos import add_version, current_params, read_photo
from .schemas import PhotoDetail

__all__ = ["router"]

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])


class LensOverrideIn(BaseModel):
    lens: str
    maker: str
    model: str


# --- scenes -----------------------------------------------------------------


@router.get("/projects/{project_id}/scenes")
def read_scenes(project: Project = Depends(get_project), session: Session = Depends(get_session)):
    """The scene clusters, clustering first if photos were analysed since."""
    service.enqueue_missing(session, project.id)
    data = service.scenes(session, project.id)
    ok, reason, _entry = models_registry.feature_status("embedding")
    data["embedding"] = {"available": ok, "reason": reason.value if reason else None}
    return data


@router.post("/projects/{project_id}/analysis")
def queue_analysis(
    embeddings: bool = False,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict[str, int]:
    return {"queued": service.enqueue_missing(session, project.id, embeddings=embeddings)}


# --- lenses -----------------------------------------------------------------


@router.get("/projects/{project_id}/lenses")
def read_lenses(project: Project = Depends(get_project), session: Session = Depends(get_session)):
    return {"lenses": service.lens_summary(session, project.id), "lensfun": _lensfun_state()}


@router.get("/lenses/search")
def search_lenses(q: str = Query("", max_length=120)) -> list[dict[str, Any]]:
    return lensdb.search_lenses(q)


def _redo_lens(session: Session, lens: str | None = None, *, unprofiled: bool = False) -> int:
    """Re-render and re-analyse the photos a change of profile affects.

    The proxy carries the lens correction, and the analysis reads the proxy:
    both are redone, in that order, by one forced proxy job each (the proxy
    job chains the analysis). Across projects, because a lens association is
    global.
    """
    query = select(Photo).where(
        Photo.missing.is_(False), Photo.culled.is_(False), Photo.superseded.is_(False)
    )
    if lens is not None:
        query = query.where(Photo.lens == lens)
    queued = 0
    for photo in session.scalars(query).all():
        if not photo.proxy_path:
            continue
        if unprofiled and (photo.analysis or {}).get("lens") is not None:
            continue
        preview_cache.forget(photo.id)
        job = enqueue(
            session, JobKind.PROXY, {"photo_id": photo.id, "force": True},
            project_id=photo.project_id, dedupe_key=f"proxy:{photo.id}",
        )
        queued += int(job is not None)
    return queued


@router.put("/lenses/override")
def set_lens_override(body: LensOverrideIn, session: Session = Depends(get_session)):
    """Associate a lensfun profile with an EXIF lens name, for every project."""
    if not any(
        lens.maker == body.maker and lens.model == body.model
        for lens in (lensdb.database().lenses if lensdb.lensfun_available() else [])
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "profilo lensfun inesistente")
    row = session.get(LensProfileOverride, body.lens)
    if row is None:
        session.add(
            LensProfileOverride(
                lens_model=body.lens, lensfun_maker=body.maker, lensfun_model=body.model
            )
        )
    else:
        row.lensfun_maker, row.lensfun_model = body.maker, body.model
    session.flush()
    return {"queued": _redo_lens(session, body.lens)}


@router.delete("/lenses/override")
def clear_lens_override(lens: str, session: Session = Depends(get_session)):
    row = session.get(LensProfileOverride, lens)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "nessuna associazione per questo obiettivo")
    session.delete(row)
    session.flush()
    return {"queued": _redo_lens(session, lens)}


# --- lensfun data -----------------------------------------------------------

_lensfun_job: dict[str, Any] = {"download": None, "error": None}


def _lensfun_state() -> dict[str, Any]:
    download = _lensfun_job["download"]
    return {
        **lensdb.data_status(),
        "update": None if download is None else {
            "state": download.state.value,
            "fraction": download.fraction,
            "error": _lensfun_job["error"] or download.error,
        },
    }


@router.get("/lensfun")
def read_lensfun() -> dict[str, Any]:
    return _lensfun_state()


@router.post("/lensfun/update")
def update_lensfun() -> dict[str, Any]:
    """Fetch the current lensfun data. The user pressed the button; nothing else does."""
    from ..downloads import Download, DownloadState

    current = _lensfun_job["download"]
    if current is not None and current.state is DownloadState.RUNNING:
        return _lensfun_state()
    archive = get_settings().data_dir / "lensfun" / "version_1.tar.bz2"
    download = Download(url=lensdb.LENSFUN_DATA_URL, destination=archive)
    _lensfun_job.update(download=download, error=None)

    def installed(finished: Download) -> None:
        if finished.state is not DownloadState.DONE:
            return
        try:
            lensdb.install_update(archive)
        except ValueError as exc:
            _lensfun_job["error"] = str(exc)
            return
        finally:
            archive.unlink(missing_ok=True)
        # Lenses that had no profile may have one now.
        with session_scope() as session:
            _redo_lens(session, unprofiled=True)

    download.start(installed)
    return _lensfun_state()


# --- models -----------------------------------------------------------------


def _model_state(entry: models_registry.ModelEntry) -> dict[str, Any]:
    ok, reason, _ = models_registry.feature_status(entry.feature)
    download = models_registry.active_download(entry.feature)
    return {
        "name": entry.name,
        "feature": entry.feature,
        "licence": entry.licence,
        "size_mb": entry.size_mb,
        "url": entry.url,
        "notice": entry.notice,
        "available": ok,
        "reason": reason.value if reason else None,
        "downloadable": entry.sha256 is not None and entry.download_url is not None,
        "download": None if download is None else {
            "state": download.state.value,
            "fraction": download.fraction,
            "error": download.error,
        },
    }


@router.get("/models")
def read_models() -> list[dict[str, Any]]:
    return [_model_state(entry) for entry in models_registry.MODELS]


@router.post("/models/{feature}/download")
def download_model(feature: str) -> dict[str, Any]:
    """Start the download the user asked for. Verified before it is kept (section 17)."""
    from ..downloads import Download, DownloadState

    def finished(download: Download) -> None:
        if download.state is not DownloadState.DONE or feature != "embedding":
            return
        # Every project gets the embeddings it was analysed without.
        with session_scope() as session:
            for project_id in session.scalars(select(Project.id)).all():
                service.enqueue_missing(session, project_id, embeddings=True)

    try:
        models_registry.start_download(feature, on_done=finished)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    entry = models_registry.entry_for(feature)
    assert entry is not None
    return _model_state(entry)


@router.delete("/models/{feature}/download")
def cancel_model_download(feature: str) -> dict[str, Any]:
    download = models_registry.active_download(feature)
    if download is not None:
        download.cancel()
    entry = models_registry.entry_for(feature)
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"nessun modello per {feature!r}")
    return _model_state(entry)


# --- crop proposals ---------------------------------------------------------


def _proposal(session: Session, proposal_id: int) -> tuple[CropProposal, Photo]:
    proposal = session.get(CropProposal, proposal_id)
    if proposal is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "proposta di crop inesistente")
    if proposal.decision is not CropDecision.PENDING:
        raise HTTPException(status.HTTP_409_CONFLICT, "proposta già decisa")
    photo = session.get(Photo, proposal.photo_id)
    assert photo is not None
    return proposal, photo


@router.post("/crop-proposals/{proposal_id}/apply", response_model=PhotoDetail)
def apply_crop(proposal_id: int, session: Session = Depends(get_session)) -> PhotoDetail:
    """"Applica crop proposto": the proposal becomes the crop of a new version."""
    proposal, photo = _proposal(session, proposal_id)
    params = current_params(session, photo)
    params.geometry.crop = CropRect.model_validate(proposal.rect)
    add_version(session, photo, params, EditVersionSource.USER_EDITED)
    service.record_crop_decision(session, proposal, CropDecision.APPLIED)
    session.flush()
    return read_photo(photo=photo, session=session)


@router.post("/crop-proposals/{proposal_id}/reject", response_model=PhotoDetail)
def reject_crop(proposal_id: int, session: Session = Depends(get_session)) -> PhotoDetail:
    proposal, photo = _proposal(session, proposal_id)
    service.record_crop_decision(session, proposal, CropDecision.REJECTED)
    session.flush()
    return read_photo(photo=photo, session=session)


@router.post("/projects/{project_id}/crop-proposals/resume")
def resume_crop_proposals(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict[str, bool]:
    """Undo the pause of two rejections. The proposals already computed come back."""
    project.crop_proposals_paused = False
    return {"paused": False}
