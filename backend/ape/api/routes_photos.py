# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Photos: list them, open one, move its sliders, look at the result.

Two endpoints do the work the interface is judged on.

``GET /api/photos/{id}/proxy`` serves the cached JPEG. The interface asks for
it with the revision the listing reports (``proxy_rev``), so a given URL always
means the same file, and it is served with a long-lived immutable cache header
and an ETag; the grid scrolling past a thousand thumbnails must not cost a
thousand round trips.

``POST /api/photos/{id}/preview`` renders the parameters the user is currently
holding a slider over. It goes through the ``StageRenderer`` kept by
``deps.preview_cache``, which recomputes only the stages downstream of what
changed -- a saturation slider costs a colour pass, not a decode. That is what
section 10's 150 ms is made of.

Saving is separate from previewing on purpose. A preview is a question; a
``PUT`` of the parameters is an answer, and only the answer becomes an
``EditVersion``. Otherwise dragging one slider would write two hundred rows of
history.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..db.models import (
    CropDecision,
    CropProposal,
    EditVersion,
    EditVersionSource,
    Photo,
    PhotoKind,
    PhotoStatus,
    Project,
)
from ..pipeline.params import PARAMS_VERSION, EditParams, neutral_params
from ..pipeline.render import RenderOptions
from .deps import get_photo, get_session, pixels_for_editing, preview_cache
from .photo_out import merge_badges, photo_out, restore_proxies
from .schemas import (
    CropProposalOut,
    ParamsUpdate,
    PhotoDetail,
    PhotoPage,
    VersionOut,
)

__all__ = ["router"]

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["photos"])

#: Long edge of an interactive preview. Section 12 budgets 120 ms for this size
#: from a warm stage cache.
PREVIEW_EDGE = 1024

#: Quality of the preview JPEG. Lower than an export's: it is transient, it is
#: shown at screen size, and every kilobyte is latency the user feels.
PREVIEW_QUALITY = 85


def _current_version(session: Session, photo_id: int) -> EditVersion | None:
    return session.scalars(
        select(EditVersion).where(
            EditVersion.photo_id == photo_id, EditVersion.is_current.is_(True)
        )
    ).first()


def current_params(session: Session, photo: Photo) -> EditParams:
    current = _current_version(session, photo.id)
    return EditParams.from_dict(current.params) if current else neutral_params()


def add_version(
    session: Session, photo: Photo, params: EditParams, source: EditVersionSource
) -> EditVersion:
    """A new current version whose parent is the old current one (section 23)."""
    previous = _current_version(session, photo.id)
    if previous is not None:
        previous.is_current = False
        # Flushed before the insert: the unique index allows one current version
        # per photo at a time, including for the length of this transaction.
        session.flush()
    version = EditVersion(
        photo_id=photo.id,
        parent_version_id=previous.id if previous else None,
        params=params.model_dump(mode="json"),
        params_version=PARAMS_VERSION,
        source=source,
        is_current=True,
    )
    session.add(version)
    return version


@router.get("/projects/{project_id}/photos", response_model=PhotoPage)
def list_photos(
    project_id: int,
    session: Session = Depends(get_session),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=2000),
    include_culled: bool = True,
    include_missing: bool = True,
    include_sources: bool = False,
    status_filter: PhotoStatus | None = Query(default=None, alias="status"),
) -> PhotoPage:
    """One page of a project's photos, in shooting order.

    Shooting order rather than filename order: a burst renamed by the camera's
    counter and a burst reordered by a card change look the same to a person,
    and the time is the thing they remember.
    """
    conditions = [Photo.project_id == project_id]
    if include_sources:
        # "Mostra scatti sorgente" (section 25.1): the frames merges replaced,
        # never a merged photo whose merge was undone.
        conditions.append(or_(Photo.superseded.is_(False), Photo.kind == PhotoKind.RAW))
    else:
        conditions.append(Photo.superseded.is_(False))
    if not include_culled:
        conditions.append(Photo.culled.is_(False))
    if not include_missing:
        conditions.append(Photo.missing.is_(False))
    if status_filter is not None:
        conditions.append(Photo.status == status_filter)

    total = int(session.scalar(select(func.count(Photo.id)).where(*conditions)) or 0)
    photos = session.scalars(
        select(Photo)
        .where(*conditions)
        .order_by(Photo.shot_at.is_(None), Photo.shot_at, Photo.filename, Photo.id)
        .offset(offset)
        .limit(limit)
    ).all()
    restore_proxies(session, list(photos))
    badges = merge_badges(session, [p for p in photos if p.kind is PhotoKind.MERGED])
    return PhotoPage(
        items=[photo_out(photo, badges.get(photo.id)) for photo in photos],
        total=total,
        offset=offset,
        limit=limit,
    )


def _current_layout(params: dict[str, Any]) -> dict[str, Any]:
    """A stored version in today's layout: the editor never meets an old one.

    A version written before the removals existed has no ``retouch``; migrated
    here, it reads as an empty list. Stored as it was, as history must be.
    """
    try:
        return EditParams.from_dict(params).model_dump(mode="json")
    except ValueError:
        return params


@router.get("/photos/{photo_id}", response_model=PhotoDetail)
def read_photo(
    photo: Photo = Depends(get_photo), session: Session = Depends(get_session)
) -> PhotoDetail:
    current = _current_version(session, photo.id)
    versions = session.scalars(
        select(EditVersion)
        .where(EditVersion.photo_id == photo.id)
        .order_by(EditVersion.created_at.desc(), EditVersion.id.desc())
        .limit(50)
    ).all()
    data = photo_out(photo).model_dump()
    project = session.get(Project, photo.project_id)
    paused = bool(project and project.crop_proposals_paused)
    proposal = None
    if not paused:
        proposal = session.scalars(
            select(CropProposal).where(
                CropProposal.photo_id == photo.id,
                CropProposal.decision == CropDecision.PENDING,
            )
        ).first()
    return PhotoDetail(
        **data,
        path=photo.path,
        sidecar_jpeg_path=photo.sidecar_jpeg_path,
        duplicate_paths=photo.duplicate_paths,
        params=_current_layout(current.params) if current else None,
        current_version_id=current.id if current else None,
        versions=[VersionOut.model_validate(v) for v in versions],
        analysis=photo.analysis,
        crop_proposal=CropProposalOut.model_validate(proposal) if proposal else None,
        crop_proposals_paused=paused,
    )


@router.get("/photos/{photo_id}/proxy")
def read_proxy(
    photo: Photo = Depends(get_photo), session: Session = Depends(get_session)
) -> Response:
    """The cached browsing JPEG.

    Raises:
        HTTPException 404: no proxy yet. The interface shows a placeholder and
            waits for the WebSocket to tell it the job finished; a 404 here is
            a normal state during an import, not an error.
    """
    if not photo.proxy_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "anteprima non ancora generata")
    path = Path(photo.proxy_path)
    if not path.is_file():
        restore_proxies(session, [photo])
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "anteprima assente dalla cache: in rigenerazione"
        )

    payload = path.read_bytes()
    return Response(
        content=payload,
        media_type="image/jpeg",
        headers={
            # The interface asks for ``?v=<proxy_rev>``, which changes with the
            # file: a given URL therefore always means the same bytes.
            "Cache-Control": "private, max-age=31536000, immutable",
            "ETag": f'"{hashlib.blake2b(payload, digest_size=8).hexdigest()}"',
        },
    )


@router.post("/photos/{photo_id}/preview")
def render_preview(
    payload: dict[str, Any] | None = None,
    photo: Photo = Depends(get_photo),
    session: Session = Depends(get_session),
    long_edge: int = Query(PREVIEW_EDGE, ge=256, le=4096),
    retouch: bool = Query(True),
) -> Response:
    """Render this photo with the parameters in the body, right now.

    The body is an ``EditParams`` document, or nothing for neutral. The result
    is a JPEG and is never stored: it is what the user is looking at while they
    decide. ``retouch=false`` is the editor's "mostra rimozioni" switch: the
    removals are suspended, the parameters untouched. Each eraser shows the
    best fill there is (``retouch/fills.resolve``).

    Raises:
        HTTPException 400: the parameters do not validate. The message names the
            field, because a slider sending nonsense is a bug worth seeing.
        HTTPException 409: the photo is a merge (phase 11) or has no file.
    """
    from ..export.image import ExportFormat, encode_image

    source = pixels_for_editing(session, photo)
    try:
        params = EditParams.model_validate(payload) if payload else neutral_params()
    except Exception as exc:  # pydantic's own error is the useful one
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"parametri non validi: {exc}") from exc

    # The source is decoded at the size being asked for, so a drag at 1024 px
    # runs the whole pipeline at 1024 px rather than resizing at the end
    # (section 10). Above the proxy size the request is clamped: the interface
    # never shows the full-resolution RAW.
    from ..jobs.handlers_analysis import lens_override_for

    renderer = preview_cache.renderer_for(
        photo.id,
        source,
        long_edge=long_edge,
        lens_override=lens_override_for(session, photo.lens),
    )
    if retouch and params.retouch:
        from ..retouch.service import resolve_for_editor

        params = resolve_for_editor(session, photo, params, renderer.decoded)
    try:
        image = renderer.render(params, RenderOptions(long_edge=long_edge, retouch=retouch))
    except NotImplementedError as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from exc

    data = encode_image(image, ExportFormat.JPEG, quality=PREVIEW_QUALITY)
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.put("/photos/{photo_id}/params", response_model=PhotoDetail)
def save_params(
    payload: ParamsUpdate,
    photo: Photo = Depends(get_photo),
    session: Session = Depends(get_session),
) -> PhotoDetail:
    """Record a new current version of this photo's development (section 23).

    The previous current version becomes this one's parent and stops being
    current. Nothing is overwritten and nothing is deleted, so the history stays
    a tree and "exactly one current version" stays true -- which is what the
    partial unique index in ``models.py`` enforces and test 16 checks.
    """
    try:
        params = EditParams.model_validate(payload.params)
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"parametri non validi: {exc}") from exc

    try:
        source = EditVersionSource(payload.source)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"origine della modifica sconosciuta: {payload.source}"
        ) from exc

    if any(item.kind == "erase" for item in params.retouch):
        params = _with_fills(session, photo, params)
    add_version(session, photo, params, source)
    if photo.status is PhotoStatus.IMPORTED:
        photo.status = PhotoStatus.PREDICTED
    session.flush()
    return read_photo(photo=photo, session=session)


def _with_fills(session: Session, photo: Photo, params: EditParams) -> EditParams:
    """The erasers' fills resolved before the version is written, and the
    missing ones queued: releasing the eraser is the gesture that asks for them."""
    from ..config import get_settings
    from ..jobs.handlers_analysis import lens_override_for
    from ..retouch.fills import missing
    from ..retouch.service import enqueue_fills, keys_for, resolve_for_editor

    renderer = preview_cache.renderer_for(
        photo.id,
        pixels_for_editing(session, photo),
        long_edge=get_settings().proxy_long_edge,
        lens_override=lens_override_for(session, photo.lens),
    )
    resolved = resolve_for_editor(session, photo, params, renderer.decoded)
    if missing(keys_for(photo, params, renderer.decoded)):
        enqueue_fills(session, photo)
    return resolved


@router.post("/photos/{photo_id}/versions/{version_id}/restore", response_model=PhotoDetail)
def restore_version(
    version_id: int,
    photo: Photo = Depends(get_photo),
    session: Session = Depends(get_session),
) -> PhotoDetail:
    """Make an old version current again, as a *new* version (section 23).

    Restoring does not rewind the history: it adds to it. That is what makes an
    undo undoable.
    """
    target = session.get(EditVersion, version_id)
    if target is None or target.photo_id != photo.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "versione inesistente per questa foto")

    previous = _current_version(session, photo.id)
    if previous is not None:
        previous.is_current = False
        session.flush()

    session.add(
        EditVersion(
            photo_id=photo.id,
            parent_version_id=previous.id if previous else None,
            params=target.params,
            params_version=target.params_version,
            source=EditVersionSource.REVERTED,
            is_current=True,
        )
    )
    session.flush()
    return read_photo(photo=photo, session=session)


@router.post("/photos/{photo_id}/reload", response_model=PhotoDetail)
def reload_photo(
    photo: Photo = Depends(get_photo), session: Session = Depends(get_session)
) -> PhotoDetail:
    """Drop this photo from the in-memory editing cache and re-read it.

    For the case where the user changed the file on disk under us, which the
    program does not otherwise expect and must not silently ignore.
    """
    preview_cache.forget(photo.id)
    return read_photo(photo=photo, session=session)
