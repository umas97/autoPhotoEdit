# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Removals: what the editor asks the server while the user heals and erases.

``POST /api/photos/{id}/retouch/source`` is the automatic source of a spot
(``ops/heal.find_source``), searched on the proxy frame the editor already has
open: ~0.3 s, no model. The answer goes into the parameters, and
the render never searches again.

``POST /api/photos/{id}/retouch/area`` makes an eraser's area out of one of the
photo's masks: its selection *now*, on the proxy, saved as a raster of its own
(R5). Changing the mask afterwards changes nothing about the removal.

``GET /api/photos/{id}/retouch`` is the state of each removal -- ready,
computing, to be recomputed, failed -- and queues the fills that are missing
(``retouch/service.py``); ``POST /api/photos/{id}/retouch/retry`` queues them
again after a failure. ``POST /api/projects/{id}/retouch/copy`` is "Copia punti
sulle foto selezionate" (``retouch/copy.py``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import masks_store
from ..config import get_settings
from ..db.models import Photo
from ..pipeline.params import EditParams
from .deps import get_photo, get_session, pixels_for_editing, preview_cache

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["retouch"])


class SourceRequest(BaseModel):
    #: The parameters on screen. Only the removals before ``index`` are applied
    #: to the frame searched: the spot is looked at as it will be healed.
    params: dict[str, Any]
    index: int = Field(ge=0)
    cx: float = Field(ge=0.0, le=1.0)
    cy: float = Field(ge=0.0, le=1.0)
    radius: float = Field(ge=0.001, le=0.15)
    #: Sources already tried, for "Scegli un'altra sorgente".
    avoid: list[tuple[float, float]] = Field(default_factory=list, max_length=50)


class AreaRequest(BaseModel):
    params: dict[str, Any]
    mask_index: int = Field(ge=0)


def _params(payload: dict[str, Any]) -> EditParams:
    try:
        return EditParams.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"parametri non validi: {exc}") from exc


def _proxy_renderer(session: Session, photo: Photo):
    from ..jobs.handlers_analysis import lens_override_for

    source = pixels_for_editing(session, photo)
    return preview_cache.renderer_for(
        photo.id,
        source,
        long_edge=get_settings().proxy_long_edge,
        lens_override=lens_override_for(session, photo.lens),
    )


@router.post("/photos/{photo_id}/retouch/source")
def automatic_source(
    request: SourceRequest,
    photo: Photo = Depends(get_photo),
    session: Session = Depends(get_session),
) -> dict[str, float]:
    """Where the spot at ``(cx, cy)`` is best copied from."""
    from ..pipeline.ops.heal import find_source
    from ..retouch.service import resolve_for_editor

    params = _params(request.params)
    before = params.model_copy(update={"retouch": params.retouch[: request.index]})
    renderer = _proxy_renderer(session, photo)
    before = resolve_for_editor(session, photo, before, renderer.decoded)
    frame = renderer.through(before, "retouch")
    sx, sy = find_source(
        frame, request.cx, request.cy, request.radius, avoid=[tuple(p) for p in request.avoid]
    )
    return {"sx": round(sx, 6), "sy": round(sy, 6)}


@router.post("/photos/{photo_id}/retouch/area", status_code=status.HTTP_201_CREATED)
def area_from_mask(
    request: AreaRequest,
    photo: Photo = Depends(get_photo),
    session: Session = Depends(get_session),
) -> dict[str, str]:
    """A raster of what mask ``mask_index`` selects now, on the proxy frame.

    Raises:
        HTTPException 400: no such mask, a raster it needs is missing, or it
            selects nothing.
    """
    import cv2

    from ..pipeline.ops.masks import evaluate

    params = _params(request.params)
    if not request.mask_index < len(params.masks):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"maschera {request.mask_index} inesistente"
        )
    mask = params.masks[request.mask_index]
    renderer = _proxy_renderer(session, photo)
    frame = renderer.through(params, "exposure")
    try:
        selection = evaluate(frame, mask, params.tone, masks_store.load)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    raster = np.clip(np.rint(selection.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    if not (raster >= 128).any():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "la maschera non seleziona niente da rimuovere"
        )
    ok, encoded = cv2.imencode(".png", raster)
    if not ok:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "codifica PNG fallita")
    return {"name": masks_store.save(encoded.tobytes())}


@router.post("/photos/{photo_id}/retouch/states")
def retouch_states(
    payload: dict[str, Any],
    photo: Photo = Depends(get_photo),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """The state of each removal of the parameters on screen; queues missing fills.

    A ``POST`` because it takes the parameters the editor holds, which may be
    a gesture ahead of the saved version -- and because it may queue work.
    """
    from ..retouch.ml import available
    from ..retouch.service import states

    params = _params(payload)
    renderer = _proxy_renderer(session, photo)
    ml_ready, ml_reason = available()
    return {
        "items": states(session, photo, params, renderer.decoded),
        "ml": {"available": ml_ready, "reason": ml_reason},
    }


@router.post("/photos/{photo_id}/retouch/retry")
def retry_fills(
    photo: Photo = Depends(get_photo), session: Session = Depends(get_session)
) -> dict[str, bool]:
    """Forget the failures of this photo's fills and queue them again."""
    from ..retouch.service import clear_errors, enqueue_fills

    clear_errors(photo)
    enqueue_fills(session, photo)
    return {"queued": True}


class CopyRequest(BaseModel):
    photo_id: int
    targets: list[int] = Field(min_length=1, max_length=5000)
    include_erase: bool = False


@router.post("/projects/{project_id}/retouch/copy")
def copy_spots(
    project_id: int, request: CopyRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """"Copia punti sulle foto selezionate": one job per photo copied to.

    Raises:
        HTTPException 404: the photo copied from is not in this project.
        HTTPException 400: it has nothing to copy.
    """
    from ..db.models import JobKind
    from ..jobs.queue import enqueue
    from ..retouch.copy import flip_of, sensor_items
    from ..retouch.service import current_params

    origin = session.get(Photo, request.photo_id)
    if origin is None or origin.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "foto di partenza inesistente")
    if not origin.path:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "una fusione non ha un sensore da cui copiare i punti"
        )
    params = current_params(session, origin)
    items = sensor_items(
        list(params.retouch) if params else [], flip_of(origin.path), request.include_erase
    )
    if not items:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "nessun punto da copiare")
    queued, skipped = 0, 0
    for target_id in dict.fromkeys(request.targets):
        target = session.get(Photo, target_id)
        if target is None or target.project_id != project_id or target.id == origin.id:
            continue
        if not target.path:
            skipped += 1  # a merge: no sensor to put the spots on
            continue
        enqueue(
            session, JobKind.RETOUCH_COPY, {"photo_id": target.id, "items": items},
            project_id=project_id,
        )
        queued += 1
    return {"queued": queued, "skipped": skipped, "items": len(items)}
