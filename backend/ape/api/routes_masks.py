# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Painted masks: upload and read back the rasters of ``masks_store``.

``POST /api/masks/rasters`` takes a PNG in the request body -- the brush canvas
of the mask editor, sent after each stroke -- and answers with its name, which
the interface writes into the mask's definition. Nothing about a photo changes
here: the raster becomes part of an edit only when parameters naming it are
saved, like any other change.

``GET /api/masks/rasters/{name}`` gives it back, so that painting can continue
on a mask drawn in an earlier session. A name is its content, so the answer is
served ``immutable``.

``POST /api/photos/{id}/masks/{index}/selection`` shows what a mask selects:
the pipeline's own selection, straightened and cropped as the preview is, as a
red PNG whose alpha is the selection. The interface lays it over the preview.
It is the only way to see a luminance range or a segmented subject, which the
browser cannot compute, and for the shapes it is the truth the drawn outline
only approximates.

``GET /api/photos/{id}/segments`` and ``POST /api/photos/{id}/segments/{subject}``
are the segmentation on request: whether each subject's model is here, and
whether the subject has been found on this photo; the ``POST`` queues the
``segment`` job (``jobs/handlers_masks.py``) unless the answer is already
there and still matches the proxy.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import masks_store
from ..db.models import Job, JobKind, JobState, Photo
from ..pipeline.params import EditParams
from .deps import get_photo, get_session, pixels_for_editing, preview_cache

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["masks"])

#: The overlay colour, RGB: the red every editor uses for "selected".
_OVERLAY_RGB = (255, 48, 48)


@router.post("/masks/rasters", status_code=status.HTTP_201_CREATED)
async def upload_raster(request: Request) -> dict:
    declared = int(request.headers.get("content-length") or 0)
    if declared > masks_store.MAX_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "maschera troppo grande")
    data = await request.body()
    try:
        name = masks_store.save(data)
    except masks_store.RasterError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return {"name": name}


@router.get("/masks/rasters/{name}")
def read_raster(name: str) -> Response:
    try:
        path = masks_store.path_for(name)
        payload = path.read_bytes()
    except masks_store.RasterError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "maschera non trovata") from exc
    return Response(
        payload,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.post("/photos/{photo_id}/masks/{index}/selection")
def mask_selection(
    index: int,
    payload: dict[str, Any],
    photo: Photo = Depends(get_photo),
    session: Session = Depends(get_session),
    long_edge: int = Query(1024, ge=256, le=4096),
) -> Response:
    """The selection of mask ``index`` of these parameters, as the preview frames it.

    Evaluated whatever the mask adjusts, so a mask just drawn -- every slider
    still at zero -- already shows what it covers. From the preview's stage
    cache: right after a preview, the frame it reads is a checkpoint.

    Raises:
        HTTPException 400: the parameters do not validate, there is no mask
            ``index``, or its raster is missing (``masks_store.RasterError``).
    """
    import cv2
    import numpy as np

    from ..jobs.handlers_analysis import lens_override_for
    from ..pipeline import geometry
    from ..pipeline.filters import resize_long_edge
    from ..pipeline.ops.masks import evaluate

    source = pixels_for_editing(session, photo)
    try:
        params = EditParams.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"parametri non validi: {exc}") from exc
    if not 0 <= index < len(params.masks):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"maschera {index} inesistente")

    renderer = preview_cache.renderer_for(
        photo.id, source, long_edge=long_edge,
        lens_override=lens_override_for(session, photo.lens),
    )
    frame = renderer.through(params, "exposure")
    selection = evaluate(frame, params.masks[index], params.tone, masks_store.load)
    framed = geometry.apply(selection.astype(np.float32), params.geometry)
    framed = resize_long_edge(framed, long_edge)

    alpha = np.clip(np.rint(framed * 255.0), 0, 255).astype(np.uint8)
    red, green, blue = _OVERLAY_RGB
    bgra = np.empty(alpha.shape + (4,), dtype=np.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = blue, green, red, alpha
    ok, encoded = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "codifica PNG fallita")
    return Response(
        encoded.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"}
    )


_SUBJECTS = ("sky", "person", "skin")


def _segment_state(session: Session, photo: Photo, subject: str) -> dict[str, Any]:
    from ..analysis.segment import SEGMENT_VERSION, status
    from ..jobs.handlers_masks import proxy_token, segment_key

    available, reason = status(subject)
    found = ((photo.analysis or {}).get("segments") or {}).get(subject)
    fresh = (
        found is not None
        and found.get("version") == SEGMENT_VERSION
        and found.get("proxy") == proxy_token(photo.proxy_path)
        and masks_store.exists(found["raster"])
    )
    job = session.scalars(
        select(Job)
        .where(Job.kind == JobKind.SEGMENT, Job.dedupe_key == segment_key(photo.id, subject))
        .order_by(Job.id.desc())
    ).first()
    state = "none"
    error = None
    if job is not None and job.state in (JobState.QUEUED, JobState.RUNNING):
        state = job.state.value
    elif fresh:
        state = "ready"
    elif job is not None and job.state is JobState.FAILED:
        state, error = "failed", job.error
    return {
        "subject": subject,
        "state": state,
        "raster": found["raster"] if fresh else None,
        "error": error,
        "available": available,
        "reason": None if reason is None else reason.value,
    }


@router.get("/photos/{photo_id}/segments")
def read_segments(
    photo: Photo = Depends(get_photo), session: Session = Depends(get_session)
) -> list[dict[str, Any]]:
    return [_segment_state(session, photo, subject) for subject in _SUBJECTS]


@router.post("/photos/{photo_id}/segments/{subject}")
def find_segment(
    subject: str, photo: Photo = Depends(get_photo), session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Find ``subject`` on this photo, unless it has been found already.

    Raises:
        HTTPException 404: not a subject this program segments.
        HTTPException 409: its model is not here (download it from the
            Scenes screen), or the photo has no proxy yet.
    """
    from ..jobs.handlers_masks import segment_key
    from ..jobs.queue import enqueue

    if subject not in _SUBJECTS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"soggetto sconosciuto: {subject}")
    current = _segment_state(session, photo, subject)
    if current["state"] in ("ready", "queued", "running"):
        return current
    if not current["available"]:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "il modello di segmentazione non è installato: scaricalo dalla schermata Scene",
        )
    if not photo.proxy_path:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "l'anteprima della foto non è ancora pronta: riprova tra poco"
        )
    enqueue(
        session,
        JobKind.SEGMENT,
        {"photo_id": photo.id, "subject": subject},
        project_id=photo.project_id,
        dedupe_key=segment_key(photo.id, subject),
    )
    session.commit()
    return _segment_state(session, photo, subject)
