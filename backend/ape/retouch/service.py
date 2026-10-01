# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The removals of a photo as the server sees them: references, states, the queue.

Called by the requests of the editor, which has the photo open: the keys of the
fills need the lens context of a decoded frame, and the preview cache has one.

**States** (the list of the Rimozione panel): a spot is always ``ready`` -- it
is computed at render. An eraser is ``ready`` when the patch of its current key
exists, ``computing`` while the worker makes it, ``stale`` when an old patch is
shown while the new one is made ("da ricalcolare": the lens profile changed, a
removal before it changed), ``error`` when the engine could not run.

**The one automatic recomputation** the prompt allows is here: an eraser whose
key has no patch is queued, whether it was just painted or its patch went
stale. Never in a batch: only for a photo someone is editing, or exporting.
A fill that failed is not queued again until "Riprova".
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import retouch_store
from ..db.models import EditVersion, Job, JobKind, JobState, Photo
from ..pipeline.params import EditParams
from ..pipeline.retouch_params import EraseItem
from .fills import item_keys, lens_key, photo_token, resolve, upstream_base

__all__ = [
    "clear_errors",
    "enqueue_fills",
    "fill_job_key",
    "keys_for",
    "record_error",
    "resolve_for_editor",
    "states",
]

_ERRORS = "retouch_errors"


def fill_job_key(photo_id: int) -> str:
    return f"retouch:{photo_id}"


def keys_for(photo: Photo, params: EditParams, decoded: Any) -> dict[str, str]:
    return item_keys(params, upstream_base(photo_token(photo), lens_key(params, decoded)))


def current_params(session: Session, photo: Photo) -> EditParams | None:
    version = session.scalars(
        select(EditVersion).where(
            EditVersion.photo_id == photo.id, EditVersion.is_current.is_(True)
        )
    ).first()
    return EditParams.from_dict(version.params) if version is not None else None


def resolve_for_editor(
    session: Session, photo: Photo, params: EditParams, decoded: Any
) -> EditParams:
    """``params`` with every eraser's fill resolved (``fills.resolve``)."""
    if not any(isinstance(item, EraseItem) for item in params.retouch):
        return params
    return resolve(params, keys_for(photo, params, decoded), current_params(session, photo))


def _errors(photo: Photo) -> dict[str, dict[str, str]]:
    return dict((photo.analysis or {}).get(_ERRORS) or {})


def record_error(photo: Photo, item_id: str, key: str, message: str) -> None:
    analysis = dict(photo.analysis or {})
    errors = dict(analysis.get(_ERRORS) or {})
    errors[item_id] = {"key": key, "error": message[:500]}
    analysis[_ERRORS] = errors
    photo.analysis = analysis


def clear_errors(photo: Photo) -> None:
    analysis = dict(photo.analysis or {})
    if analysis.pop(_ERRORS, None) is not None:
        photo.analysis = analysis


def _live_job(session: Session, photo: Photo) -> Job | None:
    return session.scalars(
        select(Job).where(
            Job.kind == JobKind.RETOUCH_FILL,
            Job.dedupe_key == fill_job_key(photo.id),
            Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
        )
    ).first()


def enqueue_fills(session: Session, photo: Photo) -> None:
    from ..jobs.queue import enqueue

    enqueue(
        session,
        JobKind.RETOUCH_FILL,
        {"photo_id": photo.id},
        project_id=photo.project_id,
        dedupe_key=fill_job_key(photo.id),
    )


def states(
    session: Session, photo: Photo, params: EditParams, decoded: Any, *, queue: bool = True
) -> list[dict[str, Any]]:
    """The state of each removal of ``params``, queueing the missing fills."""
    keys = keys_for(photo, params, decoded)
    resolved = resolve(params, keys, current_params(session, photo))
    errors = _errors(photo)
    running = _live_job(session, photo) is not None
    out: list[dict[str, Any]] = []
    wanted = False
    for item in resolved.retouch:
        entry: dict[str, Any] = {"id": item.id, "kind": item.kind, "state": "ready", "error": None}
        if isinstance(item, EraseItem):
            key = keys[item.id]
            failed = errors.get(item.id)
            if retouch_store.exists(key):
                pass
            elif failed is not None and failed.get("key") == key:
                entry["state"], entry["error"] = "error", failed.get("error")
            else:
                wanted = True
                entry["state"] = "stale" if item.fill is not None else "computing"
        out.append(entry)
    if wanted and queue and not running:
        enqueue_fills(session, photo)
    return out
