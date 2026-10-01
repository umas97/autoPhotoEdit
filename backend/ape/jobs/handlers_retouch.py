# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The removals' jobs: the eraser's fills, and "Copia punti" (docs/SPEC_rimozione.md 4.4, 6).

``retouch_fill`` makes the missing fills of one photo's *current* version, in
the order of its removals, on the proxy (``retouch/fills.compute_fills``). A
worker, never the server: the classic engine holds a few hundred megabytes of
patches at its peak, the model more.

A gesture made while it works wins. Before each fill the job looks at the
photo's current version: if it changed, it stops and starts again from the
new one. What it made so far is kept -- a fill is valid for its key, and an
undo may want it back -- but nothing it made can overwrite a newer result:
references are resolved by key, never written by the job. A dozen rounds at
most, so a user dragging a slider for a minute cannot keep a worker forever.

The patch arriving is not a gesture: no ``EditVersion`` is written. The
editor sees the job end through the progress socket and asks for the states
and a preview again, and the server resolves the new patch in both.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from ..db.models import EditVersion, JobKind, Photo
from ..db.session import session_scope
from .handlers import _maker
from .queue import JobRecord
from .worker import register_handler

__all__ = ["run_retouch_copy", "run_retouch_fill"]

_log = logging.getLogger(__name__)

_MAX_ROUNDS = 12


def _current(session, photo_id: int):
    from sqlalchemy import select

    from ..pipeline.params import EditParams

    version = session.scalars(
        select(EditVersion).where(
            EditVersion.photo_id == photo_id, EditVersion.is_current.is_(True)
        )
    ).first()
    if version is None:
        return None, None
    return version.id, EditParams.from_dict(version.params)


@register_handler(JobKind.RETOUCH_FILL)
def run_retouch_fill(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Payload: ``{"photo_id": int}``."""
    from ..raw.proxy import editing_proxy
    from ..raw.source import make_available, pixels_of
    from ..retouch.fills import (
        FillUnavailable,
        compute_fills,
        item_keys,
        lens_key,
        missing,
        photo_token,
        upstream_base,
    )
    from ..retouch.service import record_error
    from .handlers_analysis import lens_override_for

    photo_id = int(record.payload["photo_id"])
    maker = _maker(record.payload.get("db_path"))
    decoded = None
    for _round in range(_MAX_ROUNDS):
        with session_scope(maker) as session:
            photo = session.get(Photo, photo_id)
            if photo is None:
                return
            version_id, params = _current(session, photo_id)
            if params is None:
                return
            token = photo_token(photo)
            source = pixels_of(photo)
            override = lens_override_for(session, photo.lens)
            failed = dict((photo.analysis or {}).get("retouch_errors") or {})
        if source is None:
            raise RuntimeError("la foto non ha un file da cui calcolare le rimozioni")
        if decoded is None:
            if not source.is_file():
                source = make_available(maker, photo_id, source) or source
            decoded = editing_proxy(source, lens_override=override)
        base = upstream_base(token, lens_key(params, decoded))
        keys = item_keys(params, base)
        todo = {
            item: key for item, key in missing(keys).items()
            if (failed.get(item) or {}).get("key") != key
        }
        if not todo:
            progress(1.0)
            return

        def proceed(expected: int | None = version_id) -> bool:
            with session_scope(maker) as session:
                return _current(session, photo_id)[0] == expected

        try:
            compute_fills(decoded, params, base, progress=progress, proceed=proceed)
        except FillUnavailable as exc:
            # Not a transient failure: retrying will not bring the model
            # back. Recorded on the photo, shown on the removal, until "Riprova".
            with session_scope(maker) as session:
                photo = session.get(Photo, photo_id)
                if photo is not None:
                    for item, key in todo.items():
                        record_error(photo, item, key, str(exc))
            _log.warning("foto %d: rimozione non calcolata: %s", photo_id, exc)
            return
    _log.info("foto %d: rimozioni rimaste in coda dopo %d giri", photo_id, _MAX_ROUNDS)


@register_handler(JobKind.RETOUCH_COPY)
def run_retouch_copy(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Payload: ``{"photo_id": int, "items": [...], "source_flip": int}``."""
    from ..retouch.copy import apply_copy

    apply_copy(_maker(record.payload.get("db_path")), record.payload, progress)
