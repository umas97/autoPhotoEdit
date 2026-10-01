# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The segmentation job: one subject of one photo, on request (section 6.3).

A job rather than part of the request because the models belong in a worker:
ONNX Runtime and 88 MB of sky model in the server process would break the
400 MB of section 26 for as long as the program runs.

The result is a mask raster (``masks_store``) and a line in
``Photo.analysis["segments"]``, keyed by subject: the raster's name, the
version of the segmentation, and the proxy it was computed on -- a proxy built
again (a new lens profile) makes it stale. The interface then adds the mask
with that raster to the edit, like any other change.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable

from ..db.models import JobKind, Photo
from ..db.session import session_scope
from .handlers import _maker
from .queue import JobRecord
from .worker import register_handler

__all__ = ["proxy_token", "run_segment", "segment_key"]

_log = logging.getLogger(__name__)


def segment_key(photo_id: int, subject: str) -> str:
    """The dedupe key of a photo's segmentation job, and how the API finds it."""
    return f"segment:{photo_id}:{subject}"


def proxy_token(path: str | None) -> str | None:
    """What the proxy was when a subject was found: path, mtime, size."""
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    token = f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()
    return hashlib.blake2b(token, digest_size=8).hexdigest()


@register_handler(JobKind.SEGMENT)
def run_segment(record: JobRecord, progress: Callable[[float], None]) -> None:
    import cv2
    import numpy as np

    from .. import masks_store
    from ..analysis.segment import SEGMENT_VERSION, segment

    photo_id = int(record.payload["photo_id"])
    subject = str(record.payload["subject"])
    maker = _maker()
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:
            return
        proxy = photo.proxy_path
    if not proxy:
        raise RuntimeError("l'anteprima della foto non è ancora pronta: riprova tra poco")
    token = proxy_token(proxy)
    image = cv2.imread(proxy, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("anteprima della foto illeggibile")
    progress(0.2)
    selection = segment(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), subject)
    progress(0.8)
    raster = np.clip(np.rint(selection * 255.0), 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", raster)
    if not ok:
        raise RuntimeError("codifica della maschera fallita")
    name = masks_store.save(encoded.tobytes())

    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:
            return
        analysis = dict(photo.analysis or {})
        segments = dict(analysis.get("segments") or {})
        segments[subject] = {"raster": name, "version": SEGMENT_VERSION, "proxy": token}
        analysis["segments"] = segments
        photo.analysis = analysis
    _log.info("foto %d: soggetto %s trovato", photo_id, subject)
