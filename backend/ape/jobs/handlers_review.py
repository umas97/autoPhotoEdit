# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The renders of phase 7: a correction's thumbnail, and each photo's developed edit.

A pair made from two files carries the 512 px reference the user edited
(``style/sample.py``). A pair made from a correction in the review
(``review/feedback.py``) has no such file: its "reference" is the photo as the
user approved it, rendered here from the RAW at 512 px -- about half a second,
which is why it is a job and not part of the approval the user is waiting on.

The other render is the photo as its current version develops it, small, for
the review's grid and strips (``review/developed.py``).

``render_preview`` is the kind section 11 lists for rendering outside the
editor; the payload says what for.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..db.models import JobKind, StyleSample
from ..db.session import session_scope
from ..raw.source import pixels_of
from .handlers import _maker, available_pixels
from .queue import JobRecord, enqueue
from .worker import register_handler

__all__ = ["THUMBNAIL_EDGE", "enqueue_thumbnail", "run_render_preview"]

_log = logging.getLogger(__name__)

THUMBNAIL_EDGE = 512


def enqueue_thumbnail(session, sample: StyleSample) -> bool:
    return (
        enqueue(
            session,
            JobKind.RENDER_PREVIEW,
            {"sample_id": sample.id},
            project_id=sample.project_id,
            dedupe_key=f"sample_thumbnail:{sample.id}",
        )
        is not None
    )


@register_handler(JobKind.RENDER_PREVIEW)
def run_render_preview(record: JobRecord, progress: Callable[[float], None]) -> None:
    """The renders of phase 7: a pair's thumbnail, or a photo's developed edit."""
    if record.payload.get("purpose") == "developed":
        _render_developed(record, progress)
    else:
        _render_sample_thumbnail(record, progress)


def _render_developed(record: JobRecord, progress: Callable[[float], None]) -> None:
    """One photo's current version at ``developed.DEVELOPED_EDGE`` (``review/developed.py``).

    A version that is no longer current by the time the job runs is skipped:
    its successor has a job of its own.
    """
    from ..db.models import EditVersion, Photo
    from ..export.image import ExportFormat, save_image
    from ..pipeline.params import EditParams
    from ..pipeline.render import RenderOptions, render
    from ..raw.proxy import editing_proxy
    from ..review import developed
    from .handlers_analysis import lens_override_for

    maker = _maker(record.payload.get("db_path"))
    photo_id = int(record.payload["photo_id"])
    version_id = int(record.payload["version_id"])
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        version = session.get(EditVersion, version_id)
        source = pixels_of(photo) if photo is not None else None
        if source is None or version is None or not version.is_current:
            return
        params = EditParams.from_dict(version.params)
        override = lens_override_for(session, photo.lens)
    target = developed.developed_path(photo_id, version_id)
    if target.is_file():
        return
    edge = developed.DEVELOPED_EDGE
    source = available_pixels(maker, photo_id, source)
    decoded = editing_proxy(source, long_edge=edge, lens_override=override)
    progress(0.6)
    image = render(decoded, params, RenderOptions(long_edge=edge))
    # Written under a temporary name and renamed: the server serves whatever
    # file exists under the final one, and must never serve half of it.
    partial = target.with_suffix(".part.jpg")
    save_image(image, partial, ExportFormat.JPEG, quality=85)
    partial.replace(target)
    developed.remove_older(photo_id, target)


def _render_sample_thumbnail(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Render a proposed pair's parameters at 512 px into its thumbnail.

    A pair whose RAW is gone keeps no thumbnail: the interface shows a
    placeholder, and the pair still trains (it needs numbers, not pixels).
    """
    from ..export.image import ExportFormat, encode_image
    from ..pipeline.params import EditParams
    from ..pipeline.render import RenderOptions, render
    from ..raw.metadata import read_optics
    from ..raw.proxy import editing_proxy
    from .handlers_analysis import lens_override_for

    maker = _maker(record.payload.get("db_path"))
    sample_id = int(record.payload["sample_id"])
    with session_scope(maker) as session:
        sample = session.get(StyleSample, sample_id)
        if sample is None or not sample.raw_path:
            return
        source = Path(sample.raw_path)
        params = EditParams.from_dict(sample.params)
        override = (
            lens_override_for(session, read_optics(source).lens_model) if source.is_file() else None
        )
    if not source.is_file():
        _log.info("coppia %d: RAW assente, nessuna miniatura", sample_id)
        return
    decoded = editing_proxy(source, long_edge=THUMBNAIL_EDGE, lens_override=override)
    progress(0.5)
    image = render(decoded, params, RenderOptions(long_edge=THUMBNAIL_EDGE))
    data = encode_image(image, ExportFormat.JPEG, quality=85)
    with session_scope(maker) as session:
        sample = session.get(StyleSample, sample_id)
        # Only if the pair is still the one rendered: a correction changed in
        # the meantime has its own job queued.
        if sample is not None and sample.params == params.model_dump(mode="json"):
            sample.thumbnail = data
