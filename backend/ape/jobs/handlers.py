# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the workers actually do, one function per job kind.

``proxy`` is the job that turns a row created by the importer -- which knows a
path and a hash and nothing else -- into a photo the interface can show: EXIF
read, dimensions known, a 2048 px JPEG in the cache. It is deliberately the
*only* thing standing between an import and a usable grid, so that a thousand
photos become browsable in one pass over the files rather than three.

``cull`` is its counterpart for a project that starts with culling (section
7.1): EXIF read, the camera's embedded preview extracted and scored, and no RAW
decoded at all (section 7.2). The proxies of the photos the user keeps come
later, when they confirm the selection.

Every handler follows the same three rules.

**Read the catalogue, work outside it, write the catalogue.** A session is open
for milliseconds at each end and closed across the second of decoding in
between; sixteen workers holding write transactions through a decode is how a
SQLite database ends up serialising a pool.

**Raise on failure, with a sentence a photographer can read.** The worker turns
that into ``Job.error`` and ``Photo.error``; nothing here catches its own
exceptions to return a status code.

**Never touch the source file.** Decoding opens it read-only; everything written
lands under the cache directory of ``config.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from ..config import get_settings
from ..db.models import Job, JobKind, Photo, PhotoStatus
from ..db.session import engine_for, session_scope
from ..raw.source import make_available, pixels_of
from .queue import JobRecord
from .worker import register_handler

__all__ = ["available_pixels", "run_cull", "run_proxy"]

_log = logging.getLogger(__name__)


def _maker(db_path: str | Path | None = None) -> sessionmaker:
    engine = engine_for(db_path or get_settings().db_path)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@register_handler(JobKind.PROXY)
def run_proxy(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Read the metadata of one photo and build its browsing proxy.

    Payload: ``{"photo_id": int}``, plus an optional ``"force": true`` to
    regenerate a proxy that is already in the cache, or ``"restore": true``
    for one the cache lost (the analysis is not redone).

    Raises:
        LookupError: if the row is gone -- the project was deleted while the
            job was queued, which is not an error worth retrying.
        FileNotFoundError: if the RAW is no longer where the catalogue says.
        ValueError: if LibRaw cannot read the file.
    """
    from ..raw.metadata import read_metadata
    from ..raw.proxy import build_proxy

    maker = _maker(record.payload.get("db_path"))
    photo_id = int(record.payload["photo_id"])

    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:
            raise LookupError(f"foto {photo_id} non più nel catalogo")
        source = pixels_of(photo)
        if source is None:
            raise LookupError(f"foto {photo_id} non ha un file sorgente")
        identity = photo.hash or f"photo-{photo.id}"
        wants_grouping = _wants_grouping(session, photo)
        sidecar = Path(photo.sidecar_jpeg_path) if photo.sidecar_jpeg_path else None

    source = available_pixels(maker, photo_id, source)
    progress(0.1)
    meta = read_metadata(source)
    grouping = _grouping_features(source, sidecar, meta, identity) if wants_grouping else None
    progress(0.3)

    from .handlers_analysis import enqueue_analysis, lens_override_for

    with session_scope(maker) as session:
        override = lens_override_for(session, meta.lens_model)
    try:
        result = build_proxy(
            source, identity, force=bool(record.payload.get("force")), lens_override=override
        )
    except Exception as exc:
        _mark_failed(maker, photo_id, f"{type(exc).__name__}: {exc}")
        raise

    progress(0.9)
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:  # deleted while we worked; nothing to record
            return
        _apply_metadata(photo, meta)
        if grouping is not None and not (photo.culling_features or {}).get("signature"):
            photo.culling_features = grouping
            photo.burst_rank = None  # "not yet looked at by detection"
        photo.proxy_path = str(result.path)
        photo.missing = False
        photo.error = None
        if photo.status is PhotoStatus.FAILED:
            photo.status = PhotoStatus.IMPORTED
        if result.as_shot is not None:
            photo.analysis = {**(photo.analysis or {}), "as_shot": result.as_shot}
        # Phase 5: a photo with a proxy is a photo that can be analysed. The
        # queue's priorities run every proxy before any analysis, so the grid
        # fills first and the scenes arrive after. A proxy the cache quota
        # removed comes back identical (section 20.3): its analysis stands.
        if not (record.payload.get("restore") and photo.analysis):
            enqueue_analysis(session, photo)
    progress(1.0)


def _wants_grouping(session, photo: Photo) -> bool:
    """Whether this proxy job should also gather what merge detection reads.

    Only for a RAW of a project that looks for merges and whose culling has not
    already measured it (section 25.2 runs on the camera's preview).
    """
    from ..db.models import PhotoKind, Project

    if photo.kind is not PhotoKind.RAW or (photo.culling_features or {}).get("signature"):
        return False
    project = session.get(Project, photo.project_id)
    return bool(project and project.merge_detection_enabled)


def _grouping_features(source: Path, sidecar: Path | None, meta, identity: str) -> dict | None:
    """Signature, camera and sharpness map from the embedded preview, or None.

    A preview that cannot be read costs the photo its chance to be found in a
    merge, not its proxy: detection is a convenience (section 25.2).
    """
    from ..merge.features import grouping_features
    from ..raw.embedded import read_embedded_preview, write_preview_cache

    try:
        preview = read_embedded_preview(source, sidecar=sidecar)
        features = grouping_features(preview.image, meta, preview.flip)
        write_preview_cache(preview, identity)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        _log.info("anteprima incorporata illeggibile in %s: %s", source.name, exc)
        return None
    return features


def _apply_metadata(photo: Photo, meta) -> None:
    """Copy what EXIF knows onto the row, keeping what it does not."""
    photo.camera = meta.camera or photo.camera
    photo.lens = meta.lens_model or photo.lens
    photo.iso = meta.iso if meta.iso is not None else photo.iso
    photo.aperture = meta.aperture if meta.aperture is not None else photo.aperture
    photo.shutter = meta.shutter if meta.shutter is not None else photo.shutter
    photo.focal_length = meta.focal_length if meta.focal_length is not None else photo.focal_length
    photo.shot_at = meta.shot_at or photo.shot_at
    photo.orientation = meta.orientation if meta.orientation is not None else photo.orientation
    # The proxy is already upright and already the size it claims to be, but
    # width and height describe the *photo*, which is what the UI lays out its
    # grid from.
    photo.width = meta.width or photo.width
    photo.height = meta.height or photo.height


@register_handler(JobKind.CULL)
def run_cull(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Score one photo on its embedded preview (sections 7.2 and 7.3).

    Payload: ``{"photo_id": int}``. Writes the scores and the features, clears
    the photo's grouping mark so that the project is regrouped at the next
    look, and leaves every decision the user made exactly as it was (test 10).

    Raises:
        LookupError: the row is gone.
        FileNotFoundError: the RAW is no longer where the catalogue says.
        ValueError: the file cannot be read -- recorded on the photo as well,
            which is what the Problems panel lists.
    """
    from ..culling.features import analyse_preview
    from ..raw.embedded import read_embedded_preview, write_preview_cache
    from ..raw.metadata import read_metadata

    maker = _maker(record.payload.get("db_path"))
    photo_id = int(record.payload["photo_id"])

    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:
            raise LookupError(f"foto {photo_id} non più nel catalogo")
        if photo.path is None:
            raise LookupError(f"foto {photo_id} non ha un file sorgente")
        source = Path(photo.path)
        sidecar = Path(photo.sidecar_jpeg_path) if photo.sidecar_jpeg_path else None
        identity = photo.hash or f"photo-{photo.id}"

    if not source.is_file():
        _mark_missing(maker, photo_id)
        raise FileNotFoundError(f"{source.name} non è più nella cartella sorgente")

    try:
        preview = read_embedded_preview(source, sidecar=sidecar)
        progress(0.3)
        meta = read_metadata(source)
        analysis = analyse_preview(preview, meta, flip=preview.flip)
        progress(0.7)
        write_preview_cache(preview, identity)
    except Exception as exc:
        _mark_failed(maker, photo_id, f"{type(exc).__name__}: {exc}")
        raise

    scores = analysis.scores
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:
            return
        _apply_metadata(photo, meta)
        photo.sharpness = scores.focus
        photo.motion_blur = None if scores.motion is None else 1.0 - scores.motion
        photo.exposure_score = scores.exposure
        photo.culling_features = analysis.features
        # "Analysed since the last grouping": the server regroups on its next
        # read. The decision columns are not touched -- a user's decision
        # survives any number of re-analyses.
        photo.burst_rank = None
        photo.culling_score = None
        photo.missing = False
        photo.error = None
        if photo.status in (PhotoStatus.IMPORTED, PhotoStatus.FAILED):
            photo.status = PhotoStatus.CULLED
    progress(1.0)


def available_pixels(maker: sessionmaker, photo_id: int, source: Path) -> Path:
    """The photo's file, rebuilt first if it is a merge the cache lost.

    Raises:
        FileNotFoundError: a RAW that left the source folder; the photo is
            marked missing.
        MergeFailure: a merge that cannot be rebuilt; the photo is marked
            failed with the reason.
    """
    from ..merge.errors import MergeFailure

    try:
        found = make_available(maker, photo_id, source)
    except MergeFailure as exc:
        _mark_failed(maker, photo_id, str(exc))
        raise
    if found is None:
        _mark_missing(maker, photo_id)
        raise FileNotFoundError(f"{source.name} non è più nella cartella sorgente")
    return found


def _mark_missing(maker: sessionmaker, photo_id: int) -> None:
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is not None:
            photo.missing = True
            photo.error = "il file non è più nella cartella sorgente"


def _mark_failed(maker: sessionmaker, photo_id: int, message: str) -> None:
    """Record the failure on the photo as well as on the job.

    Section 13's test 11 asks for a photo in state ``failed`` with a readable
    message; the job's own error is for the Problems panel, and the two are not
    the same audience.
    """
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is not None:
            photo.status = PhotoStatus.FAILED
            photo.error = message[:1000]


def enqueue_proxies(session, project_id: int, *, force: bool = False) -> int:
    """Queue a proxy job for every photo of a project that has no proxy yet.

    Returns how many were added. Idempotent by construction: the dedupe key is
    the photo, so calling this after every import costs nothing.
    """
    from sqlalchemy import select

    from .queue import enqueue

    photos = session.scalars(
        select(Photo).where(Photo.project_id == project_id, Photo.missing.is_(False))
    ).all()
    added = 0
    for photo in photos:
        if photo.proxy_path and not force:
            continue
        payload: dict[str, object] = {"photo_id": photo.id}
        if force:
            payload["force"] = True
        job = enqueue(
            session,
            JobKind.PROXY,
            payload,
            project_id=project_id,
            dedupe_key=f"proxy:{photo.id}",
        )
        added += int(job is not None)
    return added


def pending_for_project(session, project_id: int) -> int:
    """How many jobs of this project are not finished. For the progress bar."""
    from sqlalchemy import func, select

    from ..db.models import JobState

    return int(
        session.scalar(
            select(func.count(Job.id)).where(
                Job.project_id == project_id,
                Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
            )
        )
        or 0
    )
