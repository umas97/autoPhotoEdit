# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The merge jobs: the preview a user waits for, and the merge they accepted.

A merge that cannot be made -- frames that do not align, a panorama that does
not stitch -- is not a crash: the group goes to ``failed`` with the reason and
the job ends normally, because the problem is in the shots, not in the program,
and retrying would give the same answer (section 25.5.6). Anything else is a
bug and fails the job with its traceback, like every other handler.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..config import get_settings
from ..db.models import JobKind, MergeDecision, MergeGroup
from ..db.session import session_scope
from .handlers import _maker
from .queue import JobRecord
from .worker import register_handler

__all__ = ["preview_name", "run_detect_merges", "run_merge", "run_merge_preview"]

_log = logging.getLogger(__name__)


def preview_name(digest: str) -> str:
    return f"{digest}.jpg"


def _write_png(path, rgba) -> None:
    import cv2

    from ..safety import guarded_open

    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:  # pragma: no cover - OpenCV always encodes a uint8 image
        raise ValueError("PNG non codificabile")
    with guarded_open(path) as handle:
        handle.write(encoded.tobytes())


@register_handler(JobKind.MERGE_PREVIEW)
def run_merge_preview(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Merge a group at 1024 px and render it neutral (section 25.6).

    Payload: ``{"group_id": int}``. The report lands in ``group.report["preview"]``,
    with the digest of the recipe it shows, so the screen knows when a preview
    is out of date after the group was edited.
    """
    from ..export.image import ExportFormat, save_image
    from ..merge.engine import run_preview
    from ..merge.errors import MergeFailure
    from ..merge.rebuild import merge_threads
    from ..merge.virtual import recipe_for
    from ..pipeline.params import neutral_params
    from ..pipeline.render import RenderOptions, render

    maker = _maker(record.payload.get("db_path"))
    group_id = int(record.payload["group_id"])
    with session_scope(maker) as session:
        group = session.get(MergeGroup, group_id)
        if group is None:
            return
        recipe = recipe_for(session, group)
    digest = recipe.digest()

    entry: dict = {"digest": digest}
    try:
        with merge_threads():
            outcome = run_preview(recipe, progress=lambda f: progress(0.9 * f))
        image = render(outcome.decoded, neutral_params(), RenderOptions(long_edge=1024))
        folder = get_settings().merge_preview_dir
        save_image(image, folder / preview_name(digest), ExportFormat.JPEG, quality=88)
        extra = outcome.report.pop("coverage_image", None)
        if extra is not None:
            _write_png(folder / f"{digest}-coverage.png", extra)
            entry["coverage"] = True
        entry.update(outcome.report, file=preview_name(digest))
    except MergeFailure as exc:
        entry["error"] = str(exc)

    with session_scope(maker) as session:
        group = session.get(MergeGroup, group_id)
        if group is not None:
            group.report = {**(group.report or {}), "preview": entry}
    progress(1.0)


def _reusable(session, group: MergeGroup, recipe):
    """The result of an earlier run of the same recipe, if the cache still has it.

    Accepting a merge that was undone -- the undo of the undo -- then costs
    nothing: same recipe, same file, same merged photo.
    """
    from ..db.models import Photo
    from ..merge.virtual import intermediate_path_for

    path = intermediate_path_for(recipe)
    photo = session.get(Photo, group.result_photo_id) if group.result_photo_id else None
    full = (group.report or {}).get("full")
    if photo is None or full is None or photo.intermediate_path != str(path):
        return None
    if not path.is_file() or not photo.width or not photo.height:
        return None
    return path, full, (photo.width, photo.height)


@register_handler(JobKind.MERGE)
def run_merge(record: JobRecord, progress: Callable[[float], None]) -> None:
    """The full-resolution merge of an accepted group, and its merged photo.

    Payload: ``{"group_id": int}``. A group undone while the job waited is
    left alone.
    """
    from ..merge.errors import MergeFailure
    from ..merge.rebuild import run_recipe
    from ..merge.virtual import recipe_for, record_failure, record_success

    maker = _maker(record.payload.get("db_path"))
    group_id = int(record.payload["group_id"])
    with session_scope(maker) as session:
        group = session.get(MergeGroup, group_id)
        if group is None or group.decision is not MergeDecision.ACCEPTED:
            return
        recipe = recipe_for(session, group)
        reusable = _reusable(session, group, recipe)

    try:
        if reusable is not None:
            path, report, size = reusable
        else:
            path, report, size = run_recipe(recipe, progress=lambda f: progress(0.95 * f))
    except MergeFailure as exc:
        _log.info("fusione %d non riuscita: %s", group_id, exc)
        with session_scope(maker) as session:
            group = session.get(MergeGroup, group_id)
            if group is not None:
                record_failure(session, group, str(exc))
        return
    except MemoryError:
        with session_scope(maker) as session:
            group = session.get(MergeGroup, group_id)
            if group is not None:
                record_failure(session, group, "memoria insufficiente per questa fusione")
        raise

    with session_scope(maker) as session:
        group = session.get(MergeGroup, group_id)
        if group is None or group.decision is not MergeDecision.ACCEPTED:
            # Undone while it ran: the file is a valid cache entry for when the
            # user accepts again, and nothing else needs doing.
            return
        record_success(session, group, path, report, size)
    progress(1.0)


@register_handler(JobKind.DETECT_MERGES)
def run_detect_merges(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Look for panoramas among a project's photos, on their thumbnails (25.2).

    Payload: ``{"project_id": int}``. The thumbnails are the cached camera
    previews, else the browsing proxies, else the preview read from the RAW
    again -- never a decode. New groups send the project's photos back to
    grouping, so culling's bursts are redone without the panorama's frames.
    """
    import cv2
    from sqlalchemy import select, update

    from ..db.models import Photo, PhotoKind, Project
    from ..merge.detect import BracketShot, exposure_brightness
    from ..merge.detect_pano import PanoShot, detect_panoramas
    from ..merge.proposals import protected_ids, record_merges
    from ..merge.service import order_of
    from ..raw.embedded import preview_paths_for, read_embedded_preview

    maker = _maker(record.payload.get("db_path"))
    project_id = int(record.payload["project_id"])
    with session_scope(maker) as session:
        project = session.get(Project, project_id)
        if project is None or not project.merge_detection_enabled:
            return
        photos = session.scalars(
            select(Photo).where(
                Photo.project_id == project_id,
                Photo.kind == PhotoKind.RAW,
                Photo.missing.is_(False),
                Photo.culling_features.is_not(None),
            )
        ).all()
        shots = [
            PanoShot(
                id=p.id, shot_at=p.shot_at, order=order_of(p), camera=p.camera,
                focal_length=p.focal_length,
                brightness=exposure_brightness(
                    BracketShot(p.id, p.shot_at, (), p.focal_length, p.aperture, p.iso,
                                p.shutter, None, None, None)
                ),
            )
            for p in photos
        ]
        files = {
            p.id: (p.hash or f"photo-{p.id}", p.proxy_path, p.path, p.sidecar_jpeg_path)
            for p in photos
        }
        excluded = protected_ids(session, project_id)

    def thumbnail(photo_id: int):
        identity, proxy, path, sidecar = files[photo_id]
        for candidate in (preview_paths_for(identity).grid, proxy):
            if candidate and Path(candidate).is_file():
                image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
                if image is not None:
                    return image
        try:
            return read_embedded_preview(path, sidecar=sidecar).image if path else None
        except (OSError, ValueError):
            return None

    detected = detect_panoramas(shots, thumbnail, excluded=excluded)
    progress(0.9)
    if not detected:
        return
    with session_scope(maker) as session:
        project = session.get(Project, project_id)
        if project is None:
            return
        created = record_merges(session, project, detected, protected_ids(session, project_id))
        if created:
            _log.info("progetto %d: %d panorami proposti", project_id, created)
            session.execute(
                update(Photo)
                .where(Photo.project_id == project_id, Photo.culling_features.is_not(None))
                .values(burst_rank=None)
            )
    progress(1.0)
