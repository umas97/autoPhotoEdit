# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``export`` job: one photo of an export batch (phase 8).

Read the item, the frozen settings and the edit; develop and write with no
session open (``export/run.py``); record what happened. A failure is recorded,
not raised: the queue's retries are for transient trouble, and neither an
unreadable RAW nor a full disk goes away by trying again three times in a
row. The user retries from the Export screen once the cause is gone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import select

from ..db.models import (
    EditVersion,
    ExportBatch,
    ExportBatchState,
    ExportConflict,
    ExportItem,
    ExportItemState,
    JobKind,
    Photo,
    PhotoStatus,
    Project,
    utcnow,
)
from ..db.session import read_setting, session_scope
from ..raw.source import has_raw_file, pixels_of
from .handlers import _maker
from .queue import JobRecord
from .worker import register_handler

__all__ = ["run_export"]

_log = logging.getLogger(__name__)


def _policy(item: ExportItem, batch_policy: ExportConflict) -> ExportConflict:
    """The item's own answer, else the batch's. "Ask" at this point means a
    file appeared after the plan: renamed, the one choice that loses nothing."""
    policy = item.on_conflict or batch_policy
    return ExportConflict.RENAME if policy is ExportConflict.ASK else policy


def _rebuilt(maker, item_id: int, source: Path) -> Path:
    """A merged photo whose intermediate the cache lost: merge it again first."""
    from ..export.run import ExportFailure
    from ..merge.errors import MergeFailure
    from ..raw.source import make_available

    with session_scope(maker) as session:
        photo_id = session.get(ExportItem, item_id).photo_id
    try:
        found = make_available(maker, photo_id, source)
    except MergeFailure as exc:
        raise ExportFailure(str(exc), photo_fault=True) from exc
    if found is None:
        raise ExportFailure("la fusione non è più disponibile", photo_fault=True)
    return found


@register_handler(JobKind.EXPORT)
def run_export(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Payload: ``{"item_id": int}``."""
    from ..export.batch import finish_if_done
    from ..export.run import ExportFailure, ExportWork, export_one
    from ..export.settings import ExportSettings
    from ..pipeline.params import EditParams, neutral_params
    from ..retouch.fills import photo_token
    from .handlers_analysis import lens_override_for

    maker = _maker(record.payload.get("db_path"))
    item_id = int(record.payload["item_id"])
    with session_scope(maker) as session:
        item = session.get(ExportItem, item_id)
        if item is None or item.state is not ExportItemState.QUEUED:
            return
        batch = session.get(ExportBatch, item.batch_id)
        photo = session.get(Photo, item.photo_id)
        if batch is None or batch.state is ExportBatchState.CANCELLED or photo is None:
            return
        project = session.get(Project, batch.project_id)
        settings = ExportSettings.model_validate(batch.settings)
        version = session.get(EditVersion, item.version_id) if item.version_id else None
        params = EditParams.from_dict(version.params) if version else neutral_params()
        reserved = tuple(
            session.scalars(
                select(ExportItem.name).where(
                    ExportItem.batch_id == batch.id, ExportItem.id != item.id
                )
            ).all()
        )
        work = ExportWork(
            source=pixels_of(photo) or Path(""),
            raw_file=has_raw_file(photo),
            filename=photo.filename,
            name=item.name,
            params=params,
            settings=settings,
            policy=_policy(item, settings.on_conflict),
            protected=project.source_dir if project else None,
            reserved=reserved,
            lens_override=lens_override_for(session, photo.lens),
            artist=read_setting(session, "artist") or None,
            copyright=read_setting(session, "copyright") or None,
            shot_at=photo.shot_at,
            orientation=photo.orientation,
            photo_token=photo_token(photo),
        )

    progress(0.05)
    failure: ExportFailure | None = None
    try:
        if not work.source.is_file() and not work.raw_file:
            work.source = _rebuilt(maker, item_id, work.source)
        result = export_one(work)
    except ExportFailure as exc:
        failure = exc
        _log.warning("export di %s non riuscito: %s", work.filename, exc)

    with session_scope(maker) as session:
        item = session.get(ExportItem, item_id)
        if item is None:
            return
        item.finished_at = utcnow()
        if failure is not None:
            item.state = ExportItemState.FAILED
            item.error = str(failure)[:1000]
            if failure.photo_fault:
                photo = session.get(Photo, item.photo_id)
                if photo is not None:
                    photo.status = PhotoStatus.FAILED
                    photo.error = str(failure)[:1000]
        else:
            image = result.image
            item.sidecars = result.sidecars or None
            if image is not None:
                item.outcome = image.outcome
                item.path = str(image.path) if image.path else None
            elif result.sidecars:
                outcomes = {entry["outcome"] for entry in result.sidecars}
                item.outcome = "skipped" if outcomes == {"skipped"} else "written"
            item.state = (
                ExportItemState.SKIPPED if item.outcome == "skipped" else ExportItemState.DONE
            )
        batch = session.get(ExportBatch, item.batch_id)
        if batch is not None:
            session.flush()
            finish_if_done(session, batch)
    progress(1.0)
