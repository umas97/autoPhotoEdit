# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Project snapshots: a named state of the whole project, and the way back to it.

Section 23.2. A snapshot records, photo by photo, *which version was current*
and what the user had decided -- culling and review -- and nothing else: a
thousand photos cost a few kilobytes and no pixel is copied.

Rolling back never rewinds anything. A photo whose version differs gets a
**new** version with the old parameters (``snapshot_restored``), so the history
of section 23.1 only grows; and before touching anything the current state is
itself snapshotted, which is what makes a rollback undoable -- rolling back to
that snapshot is the undo.

Automatic snapshots mark the passages of the workflow -- end of culling, end of
prediction, end of review -- and only the last :data:`AUTO_KEEP` are kept. A
passage that happens again without anything having changed (the style applied
twice, the last approval undone and redone) does not add a copy of the last one.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db.enums import CullDecidedBy, EditVersionSource, PhotoStatus, SnapshotKind
from .db.models import EditVersion, Photo, Project, ProjectSnapshot, SnapshotEntry
from .pipeline.params import EditParams

__all__ = [
    "AUTO_CULLING",
    "AUTO_KEEP",
    "AUTO_PREDICTION",
    "AUTO_REVIEW",
    "SnapshotError",
    "describe",
    "remove",
    "restore",
    "take",
    "take_auto",
]

_log = logging.getLogger(__name__)

#: Section 23.2: "limitati agli ultimi 10".
AUTO_KEEP = 10

AUTO_CULLING = "Fine cernita"
AUTO_PREDICTION = "Fine predizione"
AUTO_REVIEW = "Fine revisione"

#: States that describe the file, not the user's work: a rollback leaves them.
_KEPT_STATUSES = frozenset({PhotoStatus.FAILED})


class SnapshotError(ValueError):
    """A snapshot operation that cannot be carried out, in words for the UI."""


def _state(session: Session, project: Project) -> list[dict[str, Any]]:
    """Every photo of the project as a snapshot would record it, by photo id."""
    photos = session.scalars(
        select(Photo).where(Photo.project_id == project.id).order_by(Photo.id)
    ).all()
    current = dict(
        session.execute(
            select(EditVersion.photo_id, EditVersion.id).where(
                EditVersion.photo_id.in_([p.id for p in photos]),
                EditVersion.is_current.is_(True),
            )
        ).all()
    )
    return [
        {
            "photo_id": photo.id,
            "edit_version_id": current.get(photo.id),
            "culled": bool(photo.culled),
            "status": photo.status,
            "decisions": {
                "review": photo.review,
                "cull_decided_by": photo.cull_decided_by.value if photo.cull_decided_by else None,
            },
        }
        for photo in photos
    ]


def _entries_of(snapshot: ProjectSnapshot) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "photo_id": e.photo_id,
                "edit_version_id": e.edit_version_id,
                "culled": bool(e.culled),
                "status": e.status,
                "decisions": e.decisions,
            }
            for e in snapshot.entries
        ),
        key=lambda entry: entry["photo_id"],
    )


def _looks(session: Session, entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """The entries by photo, with the *parameters* of their version in place of its id.

    A rollback writes a new version with the old parameters: it is the same
    development under another id, and must compare equal to the snapshot.
    """
    ids = {e["edit_version_id"] for e in entries if e["edit_version_id"] is not None}
    params = (
        dict(
            session.execute(
                select(EditVersion.id, EditVersion.params).where(EditVersion.id.in_(ids))
            ).all()
        )
        if ids
        else {}
    )
    return {
        e["photo_id"]: {
            **{k: v for k, v in e.items() if k != "edit_version_id"},
            "params": json.dumps(params.get(e["edit_version_id"]), sort_keys=True),
        }
        for e in entries
    }


def _differs(recorded: dict[str, Any], now: dict[str, Any] | None) -> bool:
    if now is None:
        return True
    # A snapshot of schema < 8 knows nothing of the decisions: they cannot differ.
    keys = ("params", "culled", "status") + (
        ("decisions",) if recorded["decisions"] is not None else ()
    )
    return any(recorded[key] != now[key] for key in keys)


def _latest(session: Session, project: Project) -> ProjectSnapshot | None:
    return session.scalars(
        select(ProjectSnapshot)
        .where(ProjectSnapshot.project_id == project.id)
        .order_by(ProjectSnapshot.created_at.desc(), ProjectSnapshot.id.desc())
    ).first()


def take(
    session: Session,
    project: Project,
    name: str,
    kind: SnapshotKind = SnapshotKind.MANUAL,
    *,
    spare: int | None = None,
) -> ProjectSnapshot:
    """Record the project as it is now, under ``name``.

    ``spare``: a snapshot the pruning of old automatic ones must not remove --
    the one being rolled back to.
    """
    name = name.strip()
    if not name:
        raise SnapshotError("dai un nome allo snapshot")
    snapshot = ProjectSnapshot(project_id=project.id, name=name[:200], kind=kind)
    snapshot.entries = [SnapshotEntry(**entry) for entry in _state(session, project)]
    session.add(snapshot)
    session.flush()
    if kind is SnapshotKind.AUTO:
        _prune(session, project, spare)
    return snapshot


def take_auto(session: Session, project: Project, name: str) -> ProjectSnapshot | None:
    """An automatic snapshot, unless the project is exactly as the last one left it."""
    latest = _latest(session, project)
    if latest is not None:
        recorded = _looks(session, _entries_of(latest))
        now = _looks(session, _state(session, project))
        if recorded.keys() == now.keys() and not any(
            _differs(r, now[k]) for k, r in recorded.items()
        ):
            return None
    snapshot = take(session, project, name, SnapshotKind.AUTO)
    _log.info("progetto %d: snapshot automatico «%s»", project.id, name)
    return snapshot


def _prune(session: Session, project: Project, spare: int | None = None) -> None:
    autos = session.scalars(
        select(ProjectSnapshot)
        .where(
            ProjectSnapshot.project_id == project.id,
            ProjectSnapshot.kind == SnapshotKind.AUTO,
            ProjectSnapshot.id != (spare if spare is not None else -1),
        )
        .order_by(ProjectSnapshot.created_at.desc(), ProjectSnapshot.id.desc())
    ).all()
    for old in autos[AUTO_KEEP:]:
        session.delete(old)
    session.flush()


def _get(session: Session, project: Project, snapshot_id: int) -> ProjectSnapshot:
    snapshot = session.get(ProjectSnapshot, snapshot_id)
    if snapshot is None or snapshot.project_id != project.id:
        raise SnapshotError("snapshot inesistente per questo progetto")
    return snapshot


def restore(session: Session, project: Project, snapshot_id: int) -> dict[str, Any]:
    """Put the project back as ``snapshot_id`` recorded it; undoable.

    Returns ``restored`` (photos that changed), ``missing`` (photos of the
    snapshot no longer in the catalogue, or whose version is gone) and
    ``undo`` (the id of the snapshot of the state before, to roll back to).
    """
    from .api.routes_photos import add_version

    target = _get(session, project, snapshot_id)
    name = target.name
    entries = _entries_of(target)
    before = take(
        session, project, f"Prima del ripristino di «{name}»", SnapshotKind.AUTO, spare=target.id
    )
    now = _looks(session, _state(session, project))
    recorded = _looks(session, entries)
    restored = missing = 0
    for entry in entries:
        photo = session.get(Photo, entry["photo_id"])
        if photo is None or photo.project_id != project.id:
            missing += 1
            continue
        changed = False
        version_id = entry["edit_version_id"]
        if version_id is not None and recorded[photo.id]["params"] != now[photo.id]["params"]:
            version = session.get(EditVersion, version_id)
            if version is None or version.photo_id != photo.id:
                missing += 1
                continue
            add_version(
                session,
                photo,
                EditParams.from_dict(version.params),
                EditVersionSource.SNAPSHOT_RESTORED,
            )
            changed = True
        if bool(photo.culled) != entry["culled"]:
            photo.culled = entry["culled"]
            changed = True
        decisions = entry["decisions"]
        if decisions is not None:
            review = decisions.get("review")
            decided_by = decisions.get("cull_decided_by")
            if photo.review != review:
                photo.review = review
                changed = True
            if (photo.cull_decided_by.value if photo.cull_decided_by else None) != decided_by:
                photo.cull_decided_by = CullDecidedBy(decided_by) if decided_by else None
                changed = True
        kept = photo.status in _KEPT_STATUSES or entry["status"] in _KEPT_STATUSES
        if not kept and photo.status != entry["status"]:
            photo.status = entry["status"]
            changed = True
        restored += changed
    session.flush()
    _log.info(
        "progetto %d: ripristinato «%s» (%d foto, %d mancanti)",
        project.id,
        name,
        restored,
        missing,
    )
    return {"restored": restored, "missing": missing, "undo": before.id}


def remove(session: Session, project: Project, snapshot_id: int) -> None:
    session.delete(_get(session, project, snapshot_id))
    session.flush()


def describe(session: Session, project: Project) -> list[dict[str, Any]]:
    """The project's snapshots, newest first, with how many photos differ from now."""
    snapshots = session.scalars(
        select(ProjectSnapshot)
        .where(ProjectSnapshot.project_id == project.id)
        .order_by(ProjectSnapshot.created_at.desc(), ProjectSnapshot.id.desc())
    ).all()
    now = _looks(session, _state(session, project))
    counts = dict(
        session.execute(
            select(SnapshotEntry.snapshot_id, func.count())
            .where(SnapshotEntry.snapshot_id.in_([s.id for s in snapshots]))
            .group_by(SnapshotEntry.snapshot_id)
        ).all()
    )
    return [
        {
            "id": snapshot.id,
            "name": snapshot.name,
            "kind": snapshot.kind.value,
            "created_at": snapshot.created_at.isoformat(),
            "photos": counts.get(snapshot.id, 0),
            "differs": sum(
                _differs(entry, now.get(photo_id))
                for photo_id, entry in _looks(session, _entries_of(snapshot)).items()
            ),
        }
        for snapshot in snapshots
    ]
