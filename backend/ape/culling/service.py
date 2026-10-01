# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Culling, as the catalogue sees it: queue, group, select, confirm.

The per-photo analysis runs in the workers (``jobs/handlers.py``). What needs
the whole project at once lives here and runs in the server, because it is
arithmetic on numbers already in the catalogue:

* **grouping** -- bracketings first, bursts second (section 7.3.4) -- runs when
  photos have been analysed since the last time. It is lazy on purpose: the
  alternative, the last worker to finish triggering it, has a race in it
  (which worker is last?) that a read-time check does not have;
* **selection** runs on every change of mode, target, slider, weight, toggle or
  decision, and writes only the rows whose outcome changed;
* **the user's own decisions** and the confirmation that ends culling are in
  ``decisions.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import String, bindparam, case, cast, func, select, update
from sqlalchemy.orm import Session

from ..db.models import (
    CullDecidedBy,
    Job,
    JobKind,
    JobState,
    Photo,
    PhotoKind,
    PhotoStatus,
    Project,
    ProjectStatus,
)
from ..merge.proposals import propose_in_server, protected_ids
from ..merge.service import order_of, queue_panorama_search
from .burst import Shot, group_bursts
from .features import FEATURES_VERSION
from .select import Candidate, CullingSettings, Decision
from .select import select as run_selection

__all__ = [
    "CullingState",
    "apply_selection",
    "enqueue_culling",
    "ensure_grouped",
    "project_settings",
    "state",
]

_log = logging.getLogger(__name__)


def project_settings(project: Project) -> CullingSettings:
    return CullingSettings.from_values(
        project.culling_mode,
        project.culling_target,
        project.culling_aggressiveness,
        project.culling_weights,
        project.culling_criteria,
    )


def _photos_query(project_id: int):
    return select(Photo).where(
        Photo.project_id == project_id, Photo.kind == PhotoKind.RAW, Photo.missing.is_(False)
    )


def enqueue_culling(session: Session, project: Project, *, force: bool = False) -> int:
    """Queue the analysis of every photo that does not have a current one.

    Idempotent like the proxies: the dedupe key is the photo. Returns how many
    jobs were added.
    """
    from ..jobs.queue import enqueue

    project.culling_enabled = True
    if project.status in (ProjectStatus.NEW, ProjectStatus.IMPORTING):
        project.status = ProjectStatus.CULLING

    added = 0
    for photo in session.scalars(_photos_query(project.id)).all():
        features = photo.culling_features or {}
        current = features.get("version") == FEATURES_VERSION
        if current and not force:
            continue
        payload: dict[str, object] = {"photo_id": photo.id}
        job = enqueue(
            session, JobKind.CULL, payload, project_id=project.id, dedupe_key=f"cull:{photo.id}"
        )
        added += int(job is not None)
    return added


_order = order_of


def ensure_grouped(session: Session, project: Project, *, force: bool = False) -> bool:
    """Group bracketings and bursts if photos were analysed since the last time.

    Returns whether it regrouped. A photo is "analysed but not grouped" when it
    has features and no ``burst_rank``: the analysis job clears the rank, and
    grouping sets it on every photo it has seen.

    Nothing is grouped while analyses of the project are still queued or
    running. Grouping part of a card would record, for good, a bracketing of
    three frames that is really five -- the full one would then overlap it
    and be skipped, and its outer frames judged as single shots (section
    7.3.4). The photos stay marked as ungrouped and the first look after the
    last analysis groups them all at once.
    """
    pending = session.scalar(
        select(func.count(Job.id)).where(
            Job.project_id == project.id,
            Job.kind == JobKind.CULL,
            Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
        )
    )
    if pending:
        return False
    if not force:
        stale = session.scalar(
            select(func.count(Photo.id)).where(
                Photo.project_id == project.id,
                Photo.kind == PhotoKind.RAW,
                Photo.missing.is_(False),
                Photo.culling_features.is_not(None),
                Photo.burst_rank.is_(None),
            )
        )
        if not stale:
            return False

    photos = [p for p in session.scalars(_photos_query(project.id)).all() if p.culling_features]
    protected = protected_ids(session, project.id)

    if project.merge_detection_enabled:
        created = propose_in_server(session, project, photos, protected, order=_order)
        if created:
            _log.info("progetto %d: %d fusioni proposte", project.id, created)
        queue_panorama_search(session, project)

    shots = [
        Shot(
            id=p.id,
            shot_at=p.shot_at,
            order=_order(p),
            signature=(p.culling_features or {}).get("signature"),
            focal_length=p.focal_length,
            camera=p.camera,
        )
        for p in photos
    ]
    group_of: dict[int, int] = {}
    for members in group_bursts(shots, excluded=protected):
        for photo_id in members:
            group_of[photo_id] = members[0]
    for photo in photos:
        photo.burst_group_id = group_of.get(photo.id)
        # Provisional: the selection below puts the real rank in. What matters
        # here is that it is no longer NULL, which is the "grouped" mark.
        photo.burst_rank = 0
    session.flush()
    return True


def _selection_rows(session: Session, project_id: int):
    """Everything the selection reads and compares against, in one query.

    The two fields it needs from the stored features are pulled out by SQLite
    (``json_extract``) rather than by decoding the whole document in Python:
    the features carry a colour histogram and a hash per photo, and decoding
    two thousand of them to read two numbers was, measured, most of the time
    of a slider move.
    """
    features = Photo.culling_features
    return session.execute(
        select(
            Photo.id,
            Photo.sharpness,
            Photo.motion_blur,
            Photo.exposure_score,
            Photo.face_score,
            Photo.aesthetic_score,
            Photo.burst_group_id,
            Photo.culled,
            Photo.cull_decided_by,
            # As stored text: they are only compared, never read, and parsing
            # two thousand timestamps and JSON lists per slider move is
            # measurable for nothing.
            cast(Photo.cull_reasons, String).label("reasons_json"),
            Photo.culling_score,
            Photo.burst_rank,
            cast(Photo.shot_at, String).label("shot_at_text"),
            Photo.filename,
            func.json_extract(features, "$.exposure_side").label("exposure_side"),
            func.json_extract(features, "$.sharpness.laplacian").label("laplacian"),
            func.json_extract(features, "$.sharpness.acuity").label("acuity"),
            features.is_not(None).label("analysed"),
        ).where(
            Photo.project_id == project_id,
            Photo.kind == PhotoKind.RAW,
            Photo.missing.is_(False),
        )
    ).all()


def _candidate(row, protected: set[int]) -> Candidate:
    # Positional unpacking: attribute access by name on a result row costs a
    # dictionary lookup per field, and there are fifteen fields and two
    # thousand rows per slider move.
    (
        photo_id, sharpness, motion_blur, exposure, faces, aesthetic, burst_group,
        culled, decided_by, _reasons, _score, _rank, shot_at, filename,
        exposure_side, laplacian, acuity, analysed,
    ) = row
    return Candidate(
        id=photo_id,
        sharpness=sharpness,
        # The column holds the amount of blur, as its name says; the selection
        # wants every criterion with 1 as good.
        motion=None if motion_blur is None else 1.0 - motion_blur,
        exposure=exposure,
        exposure_side=exposure_side,
        faces=faces,
        aesthetic=aesthetic,
        laplacian=None if laplacian is None else float(laplacian),
        acuity=None if acuity is None else float(acuity),
        burst_group=burst_group,
        user_keep=(not culled) if decided_by is CullDecidedBy.USER else None,
        protected=photo_id in protected,
        analysed=bool(analysed),
        order=(shot_at is None, shot_at or "", filename),
    )


@lru_cache(maxsize=64)
def _reasons_json(reasons: tuple[str, ...]) -> str:
    """How the JSON column spells a list of reasons. There are a dozen distinct
    lists in a whole project, and two thousand rows to compare per move."""
    return json.dumps(list(reasons))


@dataclass(frozen=True)
class CullingState:
    decisions: list[Decision]
    settings: CullingSettings
    #: Photos whose outcome, score or rank differs from what was stored before:
    #: what an interface already holding the previous state needs to hear.
    changed: frozenset[int] = frozenset()

    @property
    def selected(self) -> int:
        return sum(1 for d in self.decisions if not d.culled)


def apply_selection(
    session: Session, project: Project, settings: CullingSettings | None = None
) -> CullingState:
    """Recompute the selection from stored scores and write what changed.

    ``settings`` replaces the project's own and is persisted with it; ``None``
    reuses what the project already has.
    """
    if settings is not None:
        project.culling_mode = settings.mode
        project.culling_target = settings.target
        project.culling_aggressiveness = settings.aggressiveness
        project.culling_weights = dict(settings.weights)
        project.culling_criteria = dict(settings.criteria)
    current = settings or project_settings(project)

    session.flush()
    rows = _selection_rows(session, project.id)
    protected = protected_ids(session, project.id)
    decisions = run_selection([_candidate(row, protected) for row in rows], current)

    existing = {row.id: row for row in rows}
    changes = []
    for decision in decisions:
        row = existing[decision.id]
        decided_by = row.cull_decided_by
        if decided_by is not CullDecidedBy.USER and row.analysed:
            decided_by = CullDecidedBy.AUTO
        rank = decision.rank if row.burst_rank is not None else None
        if (
            row.culled != decision.culled
            or (row.reasons_json or "[]") != _reasons_json(decision.reasons)
            or row.culling_score != decision.score
            or row.burst_rank != rank
            or row.cull_decided_by != decided_by
        ):
            changes.append(
                {
                    "pid": decision.id,
                    "culled": decision.culled,
                    "cull_reasons": list(decision.reasons),
                    "culling_score": decision.score,
                    "burst_rank": rank,
                    "cull_decided_by": decided_by,
                }
            )
    if changes:
        # One statement, many parameter sets: SQLite runs it as a prepared
        # loop, which is what keeps a weight change on two thousand photos
        # inside the 100 ms of section 14.
        table = Photo.__table__
        session.connection().execute(
            update(table)
            .where(table.c.id == bindparam("pid"))
            .values(
                culled=bindparam("culled"),
                cull_reasons=bindparam("cull_reasons"),
                culling_score=bindparam("culling_score"),
                # The rank is also the "grouped" mark. A row that is not
                # grouped -- or that a worker re-analysed between this read
                # and this write -- keeps its NULL, so the next look groups
                # it; decided in the statement, so no commit can slip between.
                burst_rank=case(
                    (table.c.burst_rank.is_(None), None), else_=bindparam("burst_rank")
                ),
                cull_decided_by=bindparam("cull_decided_by"),
            ),
            changes,
        )
        session.expire_all()
        if project.status not in _BEFORE_EDITING:
            _develop_newly_kept(session, project, changes)
    return CullingState(
        decisions=decisions,
        settings=current,
        changed=frozenset(change["pid"] for change in changes),
    )


#: Project states in which nothing has been developed yet.
_BEFORE_EDITING = (ProjectStatus.NEW, ProjectStatus.IMPORTING, ProjectStatus.CULLING)


def _develop_newly_kept(session: Session, project: Project, changes: list[dict]) -> None:
    """After "Procedi", a photo the selection now keeps is developed too.

    A change of slider or mode once the editing has started can bring photos
    back into the selection; section 7.5 promises they are usable, not merely
    visible, so their proxies are queued like those of a photo kept by hand.
    """
    from ..jobs.queue import enqueue

    kept = [change["pid"] for change in changes if not change["culled"]]
    if not kept:
        return
    rows = session.execute(
        select(Photo.id).where(Photo.id.in_(kept), Photo.proxy_path.is_(None))
    )
    for (photo_id,) in rows:
        enqueue(
            session, JobKind.PROXY, {"photo_id": photo_id},
            project_id=project.id, dedupe_key=f"proxy:{photo_id}",
        )


def state(session: Session, project: Project) -> dict[str, int]:
    """How far the analysis has got: for the progress line of the culling screen."""
    base = [
        Photo.project_id == project.id,
        Photo.kind == PhotoKind.RAW,
        Photo.missing.is_(False),
    ]
    total = int(session.scalar(select(func.count(Photo.id)).where(*base)) or 0)
    analysed = int(
        session.scalar(
            select(func.count(Photo.id)).where(*base, Photo.culling_features.is_not(None))
        )
        or 0
    )
    failed = int(
        session.scalar(
            select(func.count(Photo.id)).where(*base, Photo.status == PhotoStatus.FAILED)
        )
        or 0
    )
    pending = int(
        session.scalar(
            select(func.count(Job.id)).where(
                Job.project_id == project.id,
                Job.kind == JobKind.CULL,
                Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
            )
        )
        or 0
    )
    return {"total": total, "analysed": analysed, "failed": failed, "pending": pending}
