# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The analysis of phase 5, as the catalogue sees it.

The per-photo work runs in the workers (``jobs/handlers_analysis.py``). What
needs the whole project runs here, in the server, on numbers already stored:

* **clustering** -- lazily, at the first look after new analyses, and never
  while analyses of the project are still queued: clustering half a card would
  number scenes that the other half then renumbers under the user's eyes;
* **the lens summary** -- which lenses the project uses, which have a lensfun
  profile, which the user associated by hand;
* **enqueueing** -- the analyses a project is missing, e.g. a catalogue from
  before phase 5, or the embeddings after the model was downloaded; and the
  crop proposals of an older build in a ratio no longer proposed.

The photos that count are the ones in the editing flow: not culled, not
missing. A photo recovered from the discards gets a proxy, then an analysis,
then joins a scene at the next look.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    CropDecision,
    CropProposal,
    Job,
    JobKind,
    JobState,
    LensProfileOverride,
    Photo,
    Project,
)
from ..db.workset import editing_set
from .cluster import ClusterItem, cluster_photos
from .embed import decode_embedding
from .scene import decode_features

__all__ = [
    "enqueue_missing",
    "ensure_clustered",
    "lens_summary",
    "pending_analyses",
    "record_crop_decision",
    "retire_stale_proposals",
    "scenes",
]

_log = logging.getLogger(__name__)


def _editing_set(project_id: int):
    return editing_set(project_id)


def pending_analyses(session: Session, project_id: int) -> int:
    return int(
        session.scalar(
            select(func.count(Job.id)).where(
                Job.project_id == project_id,
                Job.kind.in_((JobKind.ANALYZE, JobKind.PROXY)),
                Job.state.in_((JobState.QUEUED, JobState.RUNNING)),
            )
        )
        or 0
    )


def enqueue_missing(session: Session, project_id: int, *, embeddings: bool = False) -> int:
    """Queue the analysis of every photo in the editing flow that lacks one.

    With ``embeddings``, also the embedding of every analysed photo that has
    none -- what the model download triggers. Idempotent: the dedupe key is the
    photo.
    """
    from ..jobs.handlers_analysis import ANALYSIS_VERSION, enqueue_analysis

    added = 0
    for photo in session.scalars(select(Photo).where(*_editing_set(project_id))).all():
        if not photo.proxy_path:
            continue  # the proxy job chains its own analysis
        current = (photo.analysis or {}).get("version") == ANALYSIS_VERSION
        if not current or photo.scene_features is None:
            added += int(enqueue_analysis(session, photo))
        elif embeddings and photo.embedding is None:
            added += int(enqueue_analysis(session, photo, only="embedding"))
    return added


def retire_stale_proposals(session: Session) -> int:
    """Propose again, in the frame's own ratio, what older builds proposed in another.

    Builds before this one proposed 4:5, 16:9 and 1:1 too. A pending proposal
    in one of those is deleted and the photo's crop alone is computed again
    (``analyze`` with ``only="crop"``). Decided proposals are the user's and
    stay as they are, and so does any crop already applied. Idempotent: run at
    every start, it finds nothing the second time.

    Returns:
        How many photos were queued.
    """
    from ..jobs.handlers_analysis import enqueue_analysis
    from .crop import ALLOWED_ASPECTS

    stale = session.scalars(
        select(CropProposal).where(
            CropProposal.decision == CropDecision.PENDING,
            CropProposal.aspect.is_not(None),
            CropProposal.aspect.not_in(ALLOWED_ASPECTS),
        )
    ).all()
    photo_ids = sorted({proposal.photo_id for proposal in stale})
    for proposal in stale:
        session.delete(proposal)
    session.flush()
    queued = 0
    for photo_id in photo_ids:
        photo = session.get(Photo, photo_id)
        if photo is not None and photo.proxy_path:
            queued += int(enqueue_analysis(session, photo, only="crop"))
    if stale:
        _log.info(
            "%d proposte di crop in un rapporto non originale ritirate, %d foto da riproporre",
            len(stale), queued,
        )
    return queued


def ensure_clustered(session: Session, project_id: int, *, force: bool = False) -> bool:
    """Cluster the project if photos were analysed since the last time.

    A photo is "analysed but not clustered" when it has features and no
    ``cluster_id``: the analysis clears it, and clustering sets it on every
    photo it has seen. Returns whether it clustered.
    """
    if pending_analyses(session, project_id):
        return False
    photos = session.scalars(
        select(Photo).where(*_editing_set(project_id), Photo.scene_features.is_not(None))
    ).all()
    if not photos:
        return False
    if not force and all(photo.cluster_id is not None for photo in photos):
        return False

    items = [
        ClusterItem(
            photo_id=photo.id,
            embedding=decode_embedding(photo.embedding),
            features=decode_features(photo.scene_features),
            shot_at=photo.shot_at,
        )
        for photo in photos
    ]
    result = cluster_photos(items)
    for photo in photos:
        photo.cluster_id = result.labels.get(photo.id)
        photo.cluster_rank = result.ranks.get(photo.id)
    _log.info(
        "progetto %d: %d scene su %d foto (%s)",
        project_id, len(set(result.labels.values())), len(photos), result.basis,
    )
    return True


def scenes(session: Session, project_id: int) -> dict[str, Any]:
    """The clusters of a project, representative first, for the interface."""
    ensure_clustered(session, project_id)
    rows = session.execute(
        select(
            Photo.id, Photo.cluster_id, Photo.cluster_rank, Photo.shot_at, Photo.embedding.is_(None)
        )
        .where(*_editing_set(project_id), Photo.cluster_id.is_not(None))
        .order_by(Photo.cluster_id, Photo.cluster_rank)
    ).all()
    groups: dict[int, list[int]] = defaultdict(list)
    without_embedding = 0
    for photo_id, cluster, _rank, _shot, no_embedding in rows:
        groups[cluster].append(photo_id)
        without_embedding += int(no_embedding)
    total = int(
        session.scalar(select(func.count(Photo.id)).where(*_editing_set(project_id))) or 0
    )
    return {
        "clusters": [
            {"cluster": cluster, "representative": members[0], "photos": members}
            for cluster, members in sorted(groups.items())
        ],
        "clustered": len(rows),
        "total": total,
        "pending": pending_analyses(session, project_id),
        # All or nothing: a project clustered on features because some photos
        # lack an embedding says so, and offers to finish the embeddings.
        "basis": "embedding" if rows and without_embedding == 0 else "features",
    }


def lens_summary(session: Session, project_id: int) -> list[dict[str, Any]]:
    """Every lens the project uses: how many photos, which profile, whose choice."""
    rows = session.execute(
        select(Photo.lens, Photo.analysis)
        .where(Photo.project_id == project_id, Photo.missing.is_(False))
    ).all()
    lenses: dict[str | None, dict[str, Any]] = {}
    for lens_name, analysis in rows:
        entry = lenses.setdefault(
            lens_name,
            {"lens": lens_name, "photos": 0, "analysed": 0, "profile": None, "override": None},
        )
        entry["photos"] += 1
        if analysis and "lens" in analysis:
            entry["analysed"] += 1
            found = analysis["lens"]
            # A photo whose optics were corrected upstream says nothing about
            # whether the lens has a profile.
            if found is not None and found.get("source") != "applied":
                entry["profile"] = found
    for entry in lenses.values():
        if entry["lens"]:
            row = session.get(LensProfileOverride, entry["lens"])
            if row is not None:
                entry["override"] = {"maker": row.lensfun_maker, "model": row.lensfun_model}
    return sorted(lenses.values(), key=lambda e: (-e["photos"], e["lens"] or ""))


def record_crop_decision(
    session: Session, proposal: CropProposal, decision: CropDecision
) -> bool:
    """Record the user's answer to a proposal; pause proposals after two rejections.

    Returns whether proposals are now paused in the project. "Two in a row" is
    read from the decision times, across photos: the rule is about the user's
    taste in this project, not about one photo (section 6.4).
    """
    from ..db.base import utcnow

    proposal.decision = decision
    proposal.decided_at = utcnow()
    session.flush()
    photo = session.get(Photo, proposal.photo_id)
    project = session.get(Project, photo.project_id) if photo else None
    if project is None:
        return False
    last_two = session.scalars(
        select(CropProposal.decision)
        .join(Photo, Photo.id == CropProposal.photo_id)
        .where(Photo.project_id == project.id, CropProposal.decided_at.is_not(None))
        .order_by(CropProposal.decided_at.desc(), CropProposal.id.desc())
        .limit(2)
    ).all()
    if len(last_two) == 2 and all(d is CropDecision.REJECTED for d in last_two):
        project.crop_proposals_paused = True
    return bool(project.crop_proposals_paused)
