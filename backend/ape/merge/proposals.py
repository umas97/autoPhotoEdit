# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Merge proposals, as rows (section 25.2).

``detect.py`` finds groups; this module records them, and answers the one
question culling asks of them: which photos are frames of a merge the user has
not turned down, and so must not be judged as single shots (section 25.7).

A proposal is a row the user will answer. The answer is kept: the identity of a
group is its set of members, so a bracketing found again at the next reopening
matches the row that holds the user's "no" and is not proposed a second time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..db.models import (
    MergeDecision,
    MergeGroup,
    MergeKind,
    MergeMember,
    MergeRole,
    Photo,
    Project,
)
from .detect import BracketShot, DetectedMerge, detect_brackets

__all__ = [
    "ACTIVE_MERGES",
    "propose_brackets",
    "propose_in_server",
    "propose_stacks",
    "protected_ids",
    "record_merges",
]

#: Groups whose frames culling leaves alone. A rejected group's frames are
#: ordinary photos again: the user has said they are not a merge. A failed one
#: is still the user's to judge -- retried, edited or rejected.
ACTIVE_MERGES = (MergeDecision.PROPOSED, MergeDecision.ACCEPTED, MergeDecision.FAILED)


def protected_ids(session: Session, project_id: int) -> set[int]:
    rows = session.execute(
        select(MergeMember.photo_id)
        .join(MergeGroup, MergeGroup.id == MergeMember.group_id)
        .where(MergeGroup.project_id == project_id, MergeGroup.decision.in_(ACTIVE_MERGES))
    )
    return {int(row[0]) for row in rows}


def record_merges(
    session: Session, project: Project, detected: Iterable[DetectedMerge], protected: set[int]
) -> int:
    """Store new merge proposals. A group the user already answered stays answered.

    ``protected`` is updated in place with the members of what is created, so
    the caller can go on to exclude them from burst grouping.
    """
    known: set[frozenset[int]] = set()
    groups = session.scalars(select(MergeGroup).where(MergeGroup.project_id == project.id)).all()
    for group in groups:
        known.add(frozenset(member.photo_id for member in group.members))

    created = 0
    for merge in detected:
        members = frozenset(merge.members)
        if members in known or members & protected:
            continue
        group = MergeGroup(
            project_id=project.id,
            kind=MergeKind(merge.kind),
            decision=MergeDecision.PROPOSED,
            confidence=merge.confidence,
            detect_reasons=[merge.reasons],
        )
        for position, (photo_id, offset) in enumerate(
            zip(merge.members, merge.ev_offsets, strict=True)
        ):
            group.members.append(
                MergeMember(
                    photo_id=photo_id,
                    position=position,
                    ev_offset=offset,
                    role=MergeRole.REFERENCE if photo_id == merge.reference else MergeRole.MEMBER,
                )
            )
        session.add(group)
        session.flush()
        session.execute(
            update(Photo).where(Photo.id.in_(list(members))).values(merge_group_id=group.id)
        )
        protected |= members
        known.add(members)
        created += 1
    return created




def propose_brackets(
    session: Session,
    project: Project,
    photos: Sequence[Photo],
    protected: set[int],
    *,
    order: Callable[[Photo], tuple],
) -> int:
    """Detect exposure bracketings among analysed photos and record them."""
    shots = []
    for photo in photos:
        features = photo.culling_features or {}
        camera = features.get("camera") or {}
        shots.append(
            BracketShot(
                id=photo.id,
                shot_at=photo.shot_at,
                order=order(photo),
                focal_length=photo.focal_length,
                aperture=photo.aperture,
                iso=photo.iso,
                shutter=photo.shutter,
                exposure_bias=camera.get("exposure_bias"),
                release_mode=camera.get("release_mode"),
                signature=features.get("signature"),
                camera_bracket=camera.get("camera_bracket"),
            )
        )
    return record_merges(session, project, detect_brackets(shots), protected)


def _focus_position(features: dict) -> int | None:
    value = (features.get("camera") or {}).get("focus_position")
    return int(value) if value is not None else None


def propose_stacks(
    session: Session,
    project: Project,
    photos: Sequence[Photo],
    protected: set[int],
    *,
    order: Callable[[Photo], tuple],
) -> int:
    """Detect focus stacks among analysed photos and record them."""
    from .detect_stack import StackShot, detect_focus_stacks

    shots = []
    for photo in photos:
        if photo.id in protected:
            continue
        features = photo.culling_features or {}
        shots.append(
            StackShot(
                id=photo.id,
                shot_at=photo.shot_at,
                order=order(photo),
                focal_length=photo.focal_length,
                aperture=photo.aperture,
                iso=photo.iso,
                shutter=photo.shutter,
                focus_position=_focus_position(features),
                signature=features.get("signature"),
                sharp_map=features.get("sharp_map"),
            )
        )
    return record_merges(session, project, detect_focus_stacks(shots), protected)


def propose_in_server(
    session: Session,
    project: Project,
    photos: Sequence[Photo],
    protected: set[int],
    *,
    order: Callable[[Photo], tuple],
) -> int:
    """Everything detection does without looking at pixels: bracketings first,
    then focus stacks among the frames no bracketing took (section 25.2).
    Panoramas need the thumbnails, and a worker (``detect_merges``)."""
    created = propose_brackets(session, project, photos, protected, order=order)
    created += propose_stacks(session, project, photos, protected, order=order)
    return created
