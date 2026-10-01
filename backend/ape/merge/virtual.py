# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Merges as the catalogue sees them: groups, decisions and the derived photo.

The life of a group (section 25):

* ``proposed`` -- found by detection or made by hand. Its frames are left
  alone by culling. The user can preview it, change its members or reference,
  accept or reject it;
* ``accepted`` -- a full merge is queued; when it has run, a photo of kind
  ``merged`` stands in for the frames, which leave the working set
  (``Photo.superseded``) but stay in the project, behind "Mostra scatti
  sorgente";
* ``failed`` -- the merge could not be made, with the reason; no merged photo
  exists (section 25.5.6);
* ``rejected`` -- permanent: detection never proposes the same frames again.

Undoing an accepted merge takes the merged photo out of the working set and
gives the frames back. The merged photo is *kept*, with its history: accepting
the same merge again brings it back as it was, edits included, which is what
makes the undo itself undoable (section 23).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db.models import (
    Job,
    JobKind,
    JobState,
    MergeDecision,
    MergeGroup,
    MergeKind,
    MergeMember,
    MergeRole,
    Photo,
    PhotoKind,
    PhotoStatus,
    Project,
)
from ..raw.intermediate import SUFFIX
from .detect import BracketShot, exposure_brightness
from .engine import Member, Recipe

__all__ = [
    "accept",
    "create_group",
    "edit_group",
    "intermediate_path_for",
    "recipe_for",
    "record_failure",
    "record_success",
    "reject",
    "request_preview",
    "undo",
]

#: What a merged photo is called: the reference's name and the kind of merge,
#: so ``{basename}`` in an export template gives ``DSC05618_HDR``.
_NAME_SUFFIX = {
    MergeKind.HDR: "HDR",
    MergeKind.FOCUS_STACK: "STACK",
    MergeKind.PANORAMA: "PANO",
}

#: Largest group a user can put together by hand. Section 25.5.4 caps a
#: panorama at twelve frames; the other merges never need as many.
MAX_MEMBERS = 12

#: Decisions whose group may still be changed.
_EDITABLE = (MergeDecision.PROPOSED, MergeDecision.FAILED)


def _shot(photo: Photo) -> BracketShot:
    return BracketShot(
        id=photo.id, shot_at=photo.shot_at, order=(), focal_length=photo.focal_length,
        aperture=photo.aperture, iso=photo.iso, shutter=photo.shutter,
        exposure_bias=None, release_mode=None, signature=None,
    )


def _ev_offsets(photos: Sequence[Photo], reference: Photo) -> dict[int, float | None]:
    """Stops of each frame relative to the reference, from EXIF."""
    base = exposure_brightness(_shot(reference))
    offsets: dict[int, float | None] = {}
    for photo in photos:
        value = exposure_brightness(_shot(photo))
        offsets[photo.id] = None if value is None or base is None else round(value - base, 2)
    return offsets


def recipe_for(session: Session, group: MergeGroup) -> Recipe:
    """The recipe of a group, from its current members and options."""
    members = []
    for member in sorted(group.members, key=lambda m: m.position):
        photo = session.get(Photo, member.photo_id)
        if photo is None or not photo.path:
            continue
        members.append(
            Member(
                photo_id=photo.id,
                path=photo.path,
                filename=photo.filename,
                hash=photo.hash,
                ev_offset=member.ev_offset,
                reference=member.role is MergeRole.REFERENCE,
                shot_at=photo.shot_at,
            )
        )
    return Recipe(
        group_id=group.id, kind=group.kind.value, members=tuple(members),
        options=dict((group.params or {}).get("options") or {}),
    )


def intermediate_path_for(recipe: Recipe) -> Path:
    """Named by the recipe: the same merge always lands on the same file."""
    digest = recipe.digest()
    return get_settings().intermediate_dir / digest[:2] / f"{digest}{SUFFIX}"


def _enqueue(session: Session, group: MergeGroup, kind: JobKind) -> Job | None:
    from ..jobs.queue import enqueue

    key = f"{kind.value}:{group.id}"
    return enqueue(
        session, kind, {"group_id": group.id}, project_id=group.project_id, dedupe_key=key
    )


def request_preview(session: Session, group: MergeGroup) -> Job | None:
    """Queue the 1024 px preview of a group (section 25.6)."""
    return _enqueue(session, group, JobKind.MERGE_PREVIEW)


def accept(session: Session, group: MergeGroup) -> None:
    """The user said yes: queue the full merge. Nothing changes until it has run."""
    group.decision = MergeDecision.ACCEPTED
    group.error = None
    _enqueue(session, group, JobKind.MERGE)


def reject(session: Session, group: MergeGroup) -> None:
    """For good. The frames become ordinary photos again, for culling too."""
    if group.decision is MergeDecision.ACCEPTED:
        undo(session, group)
    group.decision = MergeDecision.REJECTED
    _release_for_culling(session, group)


def _release_for_culling(session: Session, group: MergeGroup) -> None:
    """Mark the frames as ungrouped, so culling regroups them at its next look."""
    ids = [m.photo_id for m in group.members]
    session.execute(update(Photo).where(Photo.id.in_(ids)).values(burst_rank=None))


def _cancel_queued(session: Session, group: MergeGroup) -> None:
    session.execute(
        update(Job)
        .where(
            Job.kind == JobKind.MERGE,
            Job.dedupe_key == f"{JobKind.MERGE.value}:{group.id}",
            Job.state == JobState.QUEUED,
        )
        .values(state=JobState.CANCELLED)
    )


def undo(session: Session, group: MergeGroup) -> None:
    """Take an accepted merge back (section 25.6): the frames return, the group
    is proposed again, and the merged photo waits, hidden, with its history."""
    _cancel_queued(session, group)
    if group.result_photo_id is not None:
        session.execute(
            update(Photo).where(Photo.id == group.result_photo_id).values(superseded=True)
        )
    ids = [m.photo_id for m in group.members]
    session.execute(update(Photo).where(Photo.id.in_(ids)).values(superseded=False))
    group.decision = MergeDecision.PROPOSED


def record_success(
    session: Session, group: MergeGroup, path: Path, report: dict, size: tuple[int, int]
) -> Photo:
    """The full merge has run: create (or bring back) the merged photo."""
    from ..jobs.queue import enqueue

    recipe = recipe_for(session, group)
    reference = session.get(Photo, recipe.reference.photo_id)
    members = [session.get(Photo, m.photo_id) for m in recipe.members]
    photo = session.get(Photo, group.result_photo_id) if group.result_photo_id else None
    if photo is None or photo.kind is not PhotoKind.MERGED:
        photo = Photo(
            project_id=group.project_id,
            kind=PhotoKind.MERGED,
            path=None,
            hash=None,
            filename=f"{Path(reference.filename).stem}_{_NAME_SUFFIX[group.kind]}",
            status=PhotoStatus.IMPORTED,
        )
        session.add(photo)
    previous = photo.intermediate_path
    photo.intermediate_path = str(path)
    photo.merge_group_id = group.id
    photo.superseded = False
    photo.missing = False
    photo.error = None
    if photo.status is PhotoStatus.FAILED:
        photo.status = PhotoStatus.IMPORTED
    for name in ("camera", "lens", "iso", "aperture", "shutter", "focal_length"):
        setattr(photo, name, getattr(reference, name))
    photo.shot_at = min((m.shot_at for m in members if m and m.shot_at), default=reference.shot_at)
    photo.width, photo.height = size
    photo.orientation = None
    if previous != str(path):
        # New pixels: what was measured on the old ones -- straightening, scene
        # features, the proxy itself -- no longer describes the photo. The
        # edits stay: they are the user's, and they apply to any pixels.
        photo.proxy_path = None
        photo.analysis = None
    session.flush()

    for member in members:
        if member is not None:
            member.superseded = True
            member.merge_group_id = group.id
    group.result_photo_id = photo.id
    group.decision = MergeDecision.ACCEPTED
    group.error = None
    group.report = {**(group.report or {}), "full": report}
    if previous != str(path) or not photo.proxy_path:
        enqueue(
            session, JobKind.PROXY, {"photo_id": photo.id, "force": True},
            project_id=group.project_id, dedupe_key=f"proxy:{photo.id}",
        )
    if previous and previous != str(path):
        Path(previous).unlink(missing_ok=True)
    if previous != str(path):
        _propose_border_crop(session, photo, report.get("crop"))
    return photo


def _propose_border_crop(session: Session, photo: Photo, crop: dict | None) -> None:
    """Section 25.5.5: the panorama's clean rectangle, as a pending proposal.

    The merge knows exactly which pixels are empty; the analysis of the merged
    photo keeps this proposal rather than guessing one (``analysis/borders.py``).
    """
    from ..analysis.borders import BORDERS_ASPECT
    from ..db.models import CropDecision, CropProposal

    for old in session.scalars(
        select(CropProposal).where(
            CropProposal.photo_id == photo.id, CropProposal.decision == CropDecision.PENDING
        )
    ):
        session.delete(old)
    if crop and (crop["width"] < 1.0 or crop["height"] < 1.0):
        session.add(CropProposal(
            photo_id=photo.id,
            rect={k: float(crop[k]) for k in ("x", "y", "width", "height")},
            aspect=BORDERS_ASPECT, score=None, decision=CropDecision.PENDING,
        ))


def record_failure(session: Session, group: MergeGroup, message: str) -> None:
    """Section 25.5.6: failed, with the reason, and no merged photo in the flow."""
    group.decision = MergeDecision.FAILED
    group.error = message[:1000]
    if group.result_photo_id is not None:
        session.execute(
            update(Photo).where(Photo.id == group.result_photo_id).values(superseded=True)
        )
    ids = [m.photo_id for m in group.members]
    session.execute(update(Photo).where(Photo.id.in_(ids)).values(superseded=False))


def _default_reference(kind: MergeKind, photos: Sequence[Photo]) -> Photo:
    """The metered frame of a bracketing; the middle frame otherwise."""
    if kind is MergeKind.HDR:
        values = [(exposure_brightness(_shot(p)), i) for i, p in enumerate(photos)]
        known = sorted((v, i) for v, i in values if v is not None)
        if known:
            return photos[known[len(known) // 2][1]]
    return photos[len(photos) // 2]


def _set_members(
    session: Session, group: MergeGroup, photos: Sequence[Photo], reference_id: int | None
) -> None:
    by_id = {p.id: p for p in photos}
    reference = by_id.get(reference_id) if reference_id else None
    reference = reference or _default_reference(group.kind, photos)
    offsets = _ev_offsets(photos, reference)
    group.members.clear()
    session.flush()
    for position, photo in enumerate(photos):
        group.members.append(
            MergeMember(
                photo_id=photo.id,
                position=position,
                ev_offset=offsets[photo.id],
                role=MergeRole.REFERENCE if photo.id == reference.id else MergeRole.MEMBER,
            )
        )
        photo.merge_group_id = group.id


def _validated_photos(session: Session, project: Project, photo_ids: Sequence[int]) -> list[Photo]:
    ids = list(dict.fromkeys(int(i) for i in photo_ids))
    if len(ids) < 2:
        raise ValueError("una fusione ha bisogno di almeno due scatti")
    if len(ids) > MAX_MEMBERS:
        raise ValueError(f"al massimo {MAX_MEMBERS} scatti per fusione")
    photos = session.scalars(
        select(Photo).where(Photo.id.in_(ids), Photo.project_id == project.id)
    ).all()
    if len(photos) != len(ids):
        raise ValueError("alcuni scatti non appartengono a questo progetto")
    if any(p.kind is not PhotoKind.RAW or not p.path for p in photos):
        raise ValueError("si possono fondere solo scatti RAW, non altre fusioni")
    busy = session.scalars(
        select(MergeMember.photo_id)
        .join(MergeGroup, MergeGroup.id == MergeMember.group_id)
        .where(MergeMember.photo_id.in_(ids), MergeGroup.decision == MergeDecision.ACCEPTED)
    ).all()
    if busy:
        raise ValueError("alcuni scatti fanno già parte di una fusione accettata")
    return sorted(photos, key=lambda p: (p.shot_at is None, p.shot_at, p.filename))


def create_group(
    session: Session, project: Project, kind: MergeKind, photo_ids: Sequence[int],
    reference_id: int | None = None,
) -> MergeGroup:
    """A group made by hand from photos selected in the grid (section 25.2).

    Raises:
        ValueError: with the reason, in Italian.
    """
    photos = _validated_photos(session, project, photo_ids)
    group = MergeGroup(
        project_id=project.id, kind=kind, decision=MergeDecision.PROPOSED, confidence=None,
        detect_reasons=[{"manual": True, "frames": len(photos)}],
    )
    session.add(group)
    session.flush()
    _set_members(session, group, photos, reference_id)
    return group


def edit_group(
    session: Session, group: MergeGroup, photo_ids: Sequence[int] | None,
    reference_id: int | None, options: dict | None = None,
) -> None:
    """Change members, reference or options of a group not yet merged.

    Raises:
        ValueError: the group is accepted or rejected, or the new members are
            not valid.
    """
    if group.decision not in _EDITABLE:
        raise ValueError("si può modificare solo una fusione proposta o non riuscita")
    project = session.get(Project, group.project_id)
    ids = photo_ids if photo_ids is not None else [m.photo_id for m in group.members]
    old = {m.photo_id for m in group.members}
    photos = _validated_photos(session, project, ids)
    if reference_id is None and photo_ids is None:
        current = [m.photo_id for m in group.members if m.role is MergeRole.REFERENCE]
        reference_id = current[0] if current else None
    _set_members(session, group, photos, reference_id)
    removed = old - {p.id for p in photos}
    if removed:
        session.execute(
            update(Photo)
            .where(Photo.id.in_(list(removed)), Photo.merge_group_id == group.id)
            .values(merge_group_id=None, burst_rank=None)
        )
    if options is not None:
        group.params = {**(group.params or {}), "options": options}
    reasons = list(group.detect_reasons or [])
    if reasons and reasons[0].get("manual"):
        reasons[0] = {**reasons[0], "frames": len(photos)}
        group.detect_reasons = reasons
    group.decision = MergeDecision.PROPOSED
    group.error = None
