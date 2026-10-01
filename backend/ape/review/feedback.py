# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Learning from the review (section 9.2.4): the loop that closes a project.

Every correction the user approves becomes a training pair -- the RAW and the
parameters they accepted -- kept as a ``StyleSample`` with status
``proposed``: visible, counted, and **not trained on** until the user says so
at the end of the project. Then :func:`incorporate` makes them ordinary
samples of the profile and refits it; a built-in profile (section 22) cannot
be retrained, so its corrections found a new learned profile instead.

What counts as a correction is what the user actually looked at and changed:

* a photo approved one by one whose style is no longer the prediction (moved
  sliders, a variant, a rejection to neutral kept);
* the representative of an approved scene, if its style was corrected.

The other photos of a corrected scene received the correction as a delta and
were never looked at one by one: they would teach the profile its own guess
back, so they are not pairs.

A pair has no reference image, so its thumbnail -- what a portable profile
carries instead of files (section 20.1) -- is rendered from the RAW by a job
(``jobs/handlers_review.py``).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    Photo,
    Project,
    StyleProfile,
    StyleSample,
    StyleSampleStatus,
)
from ..pipeline.params import EditParams
from ..raw.source import pixels_of
from ..style import vector as sv
from ..style.apply import style_signature

__all__ = ["FeedbackError", "discard", "incorporate", "proposed_count", "summary", "sync"]


class FeedbackError(ValueError):
    """A request about the corrections that cannot be honoured, in words for the UI."""


def _proposed(session: Session, project: Project):
    return (
        StyleSample.project_id == project.id,
        StyleSample.profile_id == project.style_profile_id,
        StyleSample.status == StyleSampleStatus.PROPOSED,
    )


def _is_correction(photo: Photo, params: EditParams) -> bool:
    review = photo.review or {}
    if review.get("decision") != "approved":
        return False
    if review.get("by") != "photo" and not review.get("representative"):
        return False
    applied = (photo.prediction or {}).get("applied_signature")
    return applied is not None and style_signature(params) != applied


def sync(session: Session, project: Project, photo: Photo) -> StyleSample | None:
    """Create, update or drop the photo's proposed pair to match the review."""
    from ..api.routes_photos import current_params
    from ..jobs.handlers_review import enqueue_thumbnail

    if project.style_profile_id is None:
        return None
    session.flush()
    existing = session.scalars(
        select(StyleSample).where(*_proposed(session, project), StyleSample.photo_id == photo.id)
    ).first()
    params = current_params(session, photo)
    context = (photo.prediction or {}).get("context")
    if not context or not _is_correction(photo, params):
        if existing is not None:
            session.delete(existing)
        return None
    # The photo's removals are its own (docs/SPEC_rimozione.md R6): a correction
    # teaches the style its sliders, never its erased cables.
    stored = params.model_copy(update={"retouch": []}).model_dump(mode="json")
    if existing is not None and existing.params == stored:
        return existing
    sample = existing or StyleSample(
        profile_id=project.style_profile_id,
        project_id=project.id,
        photo_id=photo.id,
        status=StyleSampleStatus.PROPOSED,
        pairing="feedback",
        params={},
    )
    # A merged photo trains a style like any other (section 25.7): its
    # intermediate is the linear frame the inversion needs.
    source = pixels_of(photo)
    sample.raw_path = str(source) if source is not None else None
    sample.reference_path = None
    sample.params = stored
    sample.vector = [round(float(x), 6) for x in sv.from_params(params, sv.StyleContext(**context))]
    sample.context = {
        **context,
        # What the correction adds to the frame's own clipping is the style's
        # tolerance for it (``reference.py``); the photo was measured already.
        "tails": ((photo.analysis or {}).get("auto") or {}).get("tails"),
    }
    sample.scene_features = photo.scene_features
    sample.embedding = photo.embedding
    sample.thumbnail = None
    if existing is None:
        session.add(sample)
    session.flush()
    enqueue_thumbnail(session, sample)
    return sample


def proposed_count(session: Session, project: Project) -> int:
    if project.style_profile_id is None:
        return 0
    return int(
        session.scalar(select(func.count(StyleSample.id)).where(*_proposed(session, project))) or 0
    )


def summary(session: Session, project: Project) -> dict[str, Any]:
    """What the "Le tue correzioni" card shows."""
    profile = (
        session.get(StyleProfile, project.style_profile_id) if project.style_profile_id else None
    )
    samples = (
        session.scalars(
            select(StyleSample).where(*_proposed(session, project)).order_by(StyleSample.id)
        ).all()
        if profile
        else []
    )
    names = {}
    if samples:
        ids = [s.photo_id for s in samples if s.photo_id is not None]
        names = dict(
            session.execute(select(Photo.id, Photo.filename).where(Photo.id.in_(ids))).all()
        )
    return {
        "profile": None
        if profile is None
        else {"id": profile.id, "name": profile.name, "builtin": bool(profile.builtin)},
        "proposed": len(samples),
        "samples": [
            {
                "id": s.id,
                "photo_id": s.photo_id,
                "filename": names.get(s.photo_id),
                "has_thumbnail": s.thumbnail is not None,
            }
            for s in samples
        ],
        "suggested_name": None if profile is None else f"{profile.name} · {project.name}",
    }


def incorporate(session: Session, project: Project, name: str | None = None) -> StyleProfile:
    """Make the project's corrections training pairs, and refit.

    A learned profile gets them as new samples. A built-in one cannot be
    trained: the corrections found a new learned profile, named ``name``.

    Raises:
        FeedbackError: nothing to incorporate, or the name is taken.
    """
    from ..style.profile import retrain
    from ..style.service import ProfileError, _unique

    samples = list(session.scalars(select(StyleSample).where(*_proposed(session, project))))
    if not samples:
        raise FeedbackError("non ci sono correzioni da incorporare")
    profile = session.get(StyleProfile, project.style_profile_id)
    if profile is None:
        raise FeedbackError("il profilo del progetto non esiste più")
    if profile.builtin:
        label = (name or f"{profile.name} · {project.name}").strip()
        try:
            _unique(session, label)
        except ProfileError as exc:
            raise FeedbackError(str(exc)) from exc
        target = StyleProfile(
            name=label,
            notes=f"Dalle correzioni del progetto «{project.name}» su «{profile.name}».",
            builtin=False,
        )
        session.add(target)
        session.flush()
    else:
        target = profile
    for sample in samples:
        sample.profile_id = target.id
        sample.status = StyleSampleStatus.READY
    session.flush()
    retrain(session, target)
    return target


def discard(session: Session, project: Project) -> int:
    """Drop the project's proposed pairs without training on them."""
    samples = list(session.scalars(select(StyleSample).where(*_proposed(session, project))))
    for sample in samples:
        session.delete(sample)
    return len(samples)
