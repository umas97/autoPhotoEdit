# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Style profiles as the catalogue and the interface see them.

Everything here is quick and runs in the server: creating rows, pairing file
names, queueing the slow work for the workers (``jobs/handlers_style.py``),
and reading back what the interface shows. The two long operations --
inverting a pair, predicting a photo -- are never done here.

The folders a profile is created from are the user's, like a project's
source: they are registered as protected roots (``safety.py``) the moment
the profile knows about them, although nothing here writes anywhere near
them anyway.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import (
    Photo,
    Project,
    StyleProfile,
    StyleSample,
    StyleSampleStatus,
)
from ..db.workset import editing_set
from ..safety import register_protected_root
from . import match
from .pairing import RAW_SUFFIXES, list_references, pair_files
from .profile import MIN_PAIRS, RECOMMENDED_PAIRS, ensure_builtins, retrain

__all__ = [
    "ProfileError",
    "add_pair",
    "choose_profile",
    "create_profile",
    "delete_profile",
    "duplicate_profile",
    "proposal",
    "set_excluded",
    "summary",
]


class ProfileError(ValueError):
    """A request about profiles that cannot be honoured, in words for the UI."""


def _raws(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in RAW_SUFFIXES)


def _unique(session: Session, name: str) -> None:
    if session.scalars(select(StyleProfile.id).where(StyleProfile.name == name)).first():
        raise ProfileError(f"esiste già un profilo chiamato «{name}»")


def create_profile(
    session: Session,
    name: str,
    *,
    raw_dir: str | Path,
    reference_dir: str | Path,
    notes: str | None = None,
) -> tuple[StyleProfile, dict]:
    """A new learned profile from a folder of RAWs and a folder of edits.

    The two folders may be the same. Every pair found becomes a pending sample
    with its inversion queued; what could not be paired is reported, for the
    manual pairing screen.
    """
    from ..jobs.handlers_style import enqueue_pair

    name = name.strip()
    if not name:
        raise ProfileError("il profilo ha bisogno di un nome")
    _unique(session, name)
    raws_folder = Path(raw_dir).expanduser().resolve()
    references_folder = Path(reference_dir).expanduser().resolve()
    if not raws_folder.is_dir():
        raise ProfileError(f"cartella dei RAW non trovata: {raws_folder}")
    if not references_folder.is_dir():
        raise ProfileError(f"cartella delle foto editate non trovata: {references_folder}")
    register_protected_root(raws_folder)
    register_protected_root(references_folder)

    result = pair_files(_raws(raws_folder), list_references(references_folder))
    profile = StyleProfile(
        name=name,
        notes=notes,
        builtin=False,
        raw_dir=str(raws_folder),
        reference_dir=str(references_folder),
    )
    session.add(profile)
    session.flush()
    for pair in result.pairs:
        sample = StyleSample(
            profile_id=profile.id,
            raw_path=str(pair.raw),
            reference_path=str(pair.reference),
            params={},
            pairing=pair.method,
            status=StyleSampleStatus.PENDING,
        )
        session.add(sample)
        session.flush()
        enqueue_pair(session, sample)
    report = {
        "pairs": len(result.pairs),
        "methods": {m: sum(p.method == m for p in result.pairs) for m in ("name", "xmp", "time")},
        "unpaired_references": [str(p) for p in result.unpaired_references],
        "unpaired_raws": len(result.unpaired_raws),
        "duplicates": [str(p.reference) for p in result.duplicates],
    }
    return profile, report


def add_pair(
    session: Session, profile: StyleProfile, raw: str | Path, reference: str | Path
) -> StyleSample:
    """A pair made by hand on the manual pairing screen (section 8.1)."""
    from ..jobs.handlers_style import enqueue_pair

    if profile.builtin:
        raise ProfileError("un profilo predefinito non ha coppie: duplicalo prima")
    raw_path, ref_path = Path(raw).expanduser().resolve(), Path(reference).expanduser().resolve()
    if not raw_path.is_file() or raw_path.suffix.lower() not in RAW_SUFFIXES:
        raise ProfileError(f"non è un file RAW leggibile: {raw_path.name}")
    if not ref_path.is_file():
        raise ProfileError(f"file editato non trovato: {ref_path.name}")
    register_protected_root(raw_path.parent)
    register_protected_root(ref_path.parent)
    sample = StyleSample(
        profile_id=profile.id,
        raw_path=str(raw_path),
        reference_path=str(ref_path),
        params={},
        pairing="manual",
        status=StyleSampleStatus.PENDING,
    )
    session.add(sample)
    session.flush()
    enqueue_pair(session, sample)
    return sample


def set_excluded(session: Session, sample: StyleSample, excluded: bool) -> None:
    """Leave a sample out of (or back into) training, and refit."""
    sample.excluded = excluded
    session.flush()
    profile = session.get(StyleProfile, sample.profile_id)
    if profile is not None and not _pending(session, profile.id):
        retrain(session, profile)


def _pending(session: Session, profile_id: int) -> int:
    return int(
        session.scalar(
            select(func.count(StyleSample.id)).where(
                StyleSample.profile_id == profile_id,
                StyleSample.status == StyleSampleStatus.PENDING,
            )
        )
        or 0
    )


def delete_profile(session: Session, profile: StyleProfile) -> None:
    if profile.builtin:
        raise ProfileError("i profili predefiniti non si possono eliminare")
    session.delete(profile)


def duplicate_profile(session: Session, profile: StyleProfile, name: str) -> StyleProfile:
    """A user copy (section 22: built-ins are duplicated, never overwritten).

    A learned profile's copy carries its samples and model along, so the copy
    can be retrained or pruned without touching the original.
    """
    name = name.strip()
    if not name:
        raise ProfileError("il profilo ha bisogno di un nome")
    _unique(session, name)
    copy = StyleProfile(
        name=name,
        notes=profile.notes,
        builtin=False,
        builtin_rules=profile.builtin_rules,
        norm_stats=profile.norm_stats,
        n_pairs=profile.n_pairs,
        embedding_centroid=profile.embedding_centroid,
        raw_dir=profile.raw_dir,
        reference_dir=profile.reference_dir,
    )
    session.add(copy)
    session.flush()
    for sample in session.scalars(
        select(StyleSample).where(
            StyleSample.profile_id == profile.id,
            StyleSample.status != StyleSampleStatus.PROPOSED,
        )
    ):
        session.add(
            StyleSample(
                profile_id=copy.id,
                raw_path=sample.raw_path,
                reference_path=sample.reference_path,
                params=sample.params,
                scene_features=sample.scene_features,
                embedding=sample.embedding,
                residual_loss=sample.residual_loss,
                excluded=sample.excluded,
                status=sample.status,
                error=sample.error,
                pairing=sample.pairing,
                vector=sample.vector,
                context=sample.context,
                delta_e=sample.delta_e,
                thumbnail=sample.thumbnail,
            )
        )
    session.flush()
    if profile.model:
        # Refit rather than copy: the model names its samples by id, and the
        # copy's samples have new ones.
        retrain(session, copy)
    return copy


def _project_embeddings(session: Session, project_id: int) -> list:
    from ..analysis.embed import decode_embedding

    blobs = session.scalars(
        select(Photo.embedding).where(
            *editing_set(project_id), Photo.embedding.is_not(None)
        )
    ).all()
    return [e for e in (decode_embedding(b) for b in blobs) if e is not None]


def proposal(session: Session, project: Project) -> dict:
    """Every profile with its affinity to the project, and the one to propose."""
    from ..analysis.embed import decode_embedding

    ensure_builtins(session)
    profiles = session.scalars(
        select(StyleProfile).order_by(StyleProfile.builtin.desc(), StyleProfile.name)
    ).all()
    centroid = match.project_centroid(_project_embeddings(session, project.id))
    scores = {
        a.profile_id: a.score
        for a in match.affinities(
            centroid, [(p.id, decode_embedding(p.embedding_centroid)) for p in profiles]
        )
    }
    usable = [p for p in profiles if p.model or p.builtin_rules]
    learned = [p for p in usable if p.model and scores.get(p.id) is not None]
    best = max(learned, key=lambda p: scores[p.id], default=None)
    if best is None or scores[best.id] < match.MIN_AFFINITY:
        from .builtin import NEUTRAL_AUTO

        best = next((p for p in usable if p.builtin and p.name == NEUTRAL_AUTO), None)
    return {
        "proposed": best.id if best else None,
        "current": project.style_profile_id,
        "affinity": {p.id: scores.get(p.id) for p in profiles},
        "min_affinity": match.MIN_AFFINITY,
    }


def choose_profile(session: Session, project: Project, profile_id: int | None) -> int:
    """Set the project's profile and queue a prediction for every analysed photo."""
    from ..analysis.service import enqueue_missing
    from ..jobs.handlers_style import enqueue_prediction

    if profile_id is not None:
        profile = session.get(StyleProfile, profile_id)
        if profile is None:
            raise ProfileError("profilo non trovato")
        if not (profile.model or profile.builtin_rules):
            raise ProfileError(f"il profilo «{profile.name}» non è ancora addestrato")
    project.style_profile_id = profile_id
    session.flush()
    if profile_id is None:
        return 0
    queued = 0
    photos = session.scalars(
        select(Photo).where(*editing_set(project.id))
    ).all()
    for photo in photos:
        if photo.scene_features is not None and photo.proxy_path:
            queued += int(enqueue_prediction(session, photo, profile_id))
    # The rest predicts as soon as its analysis lands (``handlers_analysis``).
    enqueue_missing(session, project.id)
    return queued


def summary(session: Session, profile: StyleProfile) -> dict:
    """What the Styles screen shows about one profile."""
    counts = dict(
        session.execute(
            select(StyleSample.status, func.count(StyleSample.id))
            .where(StyleSample.profile_id == profile.id)
            .group_by(StyleSample.status)
        ).all()
    )
    excluded = int(
        session.scalar(
            select(func.count(StyleSample.id)).where(
                StyleSample.profile_id == profile.id, StyleSample.excluded.is_(True)
            )
        )
        or 0
    )
    ready = int(counts.get(StyleSampleStatus.READY, 0))
    trained = profile.model is not None
    return {
        "id": profile.id,
        "name": profile.name,
        "notes": profile.notes,
        "builtin": bool(profile.builtin),
        "rules": profile.builtin_rules is not None,
        "trained": trained,
        "usable": trained or profile.builtin_rules is not None,
        "n_pairs": profile.n_pairs,
        "samples": {
            "pending": int(counts.get(StyleSampleStatus.PENDING, 0)),
            "ready": ready,
            "unreproducible": int(counts.get(StyleSampleStatus.UNREPRODUCIBLE, 0)),
            "failed": int(counts.get(StyleSampleStatus.FAILED, 0)),
            "excluded": excluded,
        },
        "few_pairs": (not profile.builtin) and trained and profile.n_pairs < MIN_PAIRS,
        "min_pairs": MIN_PAIRS,
        "recommended_pairs": RECOMMENDED_PAIRS,
        "trained_at": profile.trained_at.isoformat() if profile.trained_at else None,
        "stats": profile.norm_stats,
        "raw_dir": profile.raw_dir,
        "reference_dir": profile.reference_dir,
    }
