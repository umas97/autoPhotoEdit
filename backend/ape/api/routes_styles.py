# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase 6 over HTTP: the style library, and a project's choice of style.

The library is global (profiles are reused across projects, section 8.4);
the choice is per project. Choosing queues a prediction per photo in the
workers; reading the project's style state is what applies the finished
predictions as versions (``style/apply.py``), lazily, like the clustering.

A profile travels as an ``.apestyle`` (section 20.1): exported as a download
the browser saves, imported from the request body. The server writes no file
for either.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Photo, Project, StyleProfile, StyleSample, StyleSampleStatus
from ..style import apply as style_apply
from ..style import builtin, portable, service
from ..style.pairing import RAW_SUFFIXES, list_references, pair_files
from ..style.profile import ensure_builtins
from .deps import get_photo, get_project, get_session

__all__ = ["router"]

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["styles"])


class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    raw_dir: str
    reference_dir: str
    notes: str | None = None


class ProfileEdit(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str | None = None


class NameIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class PairIn(BaseModel):
    raw: str
    reference: str


class SampleEdit(BaseModel):
    excluded: bool


class ProjectStyleIn(BaseModel):
    profile_id: int | None
    coherence_lambda: float | None = Field(default=None, ge=0.0, le=1.0)


def _profile(session: Session, profile_id: int) -> StyleProfile:
    profile = session.get(StyleProfile, profile_id)
    if profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"profilo {profile_id} inesistente")
    return profile


def _sample(session: Session, sample_id: int) -> StyleSample:
    sample = session.get(StyleSample, sample_id)
    if sample is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"coppia {sample_id} inesistente")
    return sample


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))


def _sample_out(sample: StyleSample) -> dict:
    return {
        "id": sample.id,
        "raw": Path(sample.raw_path).name if sample.raw_path else None,
        "reference": Path(sample.reference_path).name if sample.reference_path else None,
        "raw_path": sample.raw_path,
        "reference_path": sample.reference_path,
        "pairing": sample.pairing,
        "status": sample.status.value,
        "delta_e": None if sample.delta_e is None else round(sample.delta_e, 2),
        "excluded": bool(sample.excluded),
        "error": sample.error,
        "has_thumbnail": sample.thumbnail is not None,
        "raw_available": bool(sample.raw_path) and Path(sample.raw_path).is_file(),
    }


# --- library -----------------------------------------------------------------


@router.get("/styles")
def list_profiles(session: Session = Depends(get_session)) -> list[dict]:
    ensure_builtins(session)
    profiles = session.scalars(
        select(StyleProfile).order_by(StyleProfile.builtin.desc(), StyleProfile.name)
    ).all()
    return [service.summary(session, p) for p in profiles]


@router.post("/styles", status_code=status.HTTP_201_CREATED)
def create_profile(body: ProfileIn, session: Session = Depends(get_session)) -> dict:
    try:
        profile, report = service.create_profile(
            session,
            body.name,
            raw_dir=body.raw_dir,
            reference_dir=body.reference_dir,
            notes=body.notes,
        )
    except service.ProfileError as exc:
        raise _bad_request(exc) from exc
    return {**service.summary(session, profile), "pairing": report}


@router.get("/styles/{profile_id}")
def read_profile(profile_id: int, session: Session = Depends(get_session)) -> dict:
    profile = _profile(session, profile_id)
    samples = session.scalars(
        select(StyleSample)
        .where(
            StyleSample.profile_id == profile.id,
            StyleSample.status != StyleSampleStatus.PROPOSED,
        )
        .order_by(StyleSample.id)
    ).all()
    return {**service.summary(session, profile), "sample_list": [_sample_out(s) for s in samples]}


@router.patch("/styles/{profile_id}")
def edit_profile(
    profile_id: int, body: ProfileEdit, session: Session = Depends(get_session)
) -> dict:
    profile = _profile(session, profile_id)
    if profile.builtin:
        raise _bad_request(ValueError("un profilo predefinito non si modifica: duplicalo"))
    if body.name is not None and body.name.strip() != profile.name:
        clash = session.scalars(
            select(StyleProfile.id).where(StyleProfile.name == body.name.strip())
        ).first()
        if clash:
            raise _bad_request(ValueError(f"esiste già un profilo chiamato «{body.name}»"))
        profile.name = body.name.strip()
    if body.notes is not None:
        profile.notes = body.notes
    return service.summary(session, profile)


@router.delete("/styles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_profile(profile_id: int, session: Session = Depends(get_session)) -> Response:
    try:
        service.delete_profile(session, _profile(session, profile_id))
    except service.ProfileError as exc:
        raise _bad_request(exc) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/styles/{profile_id}/duplicate", status_code=status.HTTP_201_CREATED)
def duplicate_profile(
    profile_id: int, body: NameIn, session: Session = Depends(get_session)
) -> dict:
    try:
        copy = service.duplicate_profile(session, _profile(session, profile_id), body.name)
    except service.ProfileError as exc:
        raise _bad_request(exc) from exc
    return service.summary(session, copy)


@router.get("/styles/{profile_id}/unpaired")
def unpaired(profile_id: int, session: Session = Depends(get_session)) -> dict:
    """What the manual pairing screen offers: files of the two folders not in a pair."""
    profile = _profile(session, profile_id)
    used = {
        p
        for pair in session.execute(
            select(StyleSample.raw_path, StyleSample.reference_path).where(
                StyleSample.profile_id == profile.id
            )
        ).all()
        for p in pair
        if p
    }
    raws = []
    if profile.raw_dir and Path(profile.raw_dir).is_dir():
        raws = sorted(
            str(p)
            for p in Path(profile.raw_dir).iterdir()
            if p.suffix.lower() in RAW_SUFFIXES and str(p) not in used
        )
    references = []
    if profile.reference_dir:
        references = [str(p) for p in list_references(profile.reference_dir) if str(p) not in used]
    return {"raws": raws, "references": references}


@router.post("/styles/{profile_id}/pairs", status_code=status.HTTP_201_CREATED)
def add_pair(profile_id: int, body: PairIn, session: Session = Depends(get_session)) -> dict:
    try:
        sample = service.add_pair(session, _profile(session, profile_id), body.raw, body.reference)
    except service.ProfileError as exc:
        raise _bad_request(exc) from exc
    return _sample_out(sample)


@router.post("/styles/pairing-preview")
def pairing_preview(body: ProfileIn) -> dict:
    """How the two folders would pair, before creating anything."""
    raws_dir, refs_dir = Path(body.raw_dir).expanduser(), Path(body.reference_dir).expanduser()
    if not raws_dir.is_dir() or not refs_dir.is_dir():
        raise _bad_request(ValueError("cartella non trovata"))
    raws = sorted(p for p in raws_dir.iterdir() if p.suffix.lower() in RAW_SUFFIXES)
    result = pair_files(raws, list_references(refs_dir))
    return {
        "pairs": len(result.pairs),
        "methods": {m: sum(p.method == m for p in result.pairs) for m in ("name", "xmp", "time")},
        "unpaired_references": len(result.unpaired_references),
        "raws": len(raws),
    }


@router.patch("/styles/samples/{sample_id}")
def edit_sample(sample_id: int, body: SampleEdit, session: Session = Depends(get_session)) -> dict:
    sample = _sample(session, sample_id)
    service.set_excluded(session, sample, body.excluded)
    return _sample_out(sample)


@router.delete("/styles/samples/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sample(sample_id: int, session: Session = Depends(get_session)) -> Response:
    sample = _sample(session, sample_id)
    profile = session.get(StyleProfile, sample.profile_id)
    session.delete(sample)
    session.flush()
    if profile is not None and profile.model is not None:
        from ..style.profile import retrain

        retrain(session, profile)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/styles/samples/{sample_id}/thumbnail")
def sample_thumbnail(sample_id: int, session: Session = Depends(get_session)) -> Response:
    sample = _sample(session, sample_id)
    if not sample.thumbnail:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "miniatura non ancora disponibile")
    return Response(
        sample.thumbnail,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/styles/{profile_id}/export")
def export_profile(profile_id: int, session: Session = Depends(get_session)) -> Response:
    profile = _profile(session, profile_id)
    data = portable.export_profile(session, profile)
    safe = (
        "".join(c if c.isalnum() or c in "-_ " else "_" for c in profile.name).strip() or "profilo"
    )
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe}.apestyle"'},
    )


@router.post("/styles/import", status_code=status.HTTP_201_CREATED)
async def import_profile(request: Request, session: Session = Depends(get_session)) -> dict:
    data = await request.body()
    if len(data) > portable.MAX_UNCOMPRESSED:
        raise _bad_request(ValueError("file troppo grande per essere un profilo di stile"))
    try:
        profile = portable.import_profile(session, data)
    except portable.PortableError as exc:
        raise _bad_request(exc) from exc
    return service.summary(session, profile)


# --- a project's style ----------------------------------------------------------


@router.get("/projects/{project_id}/style")
def project_style(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    """The proposal, the choice, and the application of the finished predictions."""
    applied = style_apply.apply_predictions(session, project)
    data = service.proposal(session, project)
    data["coherence_lambda"] = project.coherence_lambda
    data["applied"] = applied
    data["pending"] = style_apply.pending_predictions(session, project.id)
    data["status"] = style_apply.status_counts(session, project)
    return data


@router.put("/projects/{project_id}/style")
def choose_style(
    body: ProjectStyleIn,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    if body.coherence_lambda is not None:
        project.coherence_lambda = body.coherence_lambda
    queued = 0
    if body.profile_id != project.style_profile_id:
        try:
            queued = service.choose_profile(session, project, body.profile_id)
        except service.ProfileError as exc:
            raise _bad_request(exc) from exc
    return {
        "queued": queued,
        "profile_id": project.style_profile_id,
        "coherence_lambda": project.coherence_lambda,
    }


@router.get("/photos/{photo_id}/style")
def photo_style(photo: Photo = Depends(get_photo), session: Session = Depends(get_session)) -> dict:
    """Which samples guided this photo's edit (section 8.3's transparency)."""
    prediction = photo.prediction or {}
    profile = (
        session.get(StyleProfile, prediction["profile_id"])
        if prediction.get("profile_id")
        else None
    )
    neighbours = []
    for sample_id, weight in prediction.get("neighbours", []):
        sample = session.get(StyleSample, int(sample_id))
        if sample is not None:
            neighbours.append({**_sample_out(sample), "weight": weight})
    return {
        "profile": None if profile is None else {"id": profile.id, "name": profile.name},
        "method": prediction.get("method"),
        "neighbours": neighbours,
        "kept_user_edit": bool(prediction.get("kept_user_edit")),
        "applied": bool(prediction.get("applied")),
        "neutral_profile": builtin.NEUTRAL_AUTO,
    }


@router.get("/photos/{photo_id}/style/params")
def photo_style_params(
    profile_id: int,
    photo: Photo = Depends(get_photo),
    session: Session = Depends(get_session),
) -> dict:
    """The parameters one profile gives this photo, without coherence.

    What the interface previews for "this profile *vs* neutral" (section 22):
    it renders them through the ordinary preview endpoint.
    """
    from ..style.predict import describe_photo, params_for
    from ..style.profile import load
    from .routes_photos import current_params

    try:
        profile = load(session, profile_id)
        scene, measured = describe_photo(
            photo.scene_features, photo.embedding, photo.analysis, photo.proxy_path
        )
        params = params_for(profile, scene, current_params(session, photo))
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(exc) from exc
    analysis = dict(photo.analysis or {})
    if analysis.get("auto") != measured:
        analysis["auto"] = measured
        photo.analysis = analysis
    return {"profile_id": profile_id, "params": params.model_dump(mode="json")}
