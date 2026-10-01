# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Culling over HTTP: the screen of section 7.6 and what it asks for.

One large read when the screen opens, many small writes while the user works.

``GET /api/projects/{id}/culling`` is the large one: every photo with its
scores and decision, every burst, every merge proposal. It is also where the
lazy half of the analysis runs -- regrouping when photos were analysed since
the last look -- so that the screen never shows a grouping older than the data.

The writes -- settings, decisions -- answer with only the photos whose outcome
changed. A slider being dragged over two thousand photos changes a handful of
them per step, and sending the other two thousand back each time would be most
of the 100 ms that section 14 allows for the whole round trip.

``GET /api/photos/{id}/thumb`` serves the camera's embedded preview, which is
what the culling grid shows: no proxy exists yet, and none should, until the
user has chosen what to develop.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import snapshots
from ..culling import decisions as user_decisions
from ..culling import service
from ..culling.select import CullingSettings, Decision, summarise, target_count, threshold
from ..db.models import MergeGroup, Photo, PhotoKind, Project, ProjectStatus
from ..merge.proposals import ACTIVE_MERGES
from ..models_registry import feature_status
from .deps import get_photo, get_project, get_session
from .schemas_culling import (
    CullDecision,
    CullDecisionsRequest,
    CullMerge,
    CullPhoto,
    CullSelection,
    CullSettings,
    CullSettingsUpdate,
    CullStartRequest,
    CullSummary,
    CullView,
    FeatureAvailability,
)

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["culling"])

#: Criteria that need a model. The other four are pure arithmetic and always
#: available.
_MODEL_CRITERIA = ("faces", "aesthetic")


def _availability() -> dict[str, FeatureAvailability]:
    result = {}
    for feature in _MODEL_CRITERIA:
        available, reason, entry = feature_status(feature)
        result[feature] = FeatureAvailability(
            available=available,
            reason=reason.value if reason else None,
            licence=entry.licence if entry else None,
            notice=entry.notice if entry else None,
            size_mb=entry.size_mb if entry else None,
        )
    return result


def _settings_out(settings: CullingSettings) -> CullSettings:
    return CullSettings(
        mode=settings.mode,
        target=settings.target,
        aggressiveness=settings.aggressiveness,
        weights=dict(settings.weights),
        criteria=dict(settings.criteria),
        threshold=round(threshold(settings.aggressiveness), 4),
    )


def _summary(
    session: Session, project: Project, decisions: list[Decision], settings: CullingSettings
) -> CullSummary:
    counts = summarise(decisions)
    progress = service.state(session, project)
    return CullSummary(
        total=counts["total"],
        selected=counts["selected"],
        culled=counts["culled"],
        analysed=progress["analysed"],
        pending=progress["pending"],
        failed=progress["failed"],
        target_count=target_count(settings, len(decisions)),
    )


def _decided_by(session: Session, ids: list[int]) -> dict[int, str | None]:
    if not ids:
        return {}
    rows = session.execute(select(Photo.id, Photo.cull_decided_by).where(Photo.id.in_(ids)))
    return {row.id: (row.cull_decided_by.value if row.cull_decided_by else None) for row in rows}


def _decision_out(decision: Decision, decided_by: str | None) -> CullDecision:
    return CullDecision(
        id=decision.id,
        culled=decision.culled,
        reasons=list(decision.reasons),
        score=decision.score,
        rank=decision.rank,
        criteria=dict(decision.criteria),
        decided_by=decided_by,
    )


def _selection(
    session: Session, project: Project, state: service.CullingState, *, everything: bool
) -> CullSelection:
    decisions = [d for d in state.decisions if everything or d.id in state.changed]
    decided = _decided_by(session, [d.id for d in decisions])
    return CullSelection(
        summary=_summary(session, project, state.decisions, state.settings),
        settings=_settings_out(state.settings),
        decisions=[_decision_out(d, decided.get(d.id)) for d in decisions],
    )


def _focus_point(features: dict) -> tuple[float, float] | None:
    point = (features.get("camera") or {}).get("focus_point")
    if point:
        return (float(point[0]), float(point[1]))
    region = (features.get("sharpness") or {}).get("region")
    if region:
        x, y, w, h = region
        return (round(x + w / 2, 4), round(y + h / 2, 4))
    return None


@router.get("/projects/{project_id}/culling", response_model=CullView)
def read_culling(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> CullView:
    """Everything the culling screen draws. Regroups first if it has to."""
    service.ensure_grouped(session, project)
    state = service.apply_selection(session, project)
    by_id = {d.id: d for d in state.decisions}

    photos = session.scalars(
        select(Photo)
        .where(
            Photo.project_id == project.id,
            Photo.kind == PhotoKind.RAW,
            Photo.missing.is_(False),
        )
        .order_by(Photo.shot_at.is_(None), Photo.shot_at, Photo.filename, Photo.id)
    ).all()

    merges = session.scalars(select(MergeGroup).where(MergeGroup.project_id == project.id)).all()
    active_member: dict[int, int] = {}
    merge_out = []
    for group in merges:
        members = sorted(group.members, key=lambda m: m.position)
        if group.decision in ACTIVE_MERGES:
            for member in members:
                active_member[member.photo_id] = group.id
        reference = next((m.photo_id for m in members if m.role.value == "reference"), None)
        merge_out.append(
            CullMerge(
                id=group.id,
                kind=group.kind,
                decision=group.decision,
                confidence=group.confidence,
                reasons=(group.detect_reasons or [{}])[0],
                members=[m.photo_id for m in members],
                reference=reference,
            )
        )

    items: list[CullPhoto] = []
    bursts: dict[int, list[tuple[int, int]]] = {}
    for photo in photos:
        decision = by_id[photo.id]
        features = photo.culling_features or {}
        preview = features.get("preview") or {}
        if photo.burst_group_id is not None:
            bursts.setdefault(photo.burst_group_id, []).append((decision.rank, photo.id))
        items.append(
            CullPhoto(
                **_decision_out(
                    decision, photo.cull_decided_by.value if photo.cull_decided_by else None
                ).model_dump(),
                filename=photo.filename,
                shot_at=photo.shot_at,
                status=photo.status,
                missing=photo.missing,
                error=photo.error,
                burst_group_id=photo.burst_group_id,
                merge_group_id=active_member.get(photo.id),
                focus_point=_focus_point(features),
                width=preview.get("width"),
                height=preview.get("height"),
                iso=photo.iso,
                aperture=photo.aperture,
                shutter=photo.shutter,
                focal_length=photo.focal_length,
                sharpness=photo.sharpness,
                motion=None if photo.motion_blur is None else 1.0 - photo.motion_blur,
                exposure=photo.exposure_score,
                exposure_side=features.get("exposure_side"),
            )
        )

    return CullView(
        summary=_summary(session, project, state.decisions, state.settings),
        settings=_settings_out(state.settings),
        availability=_availability(),
        photos=items,
        bursts={key: [pid for _, pid in sorted(members)] for key, members in bursts.items()},
        merges=merge_out,
        confirmed=project.status
        not in (ProjectStatus.NEW, ProjectStatus.IMPORTING, ProjectStatus.CULLING),
    )


@router.put("/projects/{project_id}/culling/settings", response_model=CullSelection)
def update_culling_settings(
    payload: CullSettingsUpdate,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> CullSelection:
    """Change mode, target, aggressiveness, weights or toggles. Never re-analyses.

    Raises:
        HTTPException 409: a criterion that needs a model was switched on and
            the model is not available. The message says which and why; the
            toggle in the interface is disabled for the same reason, so this is
            a guard, not a flow.
    """
    current = service.project_settings(project)
    criteria = dict(current.criteria)
    for name, enabled in (payload.criteria or {}).items():
        if name not in criteria:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"criterio sconosciuto: {name}")
        if enabled and name in _MODEL_CRITERIA and not criteria.get(name):
            available, reason, _ = feature_status(name)
            if not available:
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"il criterio «{name}» richiede un modello non disponibile ({reason})",
                )
        criteria[name] = bool(enabled)

    weights = dict(current.weights)
    for name, value in (payload.weights or {}).items():
        if name not in weights:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"peso sconosciuto: {name}")
        weights[name] = max(0.0, float(value))

    mode = payload.mode or current.mode
    fields = payload.model_fields_set
    target = payload.target if "target" in fields else current.target
    if mode.value != "conservative" and target is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "la modalità a obiettivo richiede un obiettivo"
        )
    settings = CullingSettings(
        mode=mode,
        target=target,
        aggressiveness=(
            payload.aggressiveness
            if payload.aggressiveness is not None
            else current.aggressiveness
        ),
        weights=weights,
        criteria=criteria,
    )
    service.ensure_grouped(session, project)
    state = service.apply_selection(session, project, settings)
    # A change of weights or toggles moves every score; send everything then.
    everything = payload.weights is not None or payload.criteria is not None
    return _selection(session, project, state, everything=everything)


@router.post("/projects/{project_id}/culling/decisions", response_model=CullSelection)
def record_decisions(
    payload: CullDecisionsRequest,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> CullSelection:
    """The user's own keep / discard / back-to-automatic, in one batch.

    A batch because the gestures of section 7.6 come in pairs -- changing the
    proposed frame of a burst keeps one photo and discards another -- and the
    undo of a pair must be one step, not two.
    """
    try:
        state = user_decisions.set_decisions(
            session, project, [(item.photo_id, item.decision) for item in payload.decisions]
        )
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    changed = set(state.changed) | {item.photo_id for item in payload.decisions}
    return _selection(
        session,
        project,
        service.CullingState(state.decisions, state.settings, frozenset(changed)),
        everything=False,
    )


@router.post("/projects/{project_id}/culling/start", response_model=dict)
def start_culling(
    payload: CullStartRequest,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    """Queue the analysis of the photos that do not have one (section 7.1).

    Also how a project imported "straight to editing" opens its culling later:
    the choice is reversible, not a dead end.
    """
    queued = service.enqueue_culling(session, project, force=payload.force)
    session.flush()
    return {"queued": queued, **service.state(session, project)}


@router.post("/projects/{project_id}/culling/confirm", response_model=dict)
def confirm_culling(
    project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    """"Procedi con l'editing". The only way from culling to development."""
    service.ensure_grouped(session, project)
    state = service.apply_selection(session, project)
    queued = user_decisions.confirm_selection(session, project)
    session.flush()
    snapshots.take_auto(session, project, snapshots.AUTO_CULLING)
    return {"queued": queued, "selected": state.selected}


@router.get("/photos/{photo_id}/thumb")
def read_thumb(
    photo: Photo = Depends(get_photo),
    size: Literal["grid", "full"] = Query("grid"),
) -> Response:
    """The camera's embedded preview, as cached by the culling analysis.

    Regenerated on the spot when the cache has been emptied (section 20.3):
    extracting it costs a few tens of milliseconds, less than the round trip
    of telling the interface to wait.

    Raises:
        HTTPException 404: the photo has no file (a merge) or the file is gone.
        HTTPException 422: the file carries nothing readable.
    """
    from ..raw.embedded import preview_paths_for, read_embedded_preview, write_preview_cache

    if photo.path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "questa foto non ha un file sorgente")
    identity = photo.hash or f"photo-{photo.id}"
    paths = preview_paths_for(identity)
    if not paths.exist():
        try:
            preview = read_embedded_preview(photo.path, sidecar=photo.sidecar_jpeg_path)
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        paths = write_preview_cache(preview, identity)

    payload = (paths.full if size == "full" else paths.grid).read_bytes()
    return Response(
        content=payload,
        media_type="image/jpeg",
        headers={
            # Named after the content hash, like the proxies: it cannot change
            # under this URL without the photo being a different photo.
            "Cache-Control": "private, max-age=31536000, immutable",
            "ETag": f'"{hashlib.blake2b(payload, digest_size=8).hexdigest()}"',
        },
    )
