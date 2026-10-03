# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``analyze`` job of phase 5: what the scene is and how the frame sits.

It runs after the proxy, on the proxy -- the neutral, lens-corrected 2048 px
JPEG -- and never on the RAW (section 26: prefer the proxy to the full
resolution). One pass over one small image answers everything section 12
budgets at 0.6 s:

* the straightening angle and how sure it is (``analysis/straighten``);
* a crop proposal on the straightened frame, unless the project stopped
  wanting them (``analysis/crop``);
* the engineered scene features and, when the model is installed, the CLIP
  embedding (``analysis/scene``, ``analysis/embed``).

What it writes is data about the photo, never a decision over the user's: the
straightening becomes the rotation of the photo's *first* version only if it
has no version yet, and a crop is only ever a pending proposal.

The payload ``{"photo_id": n, "only": "embedding"}`` computes the embedding
alone, for the photos analysed before the model was downloaded; with
``"only": "crop"``, the crop proposal alone, for the photos whose pending one
an older build made in a ratio no longer proposed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sqlalchemy import select

from ..db.models import (
    CropDecision,
    CropProposal,
    EditVersion,
    EditVersionSource,
    JobKind,
    LensProfileOverride,
    Photo,
    PhotoStatus,
    Project,
)
from ..db.session import session_scope
from ..raw.source import pixels_of
from .handlers import _maker, _mark_failed, available_pixels
from .queue import JobRecord, enqueue
from .worker import register_handler

__all__ = ["enqueue_analysis", "lens_override_for", "run_analyze"]

_log = logging.getLogger(__name__)

#: Bumped when anything the analysis writes changes meaning; a photo analysed by
#: an older version is analysed again.
ANALYSIS_VERSION = 1


def lens_override_for(session, lens_model: str | None) -> tuple[str, str] | None:
    """The lensfun profile the user associated with this EXIF lens name, if any."""
    if not lens_model:
        return None
    row = session.get(LensProfileOverride, lens_model)
    return None if row is None else (row.lensfun_maker, row.lensfun_model)


def enqueue_analysis(session, photo: Photo, *, only: str | None = None) -> bool:
    payload: dict[str, object] = {"photo_id": photo.id}
    key = f"analyze:{photo.id}"
    if only:
        payload["only"] = only
        key += f":{only}"
    return enqueue(
        session, JobKind.ANALYZE, payload, project_id=photo.project_id, dedupe_key=key
    ) is not None


def _read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _as_shot(source: Path) -> dict[str, float] | None:
    """The camera's white balance as temperature and tint, without decoding."""
    import rawpy

    from ..raw.decode import read_camera_color

    try:
        with rawpy.imread(str(source)) as raw:
            camera = read_camera_color(raw)
    except Exception as exc:  # a RAW LibRaw cannot open fails the proxy, not this
        _log.debug("bilanciamento as-shot non leggibile da %s: %s", source.name, exc)
        return None
    return {
        "temperature_k": round(float(camera.as_shot_temperature_k), 1),
        "tint": round(float(camera.as_shot_tint), 2),
    }


@register_handler(JobKind.ANALYZE)
def run_analyze(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Analyse one photo on its proxy (see the module docstring).

    Raises:
        LookupError: the row is gone.
        FileNotFoundError: the RAW is gone and the proxy has to be rebuilt.
    """
    from ..analysis import embed, straighten
    from ..analysis.scene import SceneInputs, encode_features, scene_features
    from ..raw.metadata import read_optics
    from ..raw.proxy import build_proxy, is_current

    maker = _maker(record.payload.get("db_path"))
    photo_id = int(record.payload["photo_id"])
    only = record.payload.get("only")

    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:
            raise LookupError(f"foto {photo_id} non più nel catalogo")
        source = pixels_of(photo)
        if source is None:
            raise LookupError(f"foto {photo_id} non ha un file sorgente")
        proxy_path = Path(photo.proxy_path) if photo.proxy_path else None
        identity = photo.hash or f"photo-{photo.id}"
        override = lens_override_for(session, photo.lens)
        analysis = dict(photo.analysis or {})
        project = session.get(Project, photo.project_id)
        paused = bool(project and project.crop_proposals_paused)
        decided = session.scalars(
            select(CropProposal.id).where(
                CropProposal.photo_id == photo_id,
                CropProposal.decision != CropDecision.PENDING,
            )
        ).first() is not None
        exif = SceneInputs(
            iso=photo.iso, aperture=photo.aperture, shutter=photo.shutter,
            focal_length=photo.focal_length,
        )

    # The proxy is the input. One written before lens correction existed shows
    # bent lines, and the straightening would measure the lens, not the camera.
    if not is_current(proxy_path):
        source = available_pixels(maker, photo_id, source)
        try:
            proxy_path = build_proxy(source, identity, force=True, lens_override=override).path
        except Exception as exc:
            _mark_failed(maker, photo_id, f"{type(exc).__name__}: {exc}")
            raise
    image = _read_rgb(proxy_path)
    progress(0.2)

    if only == "crop":
        proposal = None
        if not paused and not decided:
            rotation = float((analysis.get("straighten") or {}).get("rotation_deg") or 0.0)
            proposal = _propose(image, rotation)
        with session_scope(maker) as session:
            if session.get(Photo, photo_id) is not None:
                _write_proposal(session, photo_id, proposal)
        return

    if only == "embedding":
        vector = embed.embed(image)
        if vector is not None:
            with session_scope(maker) as session:
                photo = session.get(Photo, photo_id)
                if photo is not None:
                    photo.embedding = embed.encode_embedding(vector)
                    photo.cluster_id = None  # regroup at the next look
        return

    optics = read_optics(source) if source.is_file() else None
    from ..lensdb import LensIdentity, resolve

    lens_profile = None
    if optics is not None and not optics.optics_corrected:
        lens_profile = resolve(
            LensIdentity(
                optics.camera_make, optics.camera_model, optics.lens_model,
                optics.focal_length, optics.aperture, override=override,
            )
        )
    as_shot = analysis.get("as_shot") or (_as_shot(source) if source.is_file() else None)
    if as_shot:
        exif.as_shot_temperature_k = as_shot["temperature_k"]
        exif.as_shot_tint = as_shot["tint"]

    level = straighten.estimate(image)
    progress(0.5)
    proposal = None
    if not paused and not decided:
        proposal = _propose(image, level.rotation_deg)
    progress(0.7)
    features = scene_features(image, exif)
    vector = embed.embed(image)
    progress(0.9)

    analysis.update(
        version=ANALYSIS_VERSION,
        straighten=level.as_json(),
        lens=(
            # Corrected before these pixels existed: nothing missing, nothing to associate.
            {"maker": "", "model": optics.lens_model or "", "source": "applied"}
            if optics is not None and optics.optics_corrected
            else None if lens_profile is None else lens_profile.as_json()
        ),
        lens_model=optics.lens_model if optics is not None else None,
        as_shot=as_shot,
    )
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:
            return
        photo.proxy_path = str(proxy_path)
        photo.analysis = analysis
        photo.scene_features = encode_features(features)
        if vector is not None:
            photo.embedding = embed.encode_embedding(vector)
        photo.cluster_id = None  # "analysed since the last clustering"
        photo.cluster_rank = None
        photo.error = None
        if photo.status in (PhotoStatus.IMPORTED, PhotoStatus.CULLED, PhotoStatus.FAILED):
            photo.status = PhotoStatus.ANALYZED
        _write_proposal(session, photo_id, proposal)
        _first_version(session, photo_id, level.rotation_deg)
        project = session.get(Project, photo.project_id)
        if project is not None and project.style_profile_id is not None:
            # A photo analysed after the project chose its style (a late
            # import, a photo recovered from the discards) gets predicted too.
            from .handlers_style import enqueue_prediction

            enqueue_prediction(session, photo, project.style_profile_id)
    progress(1.0)


def _propose(image: np.ndarray, rotation_deg: float):
    from ..analysis.crop import propose_crop
    from ..pipeline import geometry
    from ..pipeline.params import GeometryParams

    # The crop is relative to the straightened frame, so it is looked for in it.
    frame = image
    if rotation_deg:
        frame = geometry.apply(
            image.astype(np.float32) / 255.0, GeometryParams(rotation_deg=rotation_deg)
        )
    # 512 px is plenty for saliency, and the crop search is per pixel of it.
    import cv2

    height, width = frame.shape[:2]
    scale = 512 / max(height, width)
    if scale < 1:
        frame = cv2.resize(
            frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    return propose_crop(frame)


def _write_proposal(session, photo_id: int, proposal) -> None:
    """Replace the pending proposal; never touch one the user decided on.

    Nor one that removes empty borders: whoever produced the pixels knew
    exactly which ones are empty (``analysis/borders.py``), and a composition
    guessed from the picture would keep them.
    """
    from ..analysis.borders import BORDERS_ASPECT

    pending = session.scalars(
        select(CropProposal).where(
            CropProposal.photo_id == photo_id, CropProposal.decision == CropDecision.PENDING
        )
    ).all()
    if any(old.aspect == BORDERS_ASPECT for old in pending):
        return
    for old in pending:
        session.delete(old)
    if proposal is not None:
        session.add(
            CropProposal(
                photo_id=photo_id,
                rect=proposal.rect(),
                aspect=proposal.aspect,
                score=proposal.score,
                decision=CropDecision.PENDING,
            )
        )


def _first_version(session, photo_id: int, rotation_deg: float) -> None:
    """Give a photo with no history its first version: neutral, straightened.

    Only the first. A photo that already has versions has them because the user
    or a later phase wrote them, and the analysis does not overrule either.
    """
    from ..pipeline.params import PARAMS_VERSION, neutral_params

    exists = session.scalars(
        select(EditVersion.id).where(EditVersion.photo_id == photo_id)
    ).first()
    if exists is not None:
        return
    params = neutral_params()
    params.geometry.rotation_deg = round(float(rotation_deg), 3)
    session.add(
        EditVersion(
            photo_id=photo_id,
            params=params.model_dump(mode="json"),
            params_version=PARAMS_VERSION,
            source=EditVersionSource.PREDICTED,
            is_current=True,
        )
    )
