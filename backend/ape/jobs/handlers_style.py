# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The jobs of phase 6: learning a style, and predicting with one.

``style_pair``
    one training pair: decode, register, invert (``style/sample.py``,
    ``style/learn.py``), about 10 s of one core. When it is the last pair of
    its profile still waiting, the same worker refits the profile's model --
    a fraction of a second -- so the profile is ready the moment its last pair
    is.
``predict``
    one photo of a project: the exposure anchor and automatic white balance
    measured on its proxy (``style/auto.py``), then the profile's prediction,
    stored on the photo. The *versions* are written later and all at once by
    ``style/apply.py``, because coherence (section 8.4) needs every
    prediction of the project before it can move any of them.

Both follow the rules of ``handlers.py``: read, work outside the session,
write; raise with a sentence a person can read; never write near a source.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import func, select

from ..db.models import (
    JobKind,
    Photo,
    StyleProfile,
    StyleSample,
    StyleSampleStatus,
)
from ..db.session import session_scope
from .handlers import _maker
from .queue import JobRecord, enqueue
from .worker import register_handler

__all__ = [
    "UNREPRODUCIBLE_DELTA_E",
    "enqueue_pair",
    "enqueue_prediction",
    "run_predict",
    "run_style_pair",
]

_log = logging.getLogger(__name__)

#: Section 8.2.5: a pair the pipeline cannot reproduce is not trained on.
#: Mean ΔE of the inversion (at the comparison scale); the user's 65
#: Lightroom pairs all came in under 6.3, so 8 marks an edit that uses
#: something this pipeline does not have (a local mask, a monochrome mix, a
#: heavy creative profile) rather than an ordinary one.
UNREPRODUCIBLE_DELTA_E = 8.0


def enqueue_pair(session, sample: StyleSample) -> bool:
    return (
        enqueue(
            session,
            JobKind.STYLE_PAIR,
            {"sample_id": sample.id},
            dedupe_key=f"style_pair:{sample.id}",
        )
        is not None
    )


def enqueue_prediction(session, photo: Photo, profile_id: int) -> bool:
    return (
        enqueue(
            session,
            JobKind.PREDICT,
            {"photo_id": photo.id, "profile_id": profile_id},
            project_id=photo.project_id,
            dedupe_key=f"predict:{photo.id}:{profile_id}",
        )
        is not None
    )


def _lens_override(session, raw: Path):
    from ..raw.metadata import read_optics
    from .handlers_analysis import lens_override_for

    return lens_override_for(session, read_optics(raw).lens_model)


@register_handler(JobKind.STYLE_PAIR)
def run_style_pair(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Prepare and invert one pair; retrain the profile when it is the last one.

    Raises:
        LookupError: the sample is gone (profile deleted while queued).
    """
    from ..analysis.embed import encode_embedding
    from ..analysis.scene import encode_features
    from ..style import learn, sample
    from ..style import vector as sv
    from ..style.align import AlignmentError
    from ..style.profile import retrain

    maker = _maker(record.payload.get("db_path"))
    sample_id = int(record.payload["sample_id"])
    with session_scope(maker) as session:
        row = session.get(StyleSample, sample_id)
        if row is None:
            raise LookupError(f"coppia {sample_id} non più nel catalogo")
        raw, reference = Path(row.raw_path or ""), Path(row.reference_path or "")
        profile_id = row.profile_id
        override = _lens_override(session, raw) if raw.is_file() else None

    try:
        prepared = sample.prepare(raw, reference, lens_override=override)
        progress(0.3)
        result = learn.invert(prepared.images)
    except (FileNotFoundError, ValueError, AlignmentError) as exc:
        with session_scope(maker) as session:
            row = session.get(StyleSample, sample_id)
            if row is not None:
                row.status = StyleSampleStatus.FAILED
                row.error = str(exc)[:1000]
        _maybe_retrain(maker, profile_id, retrain)
        return  # a bad pair is an outcome, not a job failure to retry
    progress(0.95)

    context = prepared.scene.context
    params = sv.to_params(result.vector, context)
    with session_scope(maker) as session:
        row = session.get(StyleSample, sample_id)
        if row is None:
            return
        row.vector = [round(float(x), 6) for x in result.vector]
        row.params = params.model_dump(mode="json")
        row.context = {
            "as_shot_temperature_k": context.as_shot_temperature_k,
            "as_shot_tint": context.as_shot_tint,
            "exposure_anchor_ev": context.exposure_anchor_ev,
            "rotation_deg": round(prepared.alignment.rotation_deg, 3),
            "coverage": round(prepared.alignment.coverage, 3),
            "tails": prepared.tails,
        }
        row.scene_features = encode_features(prepared.scene.features)
        row.embedding = (
            None if prepared.scene.embedding is None else encode_embedding(prepared.scene.embedding)
        )
        row.thumbnail = prepared.thumbnail_jpeg
        row.residual_loss = float(result.loss)
        row.delta_e = float(result.delta_e)
        row.error = None
        if result.delta_e > UNREPRODUCIBLE_DELTA_E:
            row.status = StyleSampleStatus.UNREPRODUCIBLE
            row.error = (
                f"la pipeline non riesce a riprodurre questo edit "
                f"(ΔE residuo {result.delta_e:.1f}): escluso dall'addestramento"
            )
        else:
            row.status = StyleSampleStatus.READY
    _log.info(
        "coppia %d: ΔE %.2f in %d render, %.1f s",
        sample_id,
        result.delta_e,
        result.evaluations,
        result.seconds,
    )
    _maybe_retrain(maker, profile_id, retrain)


def _maybe_retrain(maker, profile_id: int, retrain) -> None:
    with session_scope(maker) as session:
        pending = session.scalar(
            select(func.count(StyleSample.id)).where(
                StyleSample.profile_id == profile_id,
                StyleSample.status == StyleSampleStatus.PENDING,
            )
        )
        profile = session.get(StyleProfile, profile_id)
        if profile is not None and not pending and not profile.builtin:
            retrain(session, profile)


@register_handler(JobKind.PREDICT)
def run_predict(record: JobRecord, progress: Callable[[float], None]) -> None:
    """Predict one photo's style and store it on the photo (no version yet).

    Raises:
        LookupError: the photo or the profile is gone.
        ValueError: the photo has not been analysed, or the profile is untrained.
    """
    from ..style.predict import describe_photo
    from ..style.profile import load

    maker = _maker(record.payload.get("db_path"))
    photo_id = int(record.payload["photo_id"])
    profile_id = int(record.payload["profile_id"])
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:
            raise LookupError(f"foto {photo_id} non più nel catalogo")
        inputs = (
            photo.scene_features,
            photo.embedding,
            dict(photo.analysis or {}),
            photo.proxy_path,
        )
        profile = load(session, profile_id)
    scene, measured = describe_photo(*inputs)
    context = scene.context
    progress(0.5)
    prediction = profile.predict(scene)
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None:
            return
        analysis = dict(photo.analysis or {})
        analysis["auto"] = measured
        photo.analysis = analysis
        photo.prediction = {
            "profile_id": profile_id,
            "profile_updated": profile.updated_at.isoformat() if profile.updated_at else None,
            **prediction.as_json(),
            "context": {
                "as_shot_temperature_k": context.as_shot_temperature_k,
                "as_shot_tint": context.as_shot_tint,
                "exposure_anchor_ev": context.exposure_anchor_ev,
            },
            # Kept from a previous prediction: what was last written as a
            # version, so a user edit since can be told from our own.
            **{k: v for k, v in (photo.prediction or {}).items() if k in ("applied_signature",)},
            "applied": False,
        }
