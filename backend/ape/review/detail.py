# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One photo as the review screen opens it: every term of its score, and the
alternatives to choose from (sections 9.1 and 9.2).

Computed on request rather than stored: the variants depend on the current
edit, and the user is looking at exactly one photo at a time. It reads only
numbers the catalogue already has -- the proxy is opened only for a photo
predicted before phase 7, whose measurements are then taken once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Photo, Project, StyleProfile
from ..style import vector as sv
from .confidence import DEFAULT_THRESHOLD, score, weights_from
from .reference import sample_clipping
from .service import current_versions, inputs_for
from .variants import VariantInputs, variants

__all__ = ["photo_review"]


def _neutral_vector(session: Session, scene) -> np.ndarray | None:
    from ..style.builtin import NEUTRAL_AUTO
    from ..style.profile import ensure_builtins, load

    ensure_builtins(session)
    neutral_id = session.scalars(
        select(StyleProfile.id).where(
            StyleProfile.builtin.is_(True), StyleProfile.name.startswith(NEUTRAL_AUTO)
        )
    ).first()
    if neutral_id is None:
        return None
    return load(session, neutral_id).predict(scene).vector


def photo_review(session: Session, photo: Photo) -> dict[str, Any]:
    """Score, terms, decision and variants of one photo under review."""
    from ..pipeline.params import EditParams
    from ..style.predict import describe_photo
    from ..style.profile import load

    project = session.get(Project, photo.project_id)
    prediction = photo.prediction or {}
    if project is None or prediction.get("profile_id") != project.style_profile_id:
        return {"available": False}
    version = current_versions(session, [photo.id]).get(photo.id)
    params = EditParams.from_dict(version.params) if version else EditParams()
    profile = load(session, project.style_profile_id)
    row = session.get(StyleProfile, project.style_profile_id)
    clipping = sample_clipping(session, row) if profile.calibration is not None else {}
    result = score(
        inputs_for(photo, params, profile.calibration, clipping),
        weights_from(project.confidence_weights),
    )
    reasons = [t.code for t in result.reasons()]

    options: list[dict[str, Any]] = []
    context = prediction.get("context")
    if context:
        knn = prediction.get("knn_vector")
        neutral = auto_wb = None
        try:
            scene, measured = describe_photo(
                photo.scene_features, photo.embedding, photo.analysis, photo.proxy_path
            )
            neutral = _neutral_vector(session, scene)
            auto_wb = (float(measured["wb_mired_shift"]), float(measured["wb_tint_shift"]))
        except (LookupError, ValueError):
            pass  # not analysed: the variants that need the scene are skipped
        found = variants(
            VariantInputs(
                current=params,
                context=sv.StyleContext(**context),
                reasons=reasons,
                knn_vector=None if knn is None else np.asarray(knn, dtype=np.float64),
                neutral_vector=neutral,
                auto_wb=auto_wb,
            )
        )
        options = [{"key": v.key, "params": v.params.model_dump(mode="json")} for v in found]

    return {
        "available": True,
        "photo_id": photo.id,
        "confidence": round(result.value, 4),
        "threshold": float(project.confidence_threshold or DEFAULT_THRESHOLD),
        "terms": [t.as_json() for t in result.terms],
        "reasons": reasons,
        "review": photo.review,
        "status": photo.status.value,
        "cluster": photo.cluster_id,
        "variants": options,
    }
