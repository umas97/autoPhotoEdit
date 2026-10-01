# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``StyleProfile`` rows as something that predicts (section 8.3, section 22).

A profile is either **learned** -- a :class:`~.model.StyleModel` in
``StyleProfile.model`` -- or **rule-based**: a built-in of section 22, or a
user's copy of one, whose ``builtin_rules`` hold a :class:`~.builtin.BuiltinRules`.
:class:`LoadedProfile` hides the difference: both give a style vector for a
:class:`~.sample.SceneDescription`, and everything downstream is one code path.

This module also owns:

* :func:`ensure_builtins` -- the four built-ins exist in every catalogue, and
  their rules follow the program (they are ours; a user who wants different
  ones edits a copy);
* :func:`retrain` -- the model refitted on the samples that are ready and not
  excluded, plus what section 8.4 needs to match the profile to a project
  (the embedding centroid) and what the interface shows (validation, methods).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..analysis.embed import decode_embedding, encode_embedding
from ..analysis.scene import decode_features
from ..db.base import utcnow
from ..db.models import StyleProfile, StyleSample, StyleSampleStatus
from . import builtin
from .model import METHOD_NAMES, StyleModel, TrainingSample
from .sample import SceneDescription
from .train import train

__all__ = [
    "Calibration",
    "LoadedProfile",
    "MIN_PAIRS",
    "RECOMMENDED_PAIRS",
    "StylePrediction",
    "ensure_builtins",
    "load",
    "retrain",
    "retrain_centroid",
    "training_samples",
]

_log = logging.getLogger(__name__)

#: Section 8.1: below 8 pairs the profile works but the interface warns.
MIN_PAIRS = 8
RECOMMENDED_PAIRS = 30


@dataclass(slots=True)
class StylePrediction:
    vector: np.ndarray
    #: ``[(sample_id, weight), ...]`` -- the transparency of section 8.3.
    neighbours: list[tuple[int, float]]
    nearest: float | None
    dispersion: float | None
    method: str  # "model" or "rules"
    #: The k-NN's own vector (``model.Prediction.knn_vector``), for the variants.
    knn_vector: np.ndarray | None = None

    def as_json(self) -> dict:
        return {
            "vector": [round(float(x), 5) for x in self.vector],
            "neighbours": [[int(i), round(float(w), 4)] for i, w in self.neighbours],
            "nearest": None if self.nearest is None else round(self.nearest, 4),
            "dispersion": None if self.dispersion is None else round(self.dispersion, 4),
            "method": self.method,
            "knn_vector": None
            if self.knn_vector is None
            else [round(float(x), 5) for x in self.knn_vector],
        }


@dataclass(frozen=True, slots=True)
class Calibration:
    """How spread out a learned profile's own samples are (section 9.1).

    The yardsticks of the confidence: a photo is "far" from the samples when
    its nearest one is further than most samples are from each other, and
    its neighbours "disagree" when they disagree more than most samples'
    neighbours do.
    """

    #: 90th percentile of the samples' leave-one-out nearest distance.
    nearest_p90: float
    #: 90th percentile of their neighbours' dispersion.
    dispersion_p90: float
    #: Range of the samples' exposure anchors, EV.
    anchor_min: float
    anchor_max: float


@dataclass(slots=True)
class LoadedProfile:
    id: int
    name: str
    builtin: bool
    updated_at: datetime | None
    rules: builtin.BuiltinRules | None
    model: StyleModel | None
    _calibration: Calibration | None = None

    @property
    def usable(self) -> bool:
        return self.model is not None or self.rules is not None

    @property
    def calibration(self) -> Calibration | None:
        """``None`` for a profile of rules, or one with fewer than 3 samples."""
        if self._calibration is None and self.model is not None and len(self.model.sample_ids) >= 3:
            nearest, dispersion = self.model.spacing()
            anchors = np.asarray(self.model.anchors, dtype=np.float64)
            self._calibration = Calibration(
                nearest_p90=float(np.nanpercentile(nearest, 90)),
                dispersion_p90=float(np.nanpercentile(dispersion, 90)),
                anchor_min=float(anchors.min()),
                anchor_max=float(anchors.max()),
            )
        return self._calibration

    def predict(self, scene: SceneDescription) -> StylePrediction:
        """The style vector this profile gives a photo.

        Raises:
            ValueError: a learned profile that has not been trained yet.
        """
        if self.model is not None:
            p = self.model.predict(
                scene.features, scene.embedding, scene.context.exposure_anchor_ev
            )
            return StylePrediction(
                vector=p.vector,
                neighbours=[(n.sample_id, n.weight) for n in p.neighbours],
                nearest=p.nearest,
                dispersion=p.dispersion,
                method="model",
                knn_vector=p.knn_vector,
            )
        if self.rules is not None:
            vector = builtin.predict(
                self.rules,
                scene.features,
                exposure_anchor_ev=scene.context.exposure_anchor_ev,
                auto_wb_mired=scene.auto.wb_mired_shift,
                auto_wb_tint=scene.auto.wb_tint_shift,
            )
            return StylePrediction(vector, [], None, None, "rules")
        raise ValueError(f"il profilo «{self.name}» non è ancora addestrato")


_cache: dict[int, LoadedProfile] = {}


def load(session: Session, profile_id: int) -> LoadedProfile:
    """The profile, deserialised once per version of the row.

    Raises:
        LookupError: no such profile.
    """
    row = session.get(StyleProfile, profile_id)
    if row is None:
        raise LookupError(f"profilo {profile_id} non trovato")
    cached = _cache.get(profile_id)
    if cached is not None and cached.updated_at == row.updated_at:
        return cached
    model = None
    if row.model:
        try:
            model = StyleModel.from_bytes(row.model)
        except ValueError as exc:
            _log.warning("profilo %s: %s", row.name, exc)
    rules = builtin.BuiltinRules.from_json(row.builtin_rules) if row.builtin_rules else None
    loaded = LoadedProfile(row.id, row.name, bool(row.builtin), row.updated_at, rules, model)
    _cache[profile_id] = loaded
    return loaded


def ensure_builtins(session: Session) -> None:
    """Create the built-in profiles, or bring their rules up to this build's."""
    rows = {
        row.name: row
        for row in session.scalars(select(StyleProfile).where(StyleProfile.builtin.is_(True)))
    }
    for name in builtin.BUILTIN_NAMES:
        rules = builtin.RULES[name].as_json()
        row = rows.get(name) or rows.get(f"{name} (predefinito)")
        if row is None:
            label = name
            if session.scalars(select(StyleProfile).where(StyleProfile.name == name)).first():
                # A user profile took the name first: it keeps it, and the
                # built-in says what it is.
                label = f"{name} (predefinito)"
            session.add(
                StyleProfile(
                    name=label,
                    notes=builtin.DESCRIPTIONS[name],
                    builtin=True,
                    builtin_rules=rules,
                )
            )
        elif row.builtin_rules != rules:
            row.builtin_rules = rules
            row.notes = builtin.DESCRIPTIONS[name]
    session.flush()


def training_samples(session: Session, profile_id: int) -> list[StyleSample]:
    return list(
        session.scalars(
            select(StyleSample)
            .where(
                StyleSample.profile_id == profile_id,
                StyleSample.status == StyleSampleStatus.READY,
                StyleSample.excluded.is_(False),
            )
            .order_by(StyleSample.id)
        )
    )


def retrain(session: Session, profile: StyleProfile) -> bool:
    """Refit the model on the profile's usable samples. Returns whether it has one.

    Fewer than two usable samples leaves the profile untrained (``model`` NULL),
    which the interface shows as such.
    """
    if profile.builtin:
        raise ValueError("un profilo predefinito non si addestra: duplicalo")
    rows = training_samples(session, profile.id)
    samples = []
    embeddings = []
    for row in rows:
        features = decode_features(row.scene_features)
        if features is None or row.vector is None:
            continue
        embedding = decode_embedding(row.embedding)
        if embedding is not None:
            embeddings.append(embedding)
        samples.append(
            TrainingSample(
                sample_id=row.id,
                vector=np.asarray(row.vector, dtype=np.float64),
                features=features,
                embedding=embedding,
                exposure_anchor_ev=float((row.context or {}).get("exposure_anchor_ev", 0.0)),
            )
        )
    profile.n_pairs = len(samples)
    if len(samples) < 2:
        profile.model = None
        profile.norm_stats = None
        profile.embedding_centroid = None
        profile.trained_at = None
        return False
    model = train(samples)
    profile.model = model.to_bytes()
    profile.pca = None  # the PCA travels inside the model blob
    profile.trained_at = utcnow()
    profile.norm_stats = {
        "methods": {
            method: int(np.sum(model.methods == index)) for index, method in enumerate(METHOD_NAMES)
        },
        "embedding_weight": model.embedding_weight,
        "ridge_lambda": model.ridge_lambda,
        "validation": model.validation,
        "with_embedding": model.embeddings_z is not None,
    }
    profile.embedding_centroid = _centroid(embeddings)
    _cache.pop(profile.id, None)
    return True


def _centroid(embeddings: list[np.ndarray]) -> bytes | None:
    if not embeddings:
        return None
    centroid = np.mean(embeddings, axis=0)
    norm = float(np.linalg.norm(centroid))
    return encode_embedding(centroid / norm if norm else centroid)


def retrain_centroid(session: Session, profile: StyleProfile) -> None:
    """The embedding centroid of section 8.4 from the usable samples, model untouched."""
    embeddings = [
        e
        for e in (decode_embedding(r.embedding) for r in training_samples(session, profile.id))
        if e is not None
    ]
    profile.embedding_centroid = _centroid(embeddings)
    _cache.pop(profile.id, None)
