# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fitting a :class:`~.model.StyleModel`, and choosing its regressions.

Every candidate is scored by leave-one-out on the profile's own samples, in the
perceptual units of ``vector.UNITS``, and the best one is kept per parameter
(see ``model.py`` for what the candidates are and the numbers behind them).
"""

from __future__ import annotations

import numpy as np

from . import boost
from . import vector as sv
from .model import (
    _ANCHOR_LAMBDA,
    _EMBEDDING_WEIGHTS,
    _FEATURE_FLOORS,
    _RIDGE_LAMBDAS,
    ANCHOR,
    BOOST,
    BOOSTING_MIN_PAIRS,
    K_NEIGHBOURS,
    KNN,
    METHOD_NAMES,
    PCA_COMPONENTS,
    RIDGE,
    StyleModel,
    TrainingSample,
    _knn_weights,
    _ridge_apply,
    _ridge_fit,
)

__all__ = ["train"]


def _standardisation(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust centre and spread, never below the per-feature floors."""
    centre = np.nanmedian(features, axis=0)
    q75, q25 = np.nanpercentile(features, [75, 25], axis=0)
    spread = (q75 - q25) / 1.349  # the IQR of a normal is 1.349 sigma
    scale = np.maximum(np.nan_to_num(spread, nan=0.0), _FEATURE_FLOORS)
    return np.nan_to_num(centre, nan=0.0), scale


def _pca(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = embeddings.mean(axis=0)
    _u, _s, vt = np.linalg.svd(embeddings - mean, full_matrices=False)
    k = min(PCA_COMPONENTS, len(embeddings) - 2, vt.shape[0])
    components = vt[: max(k, 1)]
    scores = (embeddings - mean) @ components.T
    scale = scores.std(axis=0)
    return mean, components, np.where(scale > 1e-9, scale, 1.0)


def train(samples: list[TrainingSample]) -> StyleModel:
    """Fit a style model on the samples of a profile.

    Raises:
        ValueError: with fewer than two samples -- there is nothing to choose
            between, and a one-sample "profile" is a preset.
    """
    if len(samples) < 2:
        raise ValueError("servono almeno due coppie per addestrare un profilo")
    ids = np.array([s.sample_id for s in samples], dtype=np.int64)
    y = np.array([s.vector for s in samples], dtype=np.float64)
    features = np.array([s.features for s in samples], dtype=np.float64)
    anchors = np.array([s.exposure_anchor_ev for s in samples], dtype=np.float64)
    centre, scale = _standardisation(features)
    z = np.nan_to_num((features - centre) / scale, nan=0.0)

    have_embeddings = all(s.embedding is not None for s in samples) and len(samples) >= 4
    pca_mean = pca_components = pca_scale = e = None
    if have_embeddings:
        pca_mean, pca_components, pca_scale = _pca(np.array([s.embedding for s in samples]))
        e = ((np.array([s.embedding for s in samples]) - pca_mean) @ pca_components.T) / pca_scale

    model = StyleModel(
        sample_ids=ids,
        vectors=y,
        features_z=z,
        feature_centre=centre,
        feature_scale=scale,
        anchors=anchors,
        embeddings_z=e,
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_scale=pca_scale,
        embedding_weight=0.0,
        ridge={},
        ridge_lambda=_RIDGE_LAMBDAS[1],
        methods=np.zeros(len(sv.NAMES), dtype=np.int64),
    )
    x_ridge = model._ridge_input(z, anchors)
    x_anchor = anchors.reshape(-1, 1)
    n = len(samples)

    # Leave-one-out, per candidate: note that the standardisation and the PCA
    # are the full-sample ones -- refitting them per fold would cost more than
    # the whole training and change the distances by a few percent.
    def loo_knn(weight: float) -> np.ndarray:
        model.embedding_weight = weight
        out = np.empty_like(y)
        for i in range(n):
            distances, _ = model._distances(z[i], None if e is None else e[i])
            distances[i] = np.inf
            order, w = _knn_weights(distances[np.isfinite(distances)], K_NEIGHBOURS)
            keep = np.flatnonzero(np.isfinite(distances))[order]
            out[i] = w @ y[keep]
        return out

    def loo_ridge(x: np.ndarray, lam: float) -> np.ndarray:
        out = np.empty_like(y)
        for i in range(n):
            train_rows = np.arange(n) != i
            fit = _ridge_fit(x[train_rows], y[train_rows], lam)
            out[i] = _ridge_apply(fit, x[i : i + 1])[0]
        return out

    def rms(prediction: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(((prediction - y) / sv.UNITS) ** 2, axis=0))

    weights = _EMBEDDING_WEIGHTS if e is not None else (0.0,)
    knn_scores = {w: rms(loo_knn(w)) for w in weights}
    best_weight = min(knn_scores, key=lambda w: float(knn_scores[w].mean()))
    model.embedding_weight = best_weight
    knn_rms = knn_scores[best_weight]

    # method -> leave-one-out RMS per parameter, in perceptual units.
    scores: dict[int, np.ndarray] = {KNN: knn_rms}
    if n >= 4:
        ridge_scores = {lam: rms(loo_ridge(x_ridge, lam)) for lam in _RIDGE_LAMBDAS}
        model.ridge_lambda = min(ridge_scores, key=lambda lam: float(ridge_scores[lam].mean()))
        scores[RIDGE] = ridge_scores[model.ridge_lambda]
        scores[ANCHOR] = rms(loo_ridge(x_anchor, _ANCHOR_LAMBDA))
        model.anchor_fit = _ridge_fit(x_anchor, y, _ANCHOR_LAMBDA)
    model.ridge = _ridge_fit(x_ridge, y, model.ridge_lambda)

    if n > BOOSTING_MIN_PAIRS:
        units = y / sv.UNITS
        boost_loo = np.empty_like(y)
        for i in range(n):
            rows = np.arange(n) != i
            fitted = boost.fit(x_ridge[rows], units[rows])
            boost_loo[i] = fitted.apply(x_ridge[i : i + 1])[0] * sv.UNITS
        scores[BOOST] = rms(boost_loo)

    methods = sorted(scores)
    table = np.vstack([scores[m] for m in methods])
    # A tie goes to the k-NN (first row): it is the explainable one.
    table[0] -= 1e-9
    model.methods = np.array([methods[i] for i in np.argmin(table, axis=0)], dtype=np.int64)
    if BOOST in scores and np.any(model.methods == BOOST):
        model.boost = boost.fit(x_ridge, y / sv.UNITS)
    model.validation = {METHOD_NAMES[m]: [round(float(v), 4) for v in scores[m]] for m in methods}
    return model
