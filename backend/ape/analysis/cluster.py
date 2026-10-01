# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scenes: which photos of a project belong together (section 9.2).

Agglomerative clustering, average linkage, cut at a distance rather than at a
number of clusters -- a wedding and a walk in the woods do not have the same
number of scenes, and a fixed ``k`` would split one and merge the other.

The distance between two photos is the distance between what they show plus a
term for when they were taken, because two frames of the same meadow an hour
apart are often two different lights, and the review of section 9.2 corrects
lights, not meadows.

What they show comes from the CLIP embedding when the model is installed,
reduced to 32 dimensions by a PCA fitted on the project itself (section 8.3),
and from the engineered features of ``scene.py`` otherwise -- the clustering
never waits for a download. Each cluster's representative is its medoid: a
real photo, the one closest to all the others, rather than an average nobody
took.

A pure function: vectors and times in, labels and ranks out. The catalogue side
is ``analysis/service.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

__all__ = ["ClusterItem", "ClusterResult", "cluster_photos"]

#: Components kept by the PCA of the embeddings (section 8.3).
PCA_COMPONENTS = 32

#: Where the dendrogram is cut, per kind of description. Measured on the
#: user's 24 frames against 11 pairs that belong together (the two lake shots,
#: the stumps, the portraits at the log pile...) and 13 that do not (lake and
#: stumps, town and forest...). CLIP: from 0.30 to 0.40, 10 of 11 together and
#: 13 of 13 apart, 12 to 14 scenes; at 0.45 one wrong merge appears, so 0.40.
#: The engineered features, standardised with the floors of
#: ``scene.FEATURE_SCALES``: at 0.60, 10 of 11 together and 13 of 13 apart, 13
#: scenes; from 0.65 a wrong merge. (Without the floors the best they managed
#: was 9 of 11 and 12 of 13.)
THRESHOLD_EMBEDDING = 0.40
THRESHOLD_FEATURES = 0.60

#: The time term: ``weight * min(1, gap / horizon)``. Half an hour apart adds a
#: third of the embedding threshold -- enough to split two visits to the same
#: place, not enough to split a burst from its scene.
TIME_WEIGHT = 0.12
TIME_HORIZON_S = 1800.0


@dataclass(slots=True)
class ClusterItem:
    photo_id: int
    embedding: np.ndarray | None
    features: np.ndarray | None
    shot_at: datetime | None


@dataclass(slots=True)
class ClusterResult:
    #: ``photo_id -> cluster``, clusters numbered from 1 in shooting order.
    labels: dict[int, int]
    #: ``photo_id -> rank`` by distance to the cluster's medoid; 0 is the medoid.
    ranks: dict[int, int]
    #: ``embedding`` or ``features``: what the distances were computed on.
    basis: str


def _pca(vectors: np.ndarray, components: int) -> np.ndarray:
    centred = vectors - vectors.mean(axis=0)
    if len(vectors) <= 2:
        return centred
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    reduced = centred @ vt[: min(components, vt.shape[0])].T
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    return reduced / np.maximum(norms, 1e-9)


def _standardise(features: np.ndarray) -> np.ndarray:
    """Robust z-scores, NaN -> 0 (the median), scaled so distances are O(1).

    The spread is the interquartile range, never below the feature's own floor
    in ``scene.FEATURE_SCALES``: in a project shot in one light, the colour
    temperature must not become the most discriminating feature just because
    its noise is all that varies.
    """
    from .scene import FEATURE_NAMES, FEATURE_SCALES

    median = np.nanmedian(features, axis=0)
    spread = np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0)
    floors = np.array([FEATURE_SCALES[name] for name in FEATURE_NAMES], dtype=np.float64)
    spread = np.where(np.isfinite(spread), np.maximum(spread, floors), floors)
    z = (features - median) / spread
    z = np.where(np.isfinite(z), z, 0.0)
    return np.clip(z, -4.0, 4.0) / np.sqrt(features.shape[1])


def _content_distances(items: list[ClusterItem]) -> tuple[np.ndarray, str]:
    from scipy.spatial.distance import pdist

    if all(item.embedding is not None for item in items):
        vectors = np.stack([item.embedding for item in items]).astype(np.float64)
        reduced = _pca(vectors, PCA_COMPONENTS)
        return pdist(reduced, metric="cosine") / 2.0, "embedding"
    features = np.stack(
        [
            item.features
            if item.features is not None
            else np.full(len(next(i.features for i in items if i.features is not None)), np.nan)
            for item in items
        ]
    ).astype(np.float64)
    return pdist(_standardise(features), metric="euclidean"), "features"


def _time_distances(items: list[ClusterItem]) -> np.ndarray:
    from scipy.spatial.distance import pdist

    seconds = np.array(
        [item.shot_at.timestamp() if item.shot_at else np.nan for item in items], dtype=np.float64
    )
    gaps = pdist(seconds[:, None], metric="cityblock")
    # An unknown time says nothing either way.
    gaps = np.where(np.isfinite(gaps), gaps, 0.0)
    return TIME_WEIGHT * np.minimum(1.0, gaps / TIME_HORIZON_S)


def cluster_photos(items: list[ClusterItem]) -> ClusterResult:
    """Group photos into scenes and rank each scene's photos from its medoid.

    Items without an embedding *or* features cannot be placed and are left
    out; the caller shows them as "non analizzate".
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    usable = [item for item in items if item.embedding is not None or item.features is not None]
    if not usable:
        return ClusterResult({}, {}, "features")
    if len(usable) == 1:
        return ClusterResult({usable[0].photo_id: 1}, {usable[0].photo_id: 0}, "features")

    content, basis = _content_distances(usable)
    distances = content + _time_distances(usable)
    threshold = THRESHOLD_EMBEDDING if basis == "embedding" else THRESHOLD_FEATURES
    tree = linkage(distances, method="average")
    raw = fcluster(tree, t=threshold, criterion="distance")

    matrix = squareform(distances)
    # Number clusters in shooting order, so "scene 1" is the first one taken.
    first_shot: dict[int, float] = {}
    for index, label in enumerate(raw):
        when = usable[index].shot_at.timestamp() if usable[index].shot_at else float(index)
        first_shot[label] = min(first_shot.get(label, when), when)
    renumber = {label: n + 1 for n, label in enumerate(sorted(first_shot, key=first_shot.get))}

    labels: dict[int, int] = {}
    ranks: dict[int, int] = {}
    for label in renumber:
        members = np.nonzero(raw == label)[0]
        sub = matrix[np.ix_(members, members)]
        medoid = members[int(np.argmin(sub.sum(axis=1)))]
        order = members[np.argsort(matrix[medoid, members], kind="stable")]
        for rank, index in enumerate(order):
            labels[usable[index].photo_id] = renumber[label]
            ranks[usable[index].photo_id] = rank
    return ClusterResult(labels, ranks, basis)
