# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stylistic coherence inside a project (section 8.4).

Predictions are made one photo at a time, and one photo at a time two frames
of the same scene can land on slightly different temperatures -- which a
viewer of the finished set notices before anything else. After every photo
has its prediction, two pulls:

1. **towards the scene**: inside a cluster (``analysis/cluster.py``) every
   vector moves towards the cluster's median by ``lambda`` (the project's
   "Coerenza" slider, default 0.35);
2. **along a sequence**: consecutive shots -- under 60 s apart, focal length
   within 10%, same cluster ("stesso ambiente") -- have their white balance
   and exposure pulled towards the mean of their run by :data:`SEQUENCE_LAMBDA`,
   the stronger smoothing section 8.4 asks for.

White balance is pulled in *absolute* terms, target mired and tint, not as the
shift from as-shot the style vector stores: the camera's auto white balance
wanders from frame to frame, and equal shifts on unequal starting points are
exactly two different temperatures. Exposure is pulled as the offset from the
anchor, which is already brightness-normalised.

The slider at 0 switches both off (section 8.4). The sequence pull scales with
the slider up to its default, so that turning coherence down turns all of it
down. Pure: vectors in, vectors out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from . import vector as sv

__all__ = ["DEFAULT_LAMBDA", "SEQUENCE_LAMBDA", "CoherenceItem", "regularise"]

DEFAULT_LAMBDA = 0.35
SEQUENCE_LAMBDA = 0.6
SEQUENCE_GAP_S = 60.0
SEQUENCE_FOCAL_TOLERANCE = 0.10

_WB = sv.index("wb_mired_shift")
_TINT = sv.index("wb_tint_shift")
_EXPOSURE = sv.index("exposure_offset")


@dataclass(slots=True)
class CoherenceItem:
    photo_id: int
    vector: np.ndarray
    as_shot_temperature_k: float
    as_shot_tint: float
    cluster_id: int | None
    shot_at: datetime | None
    focal_length: float | None


def _to_absolute(items: list[CoherenceItem], vectors: np.ndarray) -> np.ndarray:
    out = vectors.copy()
    for i, item in enumerate(items):
        out[i, _WB] = 1e6 / item.as_shot_temperature_k - vectors[i, _WB]
        out[i, _TINT] = item.as_shot_tint + vectors[i, _TINT]
    return out


def _to_relative(items: list[CoherenceItem], absolute: np.ndarray) -> np.ndarray:
    out = absolute.copy()
    for i, item in enumerate(items):
        out[i, _WB] = 1e6 / item.as_shot_temperature_k - absolute[i, _WB]
        out[i, _TINT] = absolute[i, _TINT] - item.as_shot_tint
    return out


def _same_sequence(a: CoherenceItem, b: CoherenceItem) -> bool:
    if a.shot_at is None or b.shot_at is None or a.cluster_id != b.cluster_id:
        return False
    if abs((b.shot_at - a.shot_at).total_seconds()) >= SEQUENCE_GAP_S:
        return False
    if a.focal_length and b.focal_length:
        return abs(a.focal_length - b.focal_length) <= SEQUENCE_FOCAL_TOLERANCE * max(
            a.focal_length, b.focal_length
        )
    return a.focal_length == b.focal_length


def regularise(items: list[CoherenceItem], lam: float) -> dict[int, np.ndarray]:
    """The coherent vector of every item, keyed by photo id."""
    if not items:
        return {}
    vectors = np.array([item.vector for item in items], dtype=np.float64)
    if lam <= 0:
        return {item.photo_id: vectors[i] for i, item in enumerate(items)}
    lam = float(min(lam, 1.0))
    absolute = _to_absolute(items, vectors)

    clusters: dict[int, list[int]] = {}
    for i, item in enumerate(items):
        if item.cluster_id is not None:
            clusters.setdefault(item.cluster_id, []).append(i)
    pulled = absolute.copy()
    for members in clusters.values():
        if len(members) < 2:
            continue
        median = np.median(absolute[members], axis=0)
        pulled[members] += lam * (median - absolute[members])

    sequence_lam = SEQUENCE_LAMBDA * min(1.0, lam / DEFAULT_LAMBDA)
    order = sorted(
        range(len(items)),
        key=lambda i: (items[i].shot_at is None, items[i].shot_at or datetime.min),
    )
    runs: list[list[int]] = []
    for i in order:
        if runs and _same_sequence(items[runs[-1][-1]], items[i]):
            runs[-1].append(i)
        else:
            runs.append([i])
    columns = [_WB, _TINT, _EXPOSURE]
    for run in runs:
        if len(run) < 2:
            continue
        mean = pulled[np.ix_(run, columns)].mean(axis=0)
        pulled[np.ix_(run, columns)] += sequence_lam * (mean - pulled[np.ix_(run, columns)])

    relative = _to_relative(items, pulled)
    return {item.photo_id: sv.clip(relative[i]) for i, item in enumerate(items)}
