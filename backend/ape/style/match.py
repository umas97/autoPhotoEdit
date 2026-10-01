# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which profile suits a project (section 8.4).

The project's mean CLIP embedding against each learned profile's centroid,
as a cosine. The interface shows the number and preselects the best profile;
the user can always pick another (selection *with* override).

Built-in profiles have no samples and so no centroid: they are listed without
an affinity, and the proposal falls back to *Neutro automatico* when no
learned profile is close enough to be worth proposing -- a profile learned on
weddings is not a better guess for a mountain walk than the neutral one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MIN_AFFINITY", "Affinity", "affinities", "project_centroid"]

#: Below this cosine a learned profile is not proposed. Measured on centroids
#: (200 random draws each): a 30-photo profile from the user's event against
#: the rest of the event, 0.966 at worst; two halves of the 24-photo mountain
#: walk, 0.925 at worst; the event profile against a walk project, 0.852 at
#: best. 0.90 sits in the gap.
MIN_AFFINITY = 0.90


@dataclass(frozen=True, slots=True)
class Affinity:
    profile_id: int
    score: float | None


def project_centroid(embeddings: list[np.ndarray]) -> np.ndarray | None:
    if not embeddings:
        return None
    centroid = np.mean(np.asarray(embeddings, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(centroid))
    return centroid / norm if norm > 0 else None


def affinities(
    centroid: np.ndarray | None, profiles: list[tuple[int, np.ndarray | None]]
) -> list[Affinity]:
    """Cosine of the project centroid with every profile that has one, best first."""
    out = []
    for profile_id, profile_centroid in profiles:
        score = None
        if centroid is not None and profile_centroid is not None:
            score = float(np.clip(centroid @ profile_centroid, -1.0, 1.0))
        out.append(Affinity(profile_id, score))
    return sorted(out, key=lambda a: (a.score is None, -(a.score or 0.0)))
