# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How much the user's own edits burn and crush (the clipping terms of section 9.1).

A predicted edit is judged on what it *adds* to the frame's own clipping
(``tonal.added_clipping``); the style is measured the same way, on each
training sample: the user's parameters against the sample's own neutral frame,
whose tonal tails the sample keeps in its ``context`` (``tails``, measured when
the pair is prepared or copied from the photo a correction came from). A
predicted edit of a dark stage that crushes as much beyond the scene as the
user's own edits of the nearest samples do is on style, not in error.

The user's delivered JPEGs are no measure of it: they are cropped, and the
crop takes away exactly the dark corners and edges the uncropped prediction
still has -- DSC06332 of the user's event, 0% black in their JPEG, 16% in the
full frame at an exposure close to theirs.

Samples prepared before the tails were kept have none, and simply do not vote.
Computed once per version of the profile and cached, since the samples only
change when the profile does.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import StyleProfile, StyleSample, StyleSampleStatus
from ..pipeline.params import EditParams
from .tonal import added_clipping

__all__ = ["sample_clipping", "style_clipping"]

_cache: dict[tuple[int, str], dict[int, tuple[float, float]]] = {}


def sample_clipping(session: Session, profile: StyleProfile) -> dict[int, tuple[float, float]]:
    """``sample_id -> (burnt, crushed)`` the user's edit adds to each sample's frame."""
    key = (profile.id, str(profile.updated_at))
    cached = _cache.get(key)
    if cached is not None:
        return cached
    rows = session.execute(
        select(StyleSample.id, StyleSample.params, StyleSample.context).where(
            StyleSample.profile_id == profile.id,
            StyleSample.status == StyleSampleStatus.READY,
        )
    ).all()
    table = {}
    for sample_id, params, context in rows:
        context = context or {}
        found = added_clipping(
            context.get("tails"),
            EditParams.from_dict(params or {}),
            context.get("exposure_anchor_ev"),
        )
        if found is not None:
            table[int(sample_id)] = found
    for stale in [k for k in _cache if k[0] == profile.id]:
        del _cache[stale]
    _cache[key] = table
    return table


def style_clipping(
    neighbours: Iterable, table: dict[int, tuple[float, float]]
) -> tuple[float, float] | None:
    """What the user's edits of these neighbours add in burnt and crushed, weighted.

    ``neighbours`` is the prediction's ``[[sample_id, weight], ...]``.
    """
    known = [(table[int(i)], float(w)) for i, w in neighbours if int(i) in table]
    if not known:
        return None
    values = np.array([v for v, _w in known])
    weights = np.array([w for _v, w in known])
    total = float(weights.sum())
    if total <= 0:
        return None
    mean = weights @ values / total
    return float(mean[0]), float(mean[1])
