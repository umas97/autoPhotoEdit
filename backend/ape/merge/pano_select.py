# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which frames of a panorama the stitcher needs (section 25.5).

A panorama shot as a burst while turning -- a *sweep*, like the user's own in
``tests/fixtures/fase11`` -- has 17 or 19 frames, each 5-13 per cent further
along than the last. Stitching them all would be past the ceiling of twelve
frames (25.5.4), would cost a full decode per frame, and would lay down a seam
every few hundred pixels, each one a chance for parallax to show. Four of
them cover the same ground.

So the merge keeps, starting from the reference and going each way, the
farthest frame that still overlaps the last one kept by :data:`KEEP_OVERLAP`,
and sets the rest aside. The frames stay members of the group -- they are
still the panorama's, culling still leaves them alone, accepting still sets
them all aside -- they just add no pixels. A panorama shot the ordinary way,
15-60 per cent overlap between neighbours, keeps every frame: the next but one
never overlaps by half.

The overlaps are measured like detection measures them (``detect_pano``), on
the previews the camera embedded in each RAW: a few milliseconds a pair, the
same result for the preview and for the full merge, and the same result every
time, which the recipe's digest needs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .detect_pano import MIN_SWEEP_SHIFT, measure_overlap

__all__ = ["KEEP_OVERLAP", "select_frames", "selected_members"]

#: Least overlap between two frames the stitcher gets side by side. Half a
#: frame leaves the seam finder room to go around what moved or what parallax
#: displaced; the usual advice for hand-held panoramas is a third or more.
KEEP_OVERLAP = 0.5


def select_frames(
    count: int, reference: int, image: Callable[[int], np.ndarray | None]
) -> list[int]:
    """Indices of the frames to stitch, in order, always with ``reference``.

    Args:
        count: frames, in shooting order.
        reference: index of the reference frame.
        image: an 8-bit preview of a frame by index, or None when there is none.
    """
    cache: dict[int, np.ndarray | None] = {}

    def preview(index: int) -> np.ndarray | None:
        if index not in cache:
            cache[index] = image(index)
        return cache[index]

    def close_enough(a: int, b: int) -> bool:
        """Whether ``b`` may be stitched right next to ``a``, skipping what is
        between. Frames that do not move are not a sweep: a panorama cut out
        of copies of one file (the benchmark's) keeps every frame."""
        first, second = preview(a), preview(b)
        if first is None or second is None:
            return False
        pair = measure_overlap(first, second)
        return pair is not None and pair.overlap >= KEEP_OVERLAP and pair.shift >= MIN_SWEEP_SHIFT

    kept = {reference}
    for direction in (1, -1):
        current = reference
        while 0 <= current + direction < count:
            best = current + direction
            candidate = best + direction
            while 0 <= candidate < count and close_enough(current, candidate):
                best = candidate
                candidate += direction
            kept.add(best)
            current = best
    return sorted(kept)


def _embedded(member: Any) -> np.ndarray | None:
    from ..raw.embedded import read_embedded_preview

    if not getattr(member, "path", None):
        return None
    try:
        return read_embedded_preview(member.path).image
    except (OSError, ValueError):
        return None


def selected_members(members: Sequence[Any]) -> tuple[Any, ...]:
    """The members of a panorama recipe that the stitcher uses."""
    reference = next((i for i, m in enumerate(members) if m.reference), len(members) // 2)
    chosen = select_frames(len(members), reference, lambda i: _embedded(members[i]))
    return tuple(members[i] for i in chosen)
