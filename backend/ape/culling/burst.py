# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bursts and near-duplicates (section 7.3, criterion 4).

Ten frames of the same moment are one decision, not ten. So consecutive shots
that are close in time *and* look alike are grouped, and the selection proposes
one of them -- keeping the others a click away.

Both conditions, never one. Time alone would merge two different subjects shot
a second apart; looks alone would merge the same view photographed at nine in
the morning and again at five, which are two moments the user meant to keep.

The visual comparison is deliberately coarse. A burst is a subject in motion:
between the first frame and the seventh a person has taken two steps, and a
comparison fine enough to notice that would split the burst in two. What is
compared is the layout of light and dark at 8x8 (a perceptual hash) and the
distribution of colour -- which a moving subject barely changes, and which a
change of scene changes completely.

Frames that belong to a candidate merge -- a bracketing, a focus stack, a
panorama (section 25.2) -- are handed in as excluded and never grouped here:
they look like a burst and are not one, and discarding six of a seven-frame
bracketing as "duplicates" is the fastest way to make this program useless to
anyone who shoots them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

__all__ = [
    "BURST_MAX_GAP_S",
    "Shot",
    "group_bursts",
    "hamming",
    "histogram_distance",
    "similar",
    "visual_signature",
]

#: Section 7.3: shots less than two seconds apart. Sony bodies record
#: ``DateTimeOriginal`` to the second, so "less than two" means one second or
#: less on the clock -- a true gap anywhere under two seconds.
BURST_MAX_GAP_S = 2.0

#: Out of 63 bits. Measured on synthetic bursts built from the 24 A7 III
#: previews of ``tests/fixtures`` (the frame shifted by up to 4 %, exposure
#: varied by up to 0.3 EV, fresh noise on every frame): consecutive frames
#: differ by at most 8 bits. Among the 276 pairs of distinct photographs of
#: the same outing, only two come under 18 -- and those two (14 and 16 bits)
#: are genuine repeats of one view, 22 and 48 seconds apart, which the time
#: condition keeps apart.
_HASH_BITS = 16

#: Bhattacharyya distance between colour histograms, 0 identical, 1 disjoint.
#: The same synthetic bursts reach 0.33 between consecutive frames, the
#: exposure wobble being what moves the histogram most. The hash does the
#: discriminating; this is a guard against a frame whose layout happens to
#: match and whose colours do not.
_HISTOGRAM_DISTANCE = 0.40

#: A change of focal length is a reframing, not a repeat. Zooming in the middle
#: of a burst does happen, but two frames at 42 and 59 mm of the same path are
#: two compositions the photographer chose, not a duplicate to thin out.
_FOCAL_TOLERANCE = 0.05

#: Bins per channel of the colour histogram. Four is coarse on purpose: a
#: cloud crossing the sun must not split a burst.
_BINS = 4


def _phash(gray32: np.ndarray) -> int:
    """64-bit perceptual hash of a 32x32 float image (DCT, top-left 8x8)."""
    coefficients = cv2.dct(gray32)[:8, :8].ravel()
    # The DC term is the mean brightness: leaving it out is what makes the hash
    # indifferent to a small exposure change between frames.
    ac = coefficients[1:]
    bits = ac > np.median(ac)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def visual_signature(bgr: np.ndarray) -> dict:
    """What burst grouping and merge detection compare, for one preview.

    Returns a JSON-able dict: ``phash`` (hex), ``phash_eq`` (hex, of the
    histogram-equalised frame, which is what survives the exposure steps of a
    bracketing) and ``hist`` (64 floats summing to 1).
    """
    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    equalised = cv2.resize(cv2.equalizeHist(gray), (32, 32), interpolation=cv2.INTER_AREA)

    histogram = cv2.calcHist(
        [small], [0, 1, 2], None, [_BINS] * 3, [0, 256, 0, 256, 0, 256]
    ).ravel()
    histogram /= max(float(histogram.sum()), 1.0)
    return {
        "phash": f"{_phash(gray32):016x}",
        "phash_eq": f"{_phash(equalised.astype(np.float32)):016x}",
        "hist": [round(float(v), 4) for v in histogram],
    }


def hamming(a: str, b: str) -> int:
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))


def histogram_distance(a: Sequence[float], b: Sequence[float]) -> float:
    first = np.asarray(a, dtype=np.float32)
    second = np.asarray(b, dtype=np.float32)
    return float(cv2.compareHist(first, second, cv2.HISTCMP_BHATTACHARYYA))


def similar(a: dict, b: dict) -> bool:
    """Whether two signatures look like frames of the same moment."""
    return (
        hamming(a["phash"], b["phash"]) <= _HASH_BITS
        and histogram_distance(a["hist"], b["hist"]) <= _HISTOGRAM_DISTANCE
    )


@dataclass(frozen=True)
class Shot:
    """One photo as grouping sees it."""

    id: int
    shot_at: datetime | None
    #: Tie-break within a second: the camera's shot counter when it has one,
    #: the filename otherwise. Both increase with every frame.
    order: tuple
    signature: dict | None
    focal_length: float | None = None
    camera: str | None = None


def _sort_key(shot: Shot) -> tuple:
    return (shot.shot_at is None, shot.shot_at or datetime.min, shot.order)


def _chains(previous: Shot, current: Shot) -> bool:
    if previous.shot_at is None or current.shot_at is None:
        return False
    if previous.signature is None or current.signature is None:
        return False
    if previous.camera and current.camera and previous.camera != current.camera:
        return False
    gap = (current.shot_at - previous.shot_at).total_seconds()
    if not 0.0 <= gap < BURST_MAX_GAP_S:
        return False
    if previous.focal_length and current.focal_length:
        ratio = current.focal_length / previous.focal_length
        if abs(ratio - 1.0) > _FOCAL_TOLERANCE:
            return False
    return similar(previous.signature, current.signature)


def group_bursts(shots: Iterable[Shot], *, excluded: Iterable[int] = ()) -> list[list[int]]:
    """Group consecutive look-alike shots. Returns groups of two or more ids.

    The chain is built frame to frame, each compared with the one before it,
    so a burst that pans slowly stays one burst even when its first and last
    frames have little in common. An excluded shot breaks the chain: nothing on
    either side of a bracketing is merged across it.
    """
    skip = set(excluded)
    ordered = sorted(shots, key=_sort_key)

    groups: list[list[int]] = []
    current: list[int] = []
    previous: Shot | None = None
    for shot in ordered:
        if shot.id in skip:
            if len(current) > 1:
                groups.append(current)
            current, previous = [], None
            continue
        if previous is not None and _chains(previous, shot):
            current.append(shot.id)
        else:
            if len(current) > 1:
                groups.append(current)
            current = [shot.id]
        previous = shot
    if len(current) > 1:
        groups.append(current)
    return groups
