# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Finding panoramas among a project's shots (section 25.2).

The evidence is geometric: consecutive frames with the same settings that
*partly* overlap -- 15 to 60 per cent -- shifted mostly along one axis, and
always the same way along the sequence. Less overlap is not a panorama the
stitcher could close; more is the same picture twice.

...or a **sweep**: a panorama shot as a burst while turning, which is how the
user's own panoramas in ``tests/fixtures/fase11`` were taken -- 17 and 19
frames, each 5-13 per cent further along than the last, 85-95 per cent
overlap between neighbours. Pair by pair that is "the same picture twice"; the
sequence is not. So a pair with more overlap still counts when its shift is
clearly one way (at least :data:`MIN_SWEEP_SHIFT`, where a hand-held burst
jitters by a few tenths of a per cent), and what makes a run a panorama is how
far it travels in all: first and last frame may overlap at most 60 per cent,
section 25.2's upper bound, applied to the whole. Which of a sweep's frames the
stitcher uses is the merge's business (``panorama.select_frames``); all of
them are members, so culling leaves them all alone and accepting the merge
sets them all aside.

The overlap is measured with ORB on thumbnails of 256 px, as section 25.2
asks: a few milliseconds per pair, and only for pairs the EXIF has not ruled
out already. That is still too much arithmetic for the server's request
thread on two thousand photos, so this runs in a worker (the
``detect_merges`` job); everything else about detection is in the server.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from .detect import DetectedMerge

__all__ = ["MIN_SWEEP_SHIFT", "PanoShot", "PairOverlap", "detect_panoramas", "measure_overlap"]

#: Section 25.2: thumbnails of 256 px.
THUMB_EDGE = 256

#: Section 25.2: "la sovrapposizione utile è 15–60%".
_MIN_OVERLAP, _MAX_OVERLAP = 0.15, 0.60

#: Smallest shift of a sweep's pair, as a fraction of the frame. The user's
#: sweeps move 4.8-13 per cent a frame; the hand-held bracketing next to them
#: 0.4 at most.
MIN_SWEEP_SHIFT = 0.02

#: Gap between two frames of a hand-held panorama: a turn, a breath.
_MAX_GAP_S = 20.0

#: Auto exposure drifts across a panorama, and between two neighbours of a
#: sweep that turns from sun to shade it jumps: a full stop between 07351 and
#: 07352 of the user's fixtures (1/1600, 1/800 s). More than this is a
#: different scene. Brackets never get here: they are found first, and their
#: frames are excluded.
_SAME_EXPOSURE_EV = 1.5

#: "Prevalentemente su un solo asse": the main component of the shift must be
#: at least this many times the other.
_AXIS_RATIO = 2.0

#: Fewest RANSAC inliers between two thumbnails for the overlap to be believed.
_MIN_INLIERS = 15


@dataclass(frozen=True)
class PanoShot:
    id: int
    shot_at: datetime | None
    order: tuple
    camera: str | None
    focal_length: float | None
    #: ``exposure_brightness`` of the shot, in stops.
    brightness: float | None


@dataclass(frozen=True)
class PairOverlap:
    #: Fraction of the second frame covered by the first.
    overlap: float
    #: Shift of the second frame's content, as fractions of its size.
    dx: float
    dy: float
    inliers: int

    @property
    def axis(self) -> str:
        return "x" if abs(self.dx) >= abs(self.dy) else "y"

    @property
    def direction(self) -> int:
        value = self.dx if self.axis == "x" else self.dy
        return 1 if value > 0 else -1

    @property
    def shift(self) -> float:
        """The shift along the main axis, as a fraction of the frame."""
        return max(abs(self.dx), abs(self.dy))

    @property
    def one_axis(self) -> bool:
        major, minor = sorted((abs(self.dx), abs(self.dy)), reverse=True)
        return major >= _AXIS_RATIO * minor


def _thumb(image: np.ndarray) -> np.ndarray:
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape
    scale = THUMB_EDGE / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(small)


def measure_overlap(first: np.ndarray, second: np.ndarray) -> PairOverlap | None:
    """How much of ``second`` the content of ``first`` covers, and in which direction.

    Both are 8-bit images (grey or BGR) of any size; they are reduced to 256 px.
    A similarity is fitted rather than a homography: at 256 px there are too
    few points for eight degrees of freedom to be stable, and the overlap is a
    rough number anyway.
    """
    a, b = _thumb(first), _thumb(second)
    orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=6, edgeThreshold=15,
                         patchSize=15, fastThreshold=8)
    keys_a, desc_a = orb.detectAndCompute(a, None)
    keys_b, desc_b = orb.detectAndCompute(b, None)
    if desc_a is None or desc_b is None or len(keys_a) < _MIN_INLIERS or len(keys_b) < _MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < _MIN_INLIERS:
        return None
    src = np.float32([keys_a[m.queryIdx].pt for m in good])
    dst = np.float32([keys_b[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000
    )
    if matrix is None or mask is None or int(mask.sum()) < _MIN_INLIERS:
        return None
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    if not 0.8 <= scale <= 1.25:
        return None  # not the same focal length, whatever the EXIF says

    height, width = a.shape
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    mapped = cv2.transform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    frame = np.float32([[0, 0], [b.shape[1], 0], [b.shape[1], b.shape[0]], [0, b.shape[0]]])
    area, _ = cv2.intersectConvexConvex(mapped, frame)
    centre = cv2.transform(np.float32([[[width / 2, height / 2]]]), matrix).reshape(2)
    return PairOverlap(
        overlap=round(float(area) / float(b.shape[0] * b.shape[1]), 3),
        dx=round(float(centre[0] - b.shape[1] / 2) / b.shape[1], 3),
        dy=round(float(centre[1] - b.shape[0] / 2) / b.shape[0], 3),
        inliers=int(mask.sum()),
    )


def _compatible(a: PanoShot, b: PanoShot) -> bool:
    if a.shot_at is None or b.shot_at is None:
        return False
    if not 0.0 <= (b.shot_at - a.shot_at).total_seconds() <= _MAX_GAP_S:
        return False
    if a.camera != b.camera:
        return False
    if not (a.focal_length and b.focal_length) or abs(b.focal_length / a.focal_length - 1) > 0.01:
        return False
    if a.brightness is None or b.brightness is None:
        return False
    return abs(a.brightness - b.brightness) <= _SAME_EXPOSURE_EV


def _usable(pair: PairOverlap | None) -> bool:
    if pair is None or pair.overlap < _MIN_OVERLAP or not pair.one_axis:
        return False
    return pair.overlap <= _MAX_OVERLAP or pair.shift >= MIN_SWEEP_SHIFT


def _travel(pairs: Sequence[PairOverlap]) -> float:
    """How far a run goes along its axis, in frames."""
    return sum(p.shift for p in pairs)


def _proposal(shots: Sequence[PanoShot], pairs: Sequence[PairOverlap]) -> DetectedMerge:
    overlaps = [p.overlap for p in pairs]
    confidence = 0.55 + min(0.25, 0.05 * len(pairs))
    if all(p.inliers >= 3 * _MIN_INLIERS for p in pairs):
        confidence += 0.15
    first, last = shots[0].shot_at, shots[-1].shot_at
    span = (last - first).total_seconds() if first and last else 0.0
    return DetectedMerge(
        kind="panorama",
        members=tuple(s.id for s in shots),
        reference=shots[len(shots) // 2].id,
        ev_offsets=tuple(
            round((s.brightness or 0.0) - (shots[len(shots) // 2].brightness or 0.0), 2)
            for s in shots
        ),
        confidence=round(min(confidence, 1.0), 2),
        reasons={
            "frames": len(shots),
            "span_s": round(span, 1),
            "overlap_min": round(min(overlaps), 2),
            "overlap_max": round(max(overlaps), 2),
            "axis": "orizzontale" if pairs[0].axis == "x" else "verticale",
            "sweep": any(p.overlap > _MAX_OVERLAP for p in pairs),
        },
    )


def detect_panoramas(
    shots: Iterable[PanoShot],
    image: Callable[[int], np.ndarray | None],
    *,
    excluded: set[int] | frozenset[int] = frozenset(),
) -> list[DetectedMerge]:
    """Panoramas among a project's shots.

    Args:
        shots: every candidate, in any order.
        image: returns a thumbnail (any size, 8-bit) for a photo id, or None.
            Called at most once per photo.
        excluded: photos already in another merge group.
    """
    ordered = sorted(
        (s for s in shots if s.id not in excluded),
        key=lambda s: (s.shot_at is None, s.shot_at or datetime.min, s.order),
    )
    cache: dict[int, np.ndarray | None] = {}

    def thumb(photo_id: int) -> np.ndarray | None:
        if photo_id not in cache:
            cache[photo_id] = image(photo_id)
        return cache[photo_id]

    found: list[DetectedMerge] = []
    run: list[PanoShot] = []
    pairs: list[PairOverlap] = []

    def close() -> None:
        if len(run) >= 2 and _travel(pairs) >= 1.0 - _MAX_OVERLAP:
            found.append(_proposal(run, pairs))

    for shot in ordered:
        previous = run[-1] if run else None
        pair = None
        if previous is not None and _compatible(previous, shot):
            first, second = thumb(previous.id), thumb(shot.id)
            if first is not None and second is not None:
                pair = measure_overlap(first, second)
        consistent = _usable(pair) and (
            not pairs or (pair.axis == pairs[0].axis and pair.direction == pairs[0].direction)
        )
        if consistent:
            run.append(shot)
            pairs.append(pair)  # type: ignore[arg-type]
            continue
        close()
        run, pairs = [shot], []
        # Free what the next pairs will not need.
        for key in [k for k in cache if k != shot.id]:
            del cache[key]
    close()
    return found
