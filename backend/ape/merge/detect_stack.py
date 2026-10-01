# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Finding focus stacks among a project's shots (section 25.2).

A focus stack looks exactly like a burst of a still subject: same framing, same
exposure, frames seconds apart. What differs is *where* each frame is sharp,
and that is the evidence:

* **EXIF.** Sony records the focus actuator position (``FocusPosition2``). In a
  stack it moves the same way at every frame; in a burst it stays put. The
  value 255 is written when the camera did not report a position (seen on most
  of the fixtures, with AF-C), and counts as unknown.
* **Pixels.** The map of local sharpness of the preview (``features.py``),
  normalised frame by frame so that a globally softer frame -- camera shake --
  does not count: in a stack, different parts of the picture peak in different
  frames, by a margin no burst shows.

Either one is enough; both raise the confidence. Nothing is merged here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..culling.burst import hamming
from .detect import DetectedMerge

__all__ = ["StackShot", "detect_focus_stacks"]

#: Largest gap between two frames. Longer than a bracketing's three seconds:
#: a stack is often focused by hand, a turn of the ring per frame.
_MAX_GAP_S = 10.0

#: Fewest frames. Two frames with different focus are a photographer who
#: refocused far more often than a stack.
_MIN_FRAMES = 3
_MAX_FRAMES = 30

#: Hash distance to the neighbouring frame, on the plain (not equalised)
#: preview. Defocus changes fine detail, which a 32x32 DCT hash barely sees;
#: a change of framing changes it a lot. Bursts of a still subject measure
#: 8 bits or fewer on the fixtures.
_SAME_FRAMING_BITS = 14

#: Exposure must not change: a focus stack is shot at fixed settings.
_SAME_EXPOSURE_EV = 0.1

#: A tile is *focus-selective* when its sharpest frame beats the median frame
#: by this much (normalised log-sharpness). Against the median, not the
#: softest frame: one shaken frame in a burst lowers the minimum and leaves the
#: median where it was. Measured on synthetic sequences built from the 24
#: fixture previews (``tests/test_merge_detect_stack.py``): bursts with noise,
#: recompression, 2 px of jitter and one frame shaken by a 1.2 px blur have up
#: to 38% of their tiles selective -- all peaking in the *same* frame -- while
#: stacks of 3 and 5 frames with the focus plane swept across the picture have
#: 21-100%, peaking in different frames.
_SELECTIVE_MARGIN = 0.3

#: Share of the tiles that must be selective, and share of those that must
#: peak in a frame other than the most common one: the focus plane moves,
#: rather than one frame being different everywhere. Bursts: 0.00 on every
#: synthetic sequence; stacks: 0.33-0.79, save one three-frame stack at 0.20,
#: which stays undetected -- the safe error, since the user can still make the
#: group by hand.
_SELECTIVE_SHARE = 0.2
_MOVED_SHARE = 0.25

#: Sony writes this when it has no position to report.
_UNKNOWN_POSITION = 255


@dataclass(frozen=True)
class StackShot:
    id: int
    shot_at: datetime | None
    order: tuple
    focal_length: float | None
    aperture: float | None
    iso: int | None
    shutter: float | None
    focus_position: int | None
    signature: dict | None
    sharp_map: Sequence[float] | None


def _brightness(shot: StackShot) -> float | None:
    if not (shot.shutter and shot.aperture):
        return None
    value = math.log2(shot.shutter) - 2.0 * math.log2(shot.aperture)
    return value + (math.log2(shot.iso / 100.0) if shot.iso else 0.0)


def _compatible(a: StackShot, b: StackShot) -> bool:
    if a.shot_at is None or b.shot_at is None or a.signature is None or b.signature is None:
        return False
    if not 0.0 <= (b.shot_at - a.shot_at).total_seconds() <= _MAX_GAP_S:
        return False
    if not (a.focal_length and b.focal_length) or abs(b.focal_length / a.focal_length - 1) > 0.01:
        return False
    first, second = _brightness(a), _brightness(b)
    if first is None or second is None or abs(first - second) > _SAME_EXPOSURE_EV:
        return False
    return hamming(a.signature["phash"], b.signature["phash"]) <= _SAME_FRAMING_BITS


def _positions_move(shots: Sequence[StackShot]) -> bool:
    positions = [s.focus_position for s in shots]
    if any(p is None or p == _UNKNOWN_POSITION for p in positions):
        return False
    steps = [b - a for a, b in zip(positions, positions[1:], strict=False)]
    return len(set(positions)) >= _MIN_FRAMES and (
        all(s >= 0 for s in steps) or all(s <= 0 for s in steps)
    )


def _focus_moves(shots: Sequence[StackShot]) -> tuple[bool, float]:
    """Whether the sharp region moves across the frames; and the selective share."""
    maps = [s.sharp_map for s in shots]
    if any(m is None for m in maps) or len({len(m) for m in maps}) != 1:
        return False, 0.0
    values = np.asarray(maps, dtype=np.float64)
    values -= values.mean(axis=1, keepdims=True)  # a globally softer frame is shake, not focus
    margin = values.max(axis=0) - np.median(values, axis=0)
    selective = margin >= _SELECTIVE_MARGIN
    share = float(selective.mean())
    if share < _SELECTIVE_SHARE:
        return False, share
    peaks = values[:, selective].argmax(axis=0)
    moved = 1.0 - np.bincount(peaks).max() / len(peaks)
    return moved >= _MOVED_SHARE, share


def _candidate(shots: Sequence[StackShot]) -> DetectedMerge | None:
    by_exif = _positions_move(shots)
    by_pixels, share = _focus_moves(shots)
    if not (by_exif or by_pixels):
        return None
    confidence = 0.55 + (0.25 if by_exif else 0.0) + (0.2 if by_pixels else 0.0)
    reference = shots[len(shots) // 2]
    first, last = shots[0].shot_at, shots[-1].shot_at
    span = (last - first).total_seconds() if first and last else 0.0
    return DetectedMerge(
        kind="focus_stack",
        members=tuple(s.id for s in shots),
        reference=reference.id,
        ev_offsets=tuple(0.0 for _ in shots),
        confidence=round(min(confidence, 1.0), 2),
        reasons={
            "frames": len(shots),
            "span_s": round(span, 1),
            "focus_positions": by_exif,
            "focus_moves": by_pixels,
            "selective_share": round(share, 2),
        },
    )


def detect_focus_stacks(shots: Iterable[StackShot]) -> list[DetectedMerge]:
    """Focus stacks among a project's shots, longest sequences first."""
    ordered = sorted(
        shots, key=lambda s: (s.shot_at is None, s.shot_at or datetime.min, s.order)
    )
    found: list[DetectedMerge] = []
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and _compatible(ordered[end - 1], ordered[end]):
            end += 1
        run = ordered[index:end]
        position = 0
        while len(run) - position >= _MIN_FRAMES:
            for length in range(min(_MAX_FRAMES, len(run) - position), _MIN_FRAMES - 1, -1):
                detected = _candidate(run[position : position + length])
                if detected is not None:
                    found.append(detected)
                    position += length
                    break
            else:
                position += 1
        index = end
    return found
