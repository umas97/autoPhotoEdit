# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Finding multi-shot groups before culling looks at them (section 25.2).

This is the part of phase 11 that culling cannot do without. Section 7.3 is
explicit: a bracketing looks like a burst and is not one, and its dark and
bright frames are underexposed and overexposed *on purpose*. If culling ran
without knowing which frames are a bracketing, it would keep the middle frame
of every sequence and throw away the two that make the HDR possible.

So this module runs first, on the same thumbnails and metadata, and returns
*proposals*. Nothing is merged here and nothing is ever merged without the user
saying so; culling only uses the proposals to leave those frames alone.

Only exposure bracketing is detected for now. Focus stacks and panoramas need
their own evidence -- a focus distance that moves, a partial overlap measured
with feature matching -- and arrive with the merge engine itself (phase 11).

The EXIF evidence, from section 25.2: consecutive frames at most three seconds
apart, same focal length and aperture, and exposures that step evenly. The
order is not required to be monotonic, because Sony's own default is not: the
camera shoots the metered frame first, then the under, then the over. The
visual evidence is that the frames are the same picture once brightness is
taken out of the comparison -- the hash of the histogram-equalised thumbnail.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from ..culling.burst import hamming

__all__ = ["BracketShot", "DetectedMerge", "detect_brackets", "exposure_brightness"]

#: Section 25.2: frames of one bracketing are at most three seconds apart.
_MAX_GAP_S = 3.0

#: Sequences a camera shoots. Two-frame brackets exist but are rare, and a pair
#: of frames one stop apart is far more often a photographer correcting an
#: exposure than an HDR; section 25.2 names three, five and seven.
_LENGTHS = (9, 7, 5, 3)

#: Smallest step that counts as bracketing. Sony's finest is a third of a stop,
#: but at a third the frames are an exposure correction, not a dynamic range
#: extension; half a stop is the least any HDR tutorial uses.
_MIN_STEP_EV = 0.5

#: ...unless the camera itself says it was bracketing. A third of a stop is
#: the A7 III's default bracket step, and the user's own five-frame brackets
#: in ``tests/fixtures/fase11`` are shot at it: frames the photographer asked
#: for, not a correction. Less the rounding of the EXIF values.
_MIN_CAMERA_STEP_EV = 0.25

#: How far each exposure may sit from an even progression, in stops, fitted to
#: all of them. EXIF values are rounded -- Sony writes 1/15 for 2^-4 s and 1/13
#: for 2^-11/3 s, up to 0.09 stops off -- and a fit spreads one value's error
#: over the others: the user's shutter bracket 1/25-1/10 s sits within 0.07.
#: Measured on positions, not on the steps between them, where two rounding
#: errors in opposite directions make one step 0.2 stops off. Tighter than a
#: rounding needs, because three frames out of the middle of a bracket can come
#: close: 07310-07312, a third of a stop apart around the metered frame they
#: leave out, fit a step of half a stop to within 0.119.
_POSITION_TOLERANCE_EV = 0.1

#: Hash distance between histogram-equalised thumbnails of one bracketing.
#: Measured on the A7 III previews of ``tests/fixtures`` stepped to -2, 0 and
#: +2 EV: at most 14 bits from the metered frame. Distinct photographs of the
#: same outing sit at 24 or more for all but one pair in twenty.
_ALIGNED_BITS = 16

#: ``ReleaseMode2`` codes for exposure bracketing -- continuous and single
#: frame -- in ExifTool's table of Sony tags. Evidence that raises the
#: confidence, never a requirement: the A7 II writes a different makernote.
#: Only for shots analysed before ``camera_bracket`` was stored, which reads
#: the camera's tags with the right table for each (``culling/features.py``).
_SONY_BRACKET_MODES = frozenset({2, 23})


@dataclass(frozen=True)
class BracketShot:
    id: int
    shot_at: datetime | None
    order: tuple
    focal_length: float | None
    aperture: float | None
    iso: int | None
    shutter: float | None
    exposure_bias: float | None
    release_mode: int | None
    signature: dict | None
    #: The camera says it was bracketing the exposure (``camera_details``);
    #: ``None`` when the shot was analysed before this was stored.
    camera_bracket: bool | None = None


@dataclass(frozen=True)
class DetectedMerge:
    kind: str
    #: Members in shooting order.
    members: tuple[int, ...]
    #: The frame the others align to and inherit metadata from: the one
    #: nearest the metered exposure.
    reference: int
    #: Stops relative to the reference, per member, in the order of ``members``.
    ev_offsets: tuple[float, ...]
    confidence: float
    #: Structured, not a sentence: the interface formats it in its own language.
    reasons: dict = field(default_factory=dict)


def exposure_brightness(shot: BracketShot) -> float | None:
    """How much light the capture recorded, in stops, up to a constant.

    ``log2(t) + log2(ISO / 100) - 2 log2(N)``: longer, more sensitive, wider
    all mean brighter. The exposure compensation is used only when the shutter
    speed is missing -- when it is present it already contains the bracket.
    """
    if shot.shutter and shot.aperture:
        value = math.log2(shot.shutter) - 2.0 * math.log2(shot.aperture)
        if shot.iso:
            value += math.log2(shot.iso / 100.0)
        return value
    return shot.exposure_bias


def _compatible(a: BracketShot, b: BracketShot) -> bool:
    if a.shot_at is None or b.shot_at is None:
        return False
    gap = (b.shot_at - a.shot_at).total_seconds()
    if not 0.0 <= gap <= _MAX_GAP_S:
        return False
    if a.focal_length and b.focal_length and abs(b.focal_length / a.focal_length - 1) > 0.01:
        return False
    return not (
        a.aperture and b.aperture and abs(math.log2(b.aperture / a.aperture)) > 0.05
    )


def _evenly_stepped(values: Sequence[float], min_step: float) -> float | None:
    """The step of an arithmetic progression, or ``None`` if these are not one."""
    ordered = sorted(values)
    count = len(ordered)
    centre = (count - 1) / 2
    mean = sum(ordered) / count
    step = sum((i - centre) * (v - mean) for i, v in enumerate(ordered)) / sum(
        (i - centre) ** 2 for i in range(count)
    )
    if step < min_step:
        return None
    tolerance = max(_POSITION_TOLERANCE_EV, 0.1 * step)
    if any(abs(v - (mean + (i - centre) * step)) > tolerance for i, v in enumerate(ordered)):
        return None
    return step


def _camera_bracket(shot: BracketShot) -> bool:
    if shot.camera_bracket is not None:
        return shot.camera_bracket
    return shot.release_mode in _SONY_BRACKET_MODES


def _aligned(shots: Sequence[BracketShot], reference: BracketShot) -> int | None:
    """Largest equalised-hash distance to the reference, or None if unknown."""
    if reference.signature is None or any(s.signature is None for s in shots):
        return None
    return max(
        hamming(s.signature["phash_eq"], reference.signature["phash_eq"]) for s in shots
    )


def _candidate(shots: Sequence[BracketShot]) -> DetectedMerge | None:
    brightness = [exposure_brightness(s) for s in shots]
    if any(b is None for b in brightness):
        return None
    values = [float(b) for b in brightness]  # type: ignore[arg-type]
    sony = all(_camera_bracket(s) for s in shots)
    step = _evenly_stepped(values, _MIN_CAMERA_STEP_EV if sony else _MIN_STEP_EV)
    if step is None:
        return None

    median = sorted(values)[len(values) // 2]
    reference_index = min(range(len(shots)), key=lambda i: abs(values[i] - median))
    reference = shots[reference_index]
    distance = _aligned(shots, reference)
    if distance is None or distance > _ALIGNED_BITS:
        return None

    confidence = 0.6
    if sony:
        confidence += 0.2
    if distance <= _ALIGNED_BITS // 2:
        confidence += 0.2

    offsets = tuple(round(v - values[reference_index], 2) for v in values)
    first, last = shots[0].shot_at, shots[-1].shot_at
    span = (last - first).total_seconds() if first and last else 0.0
    return DetectedMerge(
        kind="hdr",
        members=tuple(s.id for s in shots),
        reference=reference.id,
        ev_offsets=offsets,
        confidence=round(min(confidence, 1.0), 2),
        reasons={
            "frames": len(shots),
            "ev_min": min(offsets),
            "ev_max": max(offsets),
            "step_ev": round(step, 2),
            "span_s": round(span, 1),
            "camera_bracket": sony,
        },
    )


def detect_brackets(shots: Iterable[BracketShot]) -> list[DetectedMerge]:
    """Exposure bracketings among a project's shots.

    Runs of compatible consecutive frames are cut into the longest valid
    sequences first: a seven-frame bracket must not come out as a three and a
    stray four.
    """
    ordered = sorted(
        shots, key=lambda s: (s.shot_at is None, s.shot_at or datetime.min, s.order)
    )
    found: list[DetectedMerge] = []
    index = 0
    while index < len(ordered):
        run_end = index + 1
        while run_end < len(ordered) and _compatible(ordered[run_end - 1], ordered[run_end]):
            run_end += 1

        position = index
        while position < run_end:
            for length in _LENGTHS:
                if position + length > run_end:
                    continue
                detected = _candidate(ordered[position : position + length])
                if detected is not None:
                    found.append(detected)
                    position += length
                    break
            else:
                position += 1
        index = run_end
    return found
