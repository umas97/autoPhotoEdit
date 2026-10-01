# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""From scores to a selection (section 7.4).

The analysis produces numbers; this module turns them into "keep" and "discard"
and does nothing else. It never looks at a pixel, never opens a file and never
touches the catalogue: it is a pure function of the scores and the user's
settings. That separation is binding in section 7.4 and it is what makes a
change of mode or a move of the aggressiveness slider instantaneous -- the whole
selection of two thousand photos is a few milliseconds of arithmetic.

**Every criterion is a score in [0, 1] where 1 is good.** The combined score is
their weighted mean, over the criteria the user has switched on, with the
weights the user can see and change. Nothing about the formula is hidden: the
interface shows each criterion's score and weight for any photo, so the answer
to "why was this one discarded?" is always on screen.

**Conservative mode** discards only what is technically compromised beyond
recovery -- a criterion below the threshold that the aggressiveness sets -- and
the duplicates of a burst. Everything else stays.

**Target mode** keeps a number of photos, covering before it densifies: the best
of every burst and of every single shot come first, and only when every moment
is represented does a second frame of any burst get in. Two hundred different
moments are worth more than two hundred variations of one. A technically failed
photo is never used to make up the number: the target is a ceiling, and the
interface says so when the valid photos fall short of it.

**A user's decision is final.** A photo the user promoted or discarded keeps
that state through every recomputation (test 10); it is counted, not decided.
Frames of a candidate merge are protected the same way, because section 25.7
forbids judging them by single-shot criteria.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .settings import (
    CRITERIA,
    DEFAULT_CRITERIA,
    DEFAULT_WEIGHTS,
    CullingSettings,
    target_count,
    threshold,
)

__all__ = [
    "CRITERIA",
    "DEFAULT_CRITERIA",
    "DEFAULT_WEIGHTS",
    "Candidate",
    "CullingSettings",
    "Decision",
    "Reason",
    "select",
    "summarise",
    "target_count",
    "threshold",
]

#: Inside a burst the frames show the same scene, so their sharpness can be
#: compared directly -- the finer measure section 7.3 asks for. This is how
#: much of a burst member's sharpness score comes from that comparison.
_RELATIVE_SHARPNESS_SHARE = 0.5


_SCORED = ("sharpness", "motion", "exposure", "faces", "aesthetic")


class Reason:
    """The codes stored in ``Photo.cull_reasons``. The interface translates them."""

    OUT_OF_FOCUS = "out_of_focus"
    MOTION_BLUR = "motion_blur"
    OVEREXPOSED = "overexposed"
    UNDEREXPOSED = "underexposed"
    BURST_DUPLICATE = "burst_duplicate"
    BELOW_TARGET = "below_target"
    USER = "user"


# ``slots`` and not ``frozen``: two thousand of these are built for every move
# of a slider, and a frozen dataclass sets each field through
# ``object.__setattr__`` -- measured, a quarter of the whole request. Identity
# rather than field equality, for the same reason: nothing compares two
# candidates by value.
@dataclass(slots=True, eq=False)
class Candidate:
    """One photo, as the selection sees it. Scores are in [0, 1], 1 is good."""

    id: int
    sharpness: float | None = None
    motion: float | None = None
    exposure: float | None = None
    #: ``over`` or ``under``, to name the reason when exposure fails.
    exposure_side: str | None = None
    faces: float | None = None
    aesthetic: float | None = None
    #: Laplacian energy of the sharpest region: comparable within a burst only.
    laplacian: float | None = None
    #: The measured focus ratio, before it is mapped onto the score. The score
    #: saturates -- every clearly sharp photo is 100 -- and a target mode that
    #: must choose among forty photos at 100 should choose the crispest, not
    #: the earliest.
    acuity: float | None = None
    burst_group: int | None = None
    #: ``True`` kept by the user, ``False`` discarded by the user, ``None`` auto.
    user_keep: bool | None = None
    #: Member of a candidate merge: never judged by single-shot criteria.
    protected: bool = False
    #: The culling analysis has run on this photo.
    analysed: bool = True
    #: Shooting order, for stable tie-breaks.
    order: tuple = ()


@dataclass(slots=True, eq=False)
class Decision:
    id: int
    culled: bool
    reasons: tuple[str, ...]
    score: float | None
    #: Position inside its burst, 0 for the proposed frame; 0 for single shots.
    rank: int
    #: The scores that went into ``score``, for the "why" of the interface.
    criteria: Mapping[str, float] = field(default_factory=dict)


def _criterion_scores(
    candidate: Candidate, settings: CullingSettings, best_laplacian: float | None
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name in _SCORED:
        if not settings.criteria.get(name):
            continue
        value = getattr(candidate, name)
        if value is None:
            continue
        if (
            name == "sharpness"
            and best_laplacian
            and candidate.laplacian is not None
            and settings.criteria.get("burst")
        ):
            relative = min(1.0, candidate.laplacian / best_laplacian)
            share = _RELATIVE_SHARPNESS_SHARE
            value = (1.0 - share) * value + share * relative
        scores[name] = float(value)
    return scores


def _combined(scores: Mapping[str, float], weights: Mapping[str, float]) -> float | None:
    total = weighted = 0.0
    for name, value in scores.items():
        weight = weights.get(name, 0.0)
        total += weight
        weighted += weight * value
    if total <= 0.0:
        return None if not scores else sum(scores.values()) / len(scores)
    return weighted / total


def _technical_failures(
    candidate: Candidate, settings: CullingSettings, limit: float
) -> list[str]:
    """Which criteria this photo fails beyond recovery. Absolute, never relative."""
    failures: list[str] = []
    checks = (
        ("sharpness", candidate.sharpness, Reason.OUT_OF_FOCUS),
        ("motion", candidate.motion, Reason.MOTION_BLUR),
    )
    for name, value, reason in checks:
        if settings.criteria.get(name) and value is not None and value < limit:
            failures.append(reason)
    if (
        settings.criteria.get("exposure")
        and candidate.exposure is not None
        and candidate.exposure < limit
    ):
        failures.append(
            Reason.UNDEREXPOSED if candidate.exposure_side == "under" else Reason.OVEREXPOSED
        )
    return failures


def select(candidates: Iterable[Candidate], settings: CullingSettings) -> list[Decision]:
    """Decide every photo. Pure: same inputs, same decisions, in input order."""
    photos = list(candidates)
    use_bursts = bool(settings.criteria.get("burst"))

    groups: dict[int, list[Candidate]] = {}
    if use_bursts:
        for photo in photos:
            if photo.burst_group is not None and not photo.protected:
                groups.setdefault(photo.burst_group, []).append(photo)
        groups = {key: members for key, members in groups.items() if len(members) > 1}

    best_laplacian: dict[int, float] = {}
    for key, members in groups.items():
        values = [m.laplacian for m in members if m.laplacian]
        if values:
            best_laplacian[key] = max(values)

    scores: dict[int, dict[str, float]] = {}
    combined: dict[int, float | None] = {}
    failures: dict[int, list[str]] = {}
    limit = threshold(settings.aggressiveness)
    for photo in photos:
        group = photo.burst_group if photo.burst_group in groups else None
        scores[photo.id] = _criterion_scores(
            photo, settings, best_laplacian.get(group) if group is not None else None
        )
        combined[photo.id] = _combined(scores[photo.id], settings.weights)
        auto = photo.user_keep is None and not photo.protected and photo.analysed
        failures[photo.id] = _technical_failures(photo, settings, limit) if auto else []

    def merit(photo: Candidate) -> tuple:
        # Higher is better. Equal scores are broken by the measured sharpness,
        # then by the earlier frame -- the one the photographer saw first, and
        # the stable choice across recomputations.
        value = combined[photo.id]
        return (
            -(value if value is not None else -1.0),
            -(photo.acuity if photo.acuity is not None else 0.0),
            photo.order,
            photo.id,
        )

    rank: dict[int, int] = {photo.id: 0 for photo in photos}
    culled: dict[int, bool] = {}
    reasons: dict[int, list[str]] = {}

    for photo in photos:
        if photo.user_keep is not None:
            culled[photo.id] = not photo.user_keep
            reasons[photo.id] = [] if photo.user_keep else [Reason.USER]
        elif photo.protected or not photo.analysed:
            culled[photo.id], reasons[photo.id] = False, []
        else:
            culled[photo.id] = bool(failures[photo.id])
            reasons[photo.id] = list(failures[photo.id])

    # Units of coverage: each burst is one moment, each single shot another.
    units: list[list[Candidate]] = [sorted(m, key=merit) for m in groups.values()]
    grouped = {m.id for members in groups.values() for m in members}
    units += [[photo] for photo in photos if photo.id not in grouped]

    for members in units:
        if len(members) < 2:
            continue
        # The proposed frame: the user's choice if they made one, otherwise the
        # best frame that is not itself a technical failure.
        chosen = [m for m in members if m.user_keep]
        if not chosen:
            viable = [m for m in members if m.user_keep is None and not failures[m.id]]
            chosen = viable[:1]
        order = chosen + [m for m in members if m not in chosen]
        for position, member in enumerate(order):
            rank[member.id] = position
            if member in chosen or member.user_keep is not None:
                continue
            if not culled[member.id]:
                culled[member.id] = True
            if Reason.BURST_DUPLICATE not in reasons[member.id]:
                reasons[member.id].append(Reason.BURST_DUPLICATE)

    target = target_count(settings, len(photos))
    if target is not None:
        # Undo the burst logic's choices for the automatic photos and fill the
        # quota in rounds instead: round one takes the best frame of every
        # unit, round two the second, and so on.
        # What is kept whatever the quota: the user's own keeps, the frames of
        # a merge -- unless the user discarded one -- and what was never judged.
        def fixed(p: Candidate) -> bool:
            if p.user_keep is not None:
                return p.user_keep
            return p.protected or not p.analysed

        kept = sum(1 for p in photos if fixed(p))
        quota = max(0, target - kept)
        rounds: list[list[Candidate]] = []
        for members in units:
            pool = [
                m for m in members if m.user_keep is None and not m.protected
                and m.analysed and not failures[m.id]
            ]
            covered = any(fixed(m) for m in members)
            for position, member in enumerate(pool):
                depth = position + (1 if covered else 0)
                while len(rounds) <= depth:
                    rounds.append([])
                rounds[depth].append(member)
        taken: set[int] = set()
        for layer in rounds:
            for member in sorted(layer, key=merit):
                if quota <= 0:
                    break
                taken.add(member.id)
                quota -= 1
        for members in units:
            for member in members:
                if member.user_keep is not None or member.protected or not member.analysed:
                    continue
                if failures[member.id]:
                    continue
                if member.id in taken:
                    culled[member.id], reasons[member.id] = False, []
                else:
                    culled[member.id] = True
                    in_burst = len(members) > 1 and any(
                        m.id in taken or fixed(m) for m in members
                    )
                    reasons[member.id] = [
                        Reason.BURST_DUPLICATE if in_burst else Reason.BELOW_TARGET
                    ]

    return [
        Decision(
            id=photo.id,
            culled=culled[photo.id],
            reasons=tuple(reasons[photo.id]),
            score=None if combined[photo.id] is None else round(combined[photo.id], 4),
            rank=rank[photo.id],
            criteria={k: round(v, 4) for k, v in scores[photo.id].items()},
        )
        for photo in photos
    ]


def summarise(decisions: Sequence[Decision]) -> dict[str, int]:
    """Counts for the bottom bar of section 7.6."""
    selected = sum(1 for d in decisions if not d.culled)
    return {"selected": selected, "culled": len(decisions) - selected, "total": len(decisions)}
