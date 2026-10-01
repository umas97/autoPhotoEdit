# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Running a merge: from a recipe to a linear frame (section 25).

A :class:`Recipe` is everything that decides the result -- which files, which
one is the reference, which options -- and nothing else, so its digest names
the intermediate: the same recipe always gives the same file, and deleting the
file loses nothing (section 25.1). This module does not know the catalogue;
``virtual.py`` builds recipes from it and records what comes out.

Two resolutions. The **preview** decodes every member at half size and merges
at 1024 px, in a few seconds, so the user can see whether a panorama stitches
before spending minutes on it (section 25.6). The **full** merge decodes at
full resolution and writes the intermediate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..raw.decode import DecodedRaw, decode_linear
from .errors import MergeFailure
from .warp import shrink

__all__ = [
    "MERGE_VERSION",
    "PREVIEW_EDGE",
    "Member",
    "MergeOutcome",
    "Recipe",
    "run_full",
    "run_preview",
]

#: Bumped when the same recipe would produce different pixels: part of the
#: digest, so an intermediate from an older engine is rebuilt, not reused.
MERGE_VERSION = 3

#: Long edge of the preview (section 25.6).
PREVIEW_EDGE = 1024

Progress = Callable[[float], None]


@dataclass(frozen=True)
class Member:
    photo_id: int
    path: str
    filename: str
    #: Content hash of the RAW: the digest follows the file, not its name.
    hash: str | None
    #: Stops relative to the reference (HDR); ``None`` when not meaningful.
    ev_offset: float | None
    reference: bool
    shot_at: datetime | None = None


@dataclass(frozen=True)
class Recipe:
    group_id: int
    kind: str
    #: In position order.
    members: tuple[Member, ...]
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def reference(self) -> Member:
        for member in self.members:
            if member.reference:
                return member
        return self.members[len(self.members) // 2]

    def digest(self) -> str:
        payload = {
            "engine": MERGE_VERSION,
            "kind": self.kind,
            "members": [[m.hash or m.path, m.reference] for m in self.members],
            "options": self.options,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class MergeOutcome:
    """A merged frame, with the reference's decode context around it."""

    decoded: DecodedRaw
    report: dict[str, Any]
    #: Whether the lens stage should still correct the merge (see
    #: ``intermediate.context_for``).
    keep_lens: bool = True


def _decode(member: Member, *, preview: bool, white_balance: np.ndarray | None = None):
    try:
        decoded = decode_linear(member.path, half_size=preview, white_balance=white_balance)
    except FileNotFoundError as exc:
        raise MergeFailure(f"{member.filename} non è più nella cartella sorgente") from exc
    except ValueError as exc:
        raise MergeFailure(str(exc)) from exc
    if preview:
        small, _ = shrink(decoded.rgb, PREVIEW_EDGE)
        decoded.rgb = np.ascontiguousarray(small)
    return decoded


def _check(recipe: Recipe) -> None:
    if len(recipe.members) < 2:
        raise MergeFailure("una fusione ha bisogno di almeno due scatti")


def _hdr(recipe: Recipe, *, preview: bool, progress: Progress | None) -> MergeOutcome:
    from .hdr import HdrOptions, merge_hdr

    reference = _decode(recipe.reference, preview=preview)
    balance = reference.camera.as_shot_multipliers
    others = [
        (
            (lambda m=member: _decode(m, preview=preview, white_balance=balance).rgb),
            float(member.ev_offset or 0.0),
        )
        for member in recipe.members
        if not member.reference
    ]
    options = HdrOptions(deghost=bool(recipe.options.get("deghost", True)))
    merged, report = merge_hdr(reference.rgb, others, options, progress=progress)
    reference.rgb = merged
    return MergeOutcome(decoded=reference, report=report.as_json())


def _dispatch(recipe: Recipe, *, preview: bool, progress: Progress | None) -> MergeOutcome:
    _check(recipe)
    if recipe.kind == "hdr":
        return _hdr(recipe, preview=preview, progress=progress)
    if recipe.kind == "focus_stack":
        from .focus_stack import run as run_stack

        return run_stack(recipe, preview=preview, progress=progress, decode=_decode)
    if recipe.kind == "panorama":
        from .panorama import run as run_panorama

        return run_panorama(recipe, preview=preview, progress=progress, decode=_decode)
    raise MergeFailure(f"tipo di fusione sconosciuto: {recipe.kind}")


def run_preview(recipe: Recipe, *, progress: Progress | None = None) -> MergeOutcome:
    """The merge at 1024 px, from half-size decodes."""
    return _dispatch(recipe, preview=True, progress=progress)


def run_full(
    recipe: Recipe, destination: Path, *, reduced_long_edge: int,
    progress: Progress | None = None,
) -> tuple[Path, dict[str, Any], tuple[int, int]]:
    """The merge at full resolution, written to ``destination``.

    Returns the intermediate's path, the report and the frame's ``(width,
    height)``. A panorama writes its bands as it composes them; the others
    produce one frame and write it at the end.
    """
    from ..raw.intermediate import context_for, write_intermediate
    from ..raw.metadata import read_metadata

    reference = recipe.reference
    metadata = read_metadata(reference.path)
    first = min((m.shot_at for m in recipe.members if m.shot_at), default=None)
    notes = {
        "kind": recipe.kind,
        "members": [m.filename for m in recipe.members],
        "reference": reference.filename,
        "digest": recipe.digest(),
    }

    if recipe.kind == "panorama":
        from .panorama import run_full as panorama_full

        def context(decoded: DecodedRaw, width: int, height: int) -> dict:
            meta = replace(metadata, width=width, height=height, shot_at=first or metadata.shot_at)
            return context_for(decoded, meta, lens=False, **notes)

        return panorama_full(
            recipe, destination, context, reduced_long_edge=reduced_long_edge,
            progress=progress, decode=_decode,
        )

    outcome = _dispatch(recipe, preview=False, progress=progress)
    height, width = outcome.decoded.rgb.shape[:2]
    meta = replace(metadata, width=width, height=height, shot_at=first or metadata.shot_at)
    context = context_for(outcome.decoded, meta, lens=outcome.keep_lens, **notes)
    path = write_intermediate(
        destination, outcome.decoded.rgb, context, reduced_long_edge=reduced_long_edge
    )
    return path, outcome.report, (width, height)
