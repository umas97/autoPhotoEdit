# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which edited file belongs to which RAW (section 8.1).

Section 8.1 pairs by base name and falls back to a manual screen. Base names
are the exception rather than the rule in practice: the user's own delivery of
65 photos is named ``evento_001.jpg`` .. ``evento_065.jpg``, against
``DSC06311.ARW`` .. ``DSC06414.ARW``. So two more automatic clues come before
the manual screen, in decreasing order of certainty:

1. **the base name** -- ``DSC06312.jpg`` for ``DSC06312.ARW``;
2. **the original file name the editor recorded** -- Lightroom and Camera Raw
   write ``xmpMM:PreservedFileName`` / ``crs:RawFileName`` on export (all 65
   of the user's files carry it);
3. **the shooting time and the body** -- unique to the second on a single
   camera, to the hundredth with ``SubSecTimeOriginal``. A time shared by
   two RAWs (a burst on a body without sub-seconds) pairs nothing: a guess is
   worse than a question.

What stays unpaired goes to the manual screen, and a RAW claimed by two edits
(a virtual copy) keeps the first and reports the rest: two different targets for
one scene would teach the profile a contradiction.

Pure: paths and metadata in, pairs out. Reading never writes (section 2).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..raw.metadata import ReferenceIdentity, read_metadata, read_reference_identity

__all__ = [
    "PAIRING_METHODS",
    "Pair",
    "PairingResult",
    "REFERENCE_SUFFIXES",
    "list_references",
    "pair_files",
]

REFERENCE_SUFFIXES = frozenset({".jpg", ".jpeg", ".tif", ".tiff"})
RAW_SUFFIXES = frozenset({".arw"})

#: How a pair was made, for the interface: the user checks the weaker ones.
PAIRING_METHODS = ("name", "xmp", "time", "manual")


@dataclass(frozen=True, slots=True)
class Pair:
    raw: Path
    reference: Path
    method: str


@dataclass(slots=True)
class PairingResult:
    pairs: list[Pair] = field(default_factory=list)
    unpaired_raws: list[Path] = field(default_factory=list)
    unpaired_references: list[Path] = field(default_factory=list)
    #: References whose RAW was already claimed by another reference.
    duplicates: list[Pair] = field(default_factory=list)


def _files(folder: Path, suffixes: frozenset[str]) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def list_references(folder: str | Path) -> list[Path]:
    return _files(Path(folder).expanduser(), REFERENCE_SUFFIXES)


def _time_key(shot_at, subsec, camera) -> tuple | None:
    if shot_at is None:
        return None
    return (shot_at.replace(microsecond=0), (subsec or "").strip()[:2], (camera or "").upper())


def pair_files(
    raws: list[Path],
    references: list[Path],
    *,
    identities: dict[Path, ReferenceIdentity] | None = None,
    raw_times: dict[Path, tuple] | None = None,
) -> PairingResult:
    """Pair RAWs and edited references with the three clues above.

    Args:
        raws: candidate RAW files.
        references: candidate edited files.
        identities: pre-read reference metadata (tests); read from disk if absent.
        raw_times: pre-read ``_time_key`` of each RAW (tests); read if absent.
    """
    result = PairingResult()
    by_stem: dict[str, Path] = {}
    for raw in raws:
        by_stem.setdefault(raw.stem.lower(), raw)
    claimed: dict[Path, Pair] = {}
    pending: list[Path] = []

    def claim(raw: Path, reference: Path, method: str) -> None:
        pair = Pair(raw, reference, method)
        if raw in claimed:
            result.duplicates.append(pair)
        else:
            claimed[raw] = pair

    for reference in references:
        raw = by_stem.get(reference.stem.lower())
        if raw is not None:
            claim(raw, reference, "name")
        else:
            pending.append(reference)

    identities = dict(identities or {})
    still: list[Path] = []
    for reference in pending:
        identity = identities.get(reference)
        if identity is None:
            identity = identities[reference] = read_reference_identity(reference)
        original = Path(identity.original_name).stem.lower() if identity.original_name else None
        raw = by_stem.get(original) if original else None
        if raw is not None:
            claim(raw, reference, "xmp")
        else:
            still.append(reference)

    if still:
        if raw_times is None:
            raw_times = {}
            for raw in raws:
                if raw in claimed:
                    continue
                meta = read_metadata(raw)
                subsec = meta.raw_tags.get("Exif.Photo.SubSecTimeOriginal")
                key = _time_key(meta.shot_at, subsec, meta.camera_model)
                if key is not None:
                    raw_times[raw] = key
        by_time: dict[tuple, list[Path]] = defaultdict(list)
        for raw, key in raw_times.items():
            if raw not in claimed:
                by_time[key].append(raw)
        for reference in still:
            identity = identities[reference]
            key = _time_key(identity.shot_at, identity.subsec, identity.camera_model)
            candidates = by_time.get(key, []) if key else []
            if len(candidates) == 1:
                claim(candidates[0], reference, "time")
            else:
                result.unpaired_references.append(reference)

    result.pairs = sorted(claimed.values(), key=lambda p: p.raw.name)
    result.unpaired_raws = [raw for raw in raws if raw not in claimed]
    return result
