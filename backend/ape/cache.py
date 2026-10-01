# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The cache and its quota (section 20.3).

Everything under ``cache/`` is regenerated from the RAWs: the browsing proxies,
the embedded previews of the culling, the developed thumbnails of the review,
the stage buffers and the intermediates of the merges. Past ``cache_max_gb``
(20 GB by default, a ``Setting`` changed from the Settings screen) the least
recently used files go, until the cache is back under 90% of the quota -- the
margin keeps one new proxy from triggering an eviction at every job.

The **merges' intermediates** are cache too, but rebuilding a panorama costs
minutes, not milliseconds: they go last, only when nothing else is left, and
the "Svuota cache" plan says how many merges will have to be rebuilt.

Three things are *not* cache and are never removed here, only counted: the
**hand-painted masks** (``masks/``), the **eraser's fills** (``retouch/``: a
fill the optional model made cannot be made again without it) and the
**models** (downloaded on request).

Nothing records that a file was removed: a missing proxy is noticed where it
is read (``api/routes_photos.py``) and queued again, marked as a restore, so
its photo is not analysed a second time. Deleting the cache by hand is
therefore as safe as this module doing it.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_settings

__all__ = [
    "CATEGORIES",
    "Category",
    "clear",
    "clear_plan",
    "enforce",
    "limit_bytes",
    "maybe_enforce",
    "usage",
]

_log = logging.getLogger(__name__)

GB = 1024**3
#: Evict down to this fraction of the quota, not just under it.
_LOW_WATER = 0.9
#: A worker checks the quota at most this often: a scan of the cache is a
#: stat per file, cheap, but not after every one of a thousand proxies.
_EVERY_S = 30.0
_last_check = 0.0
_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class Category:
    key: str
    #: Removed by the quota and by "Svuota cache".
    evictable: bool
    #: Lower goes first; the intermediates last of all.
    order: int

    def path(self) -> Path:
        settings = get_settings()
        return {
            "proxies": settings.proxy_dir,
            "previews": settings.cache_dir / "previews",
            "developed": settings.cache_dir / "developed",
            "stages": settings.stage_cache_dir,
            "intermediates": settings.intermediate_dir,
            "merges": settings.merge_preview_dir,
            "masks": settings.masks_dir,
            "retouch": settings.retouch_dir,
            "models": settings.models_dir,
        }[self.key]


CATEGORIES: tuple[Category, ...] = (
    Category("proxies", True, 0),
    Category("previews", True, 0),
    Category("developed", True, 0),
    Category("stages", True, 0),
    # The 1024 px merge previews: seconds to recompute, gone first.
    Category("merges", True, 0),
    Category("intermediates", True, 1),
    Category("masks", False, 9),
    # The eraser's fills: a fill made by a model no longer here cannot be
    # made again (docs/SPEC_rimozione.md 4.3).
    Category("retouch", False, 9),
    Category("models", False, 9),
)


def _files(root: Path) -> Iterator[os.DirEntry]:
    """Every regular file under ``root``; a folder that does not exist is empty."""
    try:
        entries = list(os.scandir(root))
    except (FileNotFoundError, NotADirectoryError):
        return
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                yield from _files(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                yield entry
        except OSError:
            continue


def _used(entry: os.DirEntry) -> float:
    """When the file was last used: its access time, or its writing if later."""
    info = entry.stat(follow_symlinks=False)
    return max(info.st_atime, info.st_mtime)


def limit_bytes(session: Any = None) -> int:
    """The quota: the ``cache_max_gb`` setting of the catalogue, or the default."""
    value = get_settings().cache_max_gb
    if session is not None:
        from .db.session import read_setting

        stored = read_setting(session, "cache_max_gb", None)
        if isinstance(stored, int | float) and stored > 0:
            value = float(stored)
    return int(value * GB)


def usage() -> dict[str, Any]:
    """Bytes and files per category, for the Settings screen."""
    categories = {}
    for category in CATEGORIES:
        size = count = 0
        for entry in _files(category.path()):
            try:
                size += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            count += 1
        categories[category.key] = {"bytes": size, "files": count, "evictable": category.evictable}
    cache = sum(v["bytes"] for v in categories.values() if v["evictable"])
    return {"categories": categories, "cache_bytes": cache}


def _intermediates_in_use() -> int:
    """How many merged photos an emptied cache would have to rebuild."""
    return sum(1 for _ in _files(get_settings().intermediate_dir))


def clear_plan() -> dict[str, Any]:
    """What "Svuota cache" frees and what it keeps, said before it runs."""
    now = usage()
    return {
        "frees_bytes": now["cache_bytes"],
        "categories": {
            key: value for key, value in now["categories"].items() if value["evictable"]
        },
        "merges_to_rebuild": _intermediates_in_use(),
        "keeps": {key: value for key, value in now["categories"].items() if not value["evictable"]},
    }


def _remove(entry: os.DirEntry) -> int:
    try:
        size = entry.stat(follow_symlinks=False).st_size
        os.unlink(entry.path)
    except FileNotFoundError:
        return 0
    except OSError as exc:
        _log.warning("non riesco a rimuovere %s dalla cache: %s", entry.path, exc)
        return 0
    return size


def enforce(limit: int | None = None) -> dict[str, int]:
    """Evict least-recently-used files until the cache is under the quota.

    Returns ``removed`` files and ``freed`` bytes.
    """
    limit = limit_bytes() if limit is None else limit
    candidates = []
    total = 0
    for category in CATEGORIES:
        if not category.evictable:
            continue
        for entry in _files(category.path()):
            try:
                size = entry.stat(follow_symlinks=False).st_size
                used = _used(entry)
            except OSError:
                continue
            total += size
            candidates.append((category.order, used, entry))
    removed = freed = 0
    if total <= limit:
        return {"removed": 0, "freed": 0}
    target = int(limit * _LOW_WATER)
    # The intermediates after everything else; inside a class, oldest use first.
    for _order, _used_at, entry in sorted(candidates, key=lambda c: (c[0], c[1])):
        if total - freed <= target:
            break
        size = _remove(entry)
        freed += size
        removed += bool(size)
    _log.info(
        "cache oltre la quota (%.1f GB su %.1f): rimossi %d file, %.1f GB",
        total / GB,
        limit / GB,
        removed,
        freed / GB,
    )
    return {"removed": removed, "freed": freed}


def maybe_enforce(session: Any = None) -> None:
    """:func:`enforce`, at most every ``_EVERY_S`` in this process. Never raises."""
    global _last_check
    now = time.monotonic()
    with _lock:
        if now - _last_check < _EVERY_S:
            return
        _last_check = now
    try:
        enforce(limit_bytes(session))
    except Exception:  # noqa: BLE001 - the quota must never fail a job
        _log.exception("controllo della quota della cache non riuscito")


def clear() -> dict[str, int]:
    """ "Svuota cache": every evictable file, now."""
    removed = freed = 0
    for category in CATEGORIES:
        if not category.evictable:
            continue
        for entry in _files(category.path()):
            size = _remove(entry)
            freed += size
            removed += bool(size)
    _log.info("cache svuotata: %d file, %.2f GB", removed, freed / GB)
    return {"removed": removed, "freed": freed}
