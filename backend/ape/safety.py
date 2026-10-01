# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime guard for the non-destructiveness invariant (docs/SPEC.md section 2).

The RAW files the user points us at are read-only, always, everywhere. This
module is the last safety net under that rule: every function in the codebase
that opens a file for writing calls :func:`assert_outside_source` first.

It is deliberately paranoid:

* comparisons happen on ``Path.resolve()``, so a symlink that leads back into
  the source folder is caught;
* both the target and its parent directory are checked, because creating a file
  inside a symlinked directory writes into whatever that symlink points at;
* on top of the source directory passed by the caller, a process-wide registry
  of protected roots is consulted, so forgetting to thread a project through a
  call chain cannot silently disable the guard.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

__all__ = [
    "SIDECAR_SUFFIX",
    "SourceWriteError",
    "assert_outside_source",
    "create_beside_source",
    "guarded_open",
    "is_inside",
    "protected_roots",
    "register_protected_root",
    "unregister_protected_root",
]

# Process-wide set of directories that must never be written into. Projects
# register their source folder here on open; the CLI registers the folder of any
# RAW it reads.
_PROTECTED: set[Path] = set()


class SourceWriteError(RuntimeError):
    """Raised when something tries to write inside a protected source folder."""


def _resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve without requiring the path to exist, following symlinks."""
    return Path(path).expanduser().resolve()


def is_inside(candidate: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    """True when ``candidate`` is ``root`` itself or lies underneath it.

    Both sides are fully resolved first, so symlinks cannot be used to sneak a
    path past the comparison.
    """
    resolved_root = _resolve(root)
    resolved_candidate = _resolve(candidate)
    if resolved_candidate == resolved_root:
        return True
    return resolved_root in resolved_candidate.parents


def register_protected_root(path: str | os.PathLike[str]) -> Path:
    """Mark a directory as read-only for the whole process. Idempotent."""
    resolved = _resolve(path)
    _PROTECTED.add(resolved)
    return resolved


def unregister_protected_root(path: str | os.PathLike[str]) -> None:
    _PROTECTED.discard(_resolve(path))


def protected_roots() -> frozenset[Path]:
    return frozenset(_PROTECTED)


def _source_dirs(source: Any) -> Iterator[Path]:
    """Yield source directories from a Project, a path, or a collection of either."""
    if source is None:
        return
    if isinstance(source, str | os.PathLike):
        yield _resolve(source)
        return
    # Duck-typed Project (the ORM model only exists from phase 2 on).
    source_dir = getattr(source, "source_dir", None)
    if source_dir is not None:
        yield _resolve(source_dir)
        return
    if isinstance(source, Iterable):
        for item in source:
            yield from _source_dirs(item)
        return
    raise TypeError(f"cannot derive a source directory from {source!r}")


def assert_outside_source(path: str | os.PathLike[str], source: Any = None) -> Path:
    """Verify that ``path`` may be written to, and return it resolved.

    ``source`` may be a Project, a directory path, a collection of either, or
    ``None`` -- in which case only the process-wide protected roots are checked.

    Raises:
        SourceWriteError: if the destination falls inside a protected folder.
    """
    target = _resolve(path)
    # The parent matters as much as the target: writing "out.jpg" into a
    # directory that is a symlink to the source folder still writes into the
    # source folder.
    candidates = (target, target.parent)

    roots = set(_source_dirs(source)) | _PROTECTED
    for root in roots:
        if any(is_inside(candidate, root) for candidate in candidates):
            raise SourceWriteError(
                f"rifiutata una scrittura dentro la cartella sorgente: {target} "
                f"(sorgente protetta: {root}). "
                "I file originali non vengono mai modificati (docs/SPEC.md §2)."
            )
    return target


@contextmanager
def guarded_open(
    path: str | os.PathLike[str],
    mode: str = "wb",
    source: Any = None,
    **kwargs: Any,
) -> Iterator[BinaryIO]:
    """``open()`` for writing, refused when the destination is a protected path.

    Read-only modes are rejected outright: reading does not belong here, and
    accepting it would make the guard look optional at call sites.
    """
    if not any(flag in mode for flag in ("w", "a", "x", "+")):
        raise ValueError(f"guarded_open is for writing; mode {mode!r} is read-only")
    target = assert_outside_source(path, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately not a `with`: the handle is what this context manager yields,
    # and it is closed in the `finally` below.
    handle = open(target, mode, **kwargs)  # noqa: SIM115
    try:
        yield handle
    finally:
        handle.close()


#: The one kind of file the program may ever create inside a source folder.
SIDECAR_SUFFIX = ".xmp"


def create_beside_source(raw_path: str | os.PathLike[str], name: str, payload: bytes) -> bool:
    """Create a new XMP sidecar next to a RAW. The one exception to this module.

    Section 2.4 allows sidecars next to the originals behind an explicit option,
    off by default, that *adds* files without modifying any. This function is
    that option's only way in, and it is narrow on purpose:

    * the name must end in ``.xmp`` and share the RAW's stem (``DSC0001.xmp``
      or ``DSC0001.ARW.xmp``) -- nothing else can be created this way;
    * the file is opened with ``O_CREAT | O_EXCL``: if a sidecar is already
      there -- the user's own darktable or Lightroom edit, most likely -- it is
      left exactly as it is and this returns ``False``;
    * the RAW itself is never opened, and no existing file is ever touched.

    Returns:
        ``True`` if the file was created, ``False`` if one already existed.

    Raises:
        SourceWriteError: for any name that is not a sidecar of that RAW, or a
            RAW that is not a regular file.
    """
    raw = _resolve(raw_path)
    if not raw.is_file():
        raise SourceWriteError(f"{raw} non è un file RAW: nessun sidecar accanto")
    allowed = {f"{raw.stem}{SIDECAR_SUFFIX}".casefold(), f"{raw.name}{SIDECAR_SUFFIX}".casefold()}
    if name.casefold() not in allowed or "/" in name or "\\" in name:
        raise SourceWriteError(
            f"«{name}» non è un sidecar XMP di {raw.name}: nella cartella sorgente "
            "si possono solo aggiungere sidecar, mai altri file (docs/SPEC.md §2.4)"
        )
    target = raw.parent / name
    if target.is_symlink():
        raise SourceWriteError(f"{target} è un collegamento: non lo seguo")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return True
