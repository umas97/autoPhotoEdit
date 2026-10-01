# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scanning a folder into the catalogue (docs/SPEC.md section 15).

Importing twice must cost the user nothing. That single sentence shapes the
whole module: a photo is identified by its *content*, never by its path, so a
renamed file is the same photo, a copy is not a second photo, and a re-import
adds only what is genuinely new. Edits, culling decisions, clusters and
approvals survive untouched, because the rows survive untouched.

How that identity is computed is the subject of ``identity.py``. What this
module does with it is decide, for each file in the folder, whether it is a
photo the catalogue already has, a second copy of one, or something new.

Files that vanish are marked ``missing`` and kept. Nothing here writes anywhere
near the source folder: the only file operation in this module is ``open(..,
'rb')``, and ``safety.register_protected_root`` is called on the source before
any of it starts.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db.models import Photo, PhotoKind, Project
from .identity import content_key, full_content_hash
from .safety import register_protected_root

__all__ = [
    "ImportSummary",
    "RAW_EXTENSION",
    "import_folder",
    "remap_source",
    "scan_folder",
]

_log = logging.getLogger(__name__)

#: The only extension that becomes a Photo. Sony, and nothing else (section 1).
RAW_EXTENSION = ".arw"

#: Recognised, deliberately refused. Named one by one so the import summary can
#: say "14 file .CR2 non importati (solo Sony ARW)" instead of staying silent.
OTHER_RAW_EXTENSIONS = frozenset(
    {
        ".3fr", ".arq", ".cr2", ".cr3", ".crw", ".dcr", ".dng", ".erf", ".fff",
        ".iiq", ".k25", ".kdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf",
        ".pef", ".raf", ".raw", ".rw2", ".rwl", ".sr2", ".srf", ".srw", ".x3f",
    }
)

_SIDECAR_EXTENSIONS = (".jpg", ".jpeg")


@dataclass
class ScanResult:
    """What is in the folder, before the catalogue has been consulted."""

    raws: list[Path] = field(default_factory=list)
    #: ``{basename lowercased: jpeg path}`` for the sidecars of section 15.
    sidecars: dict[str, Path] = field(default_factory=dict)
    subdirectories: list[Path] = field(default_factory=list)
    #: ``{extension: count}`` of RAW files from other manufacturers.
    rejected: dict[str, int] = field(default_factory=dict)
    #: Everything else, counted but not detailed.
    other_files: int = 0


def scan_folder(folder: str | Path) -> ScanResult:
    """List a folder, non-recursively, sorting what is in it by role.

    Raises:
        FileNotFoundError: if the folder is not there. A removed external disk
            is a *project* state (``source_missing``), not an exception, and is
            handled by the caller.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"cartella non trovata: {root}")

    result = ScanResult()
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            result.subdirectories.append(entry)
            continue
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix == RAW_EXTENSION:
            result.raws.append(entry)
        elif suffix in _SIDECAR_EXTENSIONS:
            result.sidecars[entry.stem.lower()] = entry
        elif suffix in OTHER_RAW_EXTENSIONS:
            result.rejected[suffix] = result.rejected.get(suffix, 0) + 1
        else:
            result.other_files += 1

    # A JPEG only counts as a sidecar when an ARW of the same basename exists;
    # a folder of plain JPEGs is a folder this program has nothing to say about.
    stems = {p.stem.lower() for p in result.raws}
    orphan_jpegs = [stem for stem in result.sidecars if stem not in stems]
    for stem in orphan_jpegs:
        del result.sidecars[stem]
        result.other_files += 1
    return result


@dataclass
class ImportSummary:
    """What an import did, in the terms the user needs to hear it in."""

    imported: int = 0
    already_present: int = 0
    moved: int = 0
    duplicates: int = 0
    restored: int = 0
    marked_missing: int = 0
    sidecars: int = 0
    subdirectories_ignored: int = 0
    rejected_formats: dict[str, int] = field(default_factory=dict)
    other_files: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_seen(self) -> int:
        return self.imported + self.already_present + self.moved + self.duplicates

    def describe(self) -> str:
        """One line per fact, in Italian, for the CLI and the import screen."""
        lines = [f"{self.imported} foto importate"]
        if self.already_present:
            lines.append(f"{self.already_present} già presenti, invariate")
        if self.moved:
            lines.append(
                f"{self.moved} riconosciute dal contenuto dopo un rinomina o uno spostamento"
            )
        if self.duplicates:
            lines.append(f"{self.duplicates} duplicati per contenuto, importati una sola volta")
        if self.restored:
            lines.append(f"{self.restored} foto ricomparse nella sorgente")
        if self.marked_missing:
            lines.append(f"{self.marked_missing} foto non più nella sorgente, marcate mancanti")
        if self.sidecars:
            lines.append(f"{self.sidecars} JPEG affiancati collegati (non importati come foto)")
        if self.subdirectories_ignored:
            lines.append(
                f"{self.subdirectories_ignored} sottocartelle ignorate "
                "(la scansione non è ricorsiva)"
            )
        for extension, count in sorted(self.rejected_formats.items()):
            lines.append(f"{count} file {extension} non importati: solo Sony ARW")
        for name, reason in self.failed:
            lines.append(f"{name}: {reason}")
        return "\n".join(lines)


def _existing_by_key(session: Session, project_id: int) -> dict[str, Photo]:
    photos = session.scalars(
        select(Photo).where(Photo.project_id == project_id, Photo.kind == PhotoKind.RAW)
    ).all()
    return {photo.hash: photo for photo in photos if photo.hash}


def _promote_to_full_hash(session: Session, photo: Photo) -> str:
    """Move a row from its quick key to its full hash, once one is needed."""
    if photo.hash and photo.hash.startswith("s:"):
        return photo.hash
    if not photo.path:
        return photo.hash or ""
    photo.hash = full_content_hash(Path(photo.path))
    session.flush()
    return photo.hash


def import_folder(
    session: Session,
    project: Project,
    folder: str | Path | None = None,
    *,
    mark_missing: bool = True,
) -> ImportSummary:
    """Scan ``folder`` into ``project``, adding only what is new.

    Args:
        session: an open session; the caller commits.
        project: the project to import into.
        folder: defaults to the project's own ``source_dir``.
        mark_missing: mark as ``missing`` the photos whose file is no longer in
            the folder. Off when importing a *second* folder into a project,
            where absence from this folder means nothing.

    Returns:
        An :class:`ImportSummary`. Nothing outside the catalogue is touched.
    """
    root = Path(folder or project.source_dir).expanduser()
    register_protected_root(root)

    scan = scan_folder(root)
    summary = ImportSummary(
        subdirectories_ignored=len(scan.subdirectories),
        rejected_formats=dict(scan.rejected),
        other_files=scan.other_files,
    )

    known = _existing_by_key(session, project.id)
    seen_photos: set[int] = set()

    for path in scan.raws:
        try:
            key = content_key(path)
        except OSError as exc:
            summary.failed.append((path.name, f"illeggibile ({exc.strerror or exc})"))
            continue

        photo = known.get(key)
        if photo is not None:
            photo = _resolve_candidate(session, photo, path, key, summary, known)
        if photo is None:
            photo = _insert_photo(session, project, path, key)
            known[photo.hash or key] = photo
            summary.imported += 1
        seen_photos.add(photo.id)

        sidecar = scan.sidecars.get(path.stem.lower())
        if sidecar is not None:
            if photo.sidecar_jpeg_path != str(sidecar):
                photo.sidecar_jpeg_path = str(sidecar)
            summary.sidecars += 1

    if mark_missing:
        summary.marked_missing = _mark_missing(session, project, seen_photos)
    summary.restored = _restore_present(session, seen_photos)
    project.source_missing = False
    return summary


def _resolve_candidate(
    session: Session,
    photo: Photo,
    path: Path,
    key: str,
    summary: ImportSummary,
    known: dict[str, Photo],
) -> Photo | None:
    """Decide what an already-known quick key means for this particular path.

    Four outcomes: the same file again (nothing to do), the same file under a
    new name (follow it), a second copy of it (record the extra path), or --
    vanishingly unlikely but not impossible -- a different file that happens to
    share the quick key, in which case both rows move to their full hashes and
    the caller inserts a new photo.
    """
    if photo.path and Path(photo.path) == path:
        summary.already_present += 1
        if photo.missing:
            photo.missing = False
        return photo

    # Same content, and the path the catalogue knows is no longer on disk: the
    # file was renamed or moved within the folder. Identity is content, so this
    # is the same photo and it keeps everything that was ever done to it.
    if not photo.path or not Path(photo.path).exists():
        photo.path = str(path)
        photo.filename = path.name
        photo.missing = False
        summary.moved += 1
        return photo

    # Same quick key, two files that both exist: confirm before deciding.
    existing_full = _promote_to_full_hash(session, photo)
    known.pop(key, None)
    known[photo.hash or key] = photo
    candidate_full = full_content_hash(path)

    if candidate_full == existing_full:
        paths = list(photo.duplicate_paths or [])
        if str(path) not in paths:
            paths.append(str(path))
            photo.duplicate_paths = paths
        summary.duplicates += 1
        return photo

    _log.warning(
        "collisione della chiave rapida fra %s e %s: uso l'hash completo",
        photo.path,
        path,
    )
    return None


def _insert_photo(session: Session, project: Project, path: Path, key: str) -> Photo:
    """Create the row. Metadata comes later, in the analysis job."""
    photo = Photo(
        project_id=project.id,
        path=str(path),
        filename=path.name,
        hash=key,
        kind=PhotoKind.RAW,
    )
    session.add(photo)
    session.flush()
    return photo


def _mark_missing(session: Session, project: Project, seen: Iterable[int]) -> int:
    seen_ids = set(seen)
    photos = session.scalars(
        select(Photo).where(
            Photo.project_id == project.id,
            Photo.kind == PhotoKind.RAW,
            Photo.missing.is_(False),
        )
    ).all()
    count = 0
    for photo in photos:
        if photo.id not in seen_ids:
            photo.missing = True
            count += 1
    return count


def _restore_present(session: Session, seen: Iterable[int]) -> int:
    seen_ids = list(seen)
    if not seen_ids:
        return 0
    photos = session.scalars(
        select(Photo).where(Photo.id.in_(seen_ids), Photo.missing.is_(True))
    ).all()
    for photo in photos:
        photo.missing = False
    return len(photos)


@dataclass
class RemapSummary:
    reattached: int = 0
    by_filename: int = 0
    still_missing: int = 0


def remap_source(session: Session, project: Project, new_folder: str | Path) -> RemapSummary:
    """Point a project at the same photos in a new place (section 15).

    Matching is by content first -- authoritative, and indifferent to renames --
    and by filename only for what content could not place. A photo that neither
    finds stays ``missing``: the row keeps every edit made to it, ready for the
    day the disk comes back.
    """
    root = Path(new_folder).expanduser()
    register_protected_root(root)
    scan = scan_folder(root)

    photos = session.scalars(
        select(Photo).where(Photo.project_id == project.id, Photo.kind == PhotoKind.RAW)
    ).all()
    by_key: dict[str, Photo] = {p.hash: p for p in photos if p.hash}
    by_name: dict[str, Photo] = {p.filename.lower(): p for p in photos}

    summary = RemapSummary()
    matched: set[int] = set()

    for path in scan.raws:
        key = content_key(path)
        photo = by_key.get(key)
        if photo is None and key.startswith("q:"):
            # The row may have been promoted to a full hash by an earlier
            # collision; that is the only other place its identity can be.
            photo = by_key.get(full_content_hash(path))
        if photo is not None:
            photo.path = str(path)
            photo.filename = path.name
            photo.missing = False
            matched.add(photo.id)
            summary.reattached += 1
            continue

        photo = by_name.get(path.name.lower())
        if photo is not None and photo.id not in matched:
            photo.path = str(path)
            photo.hash = key
            photo.missing = False
            matched.add(photo.id)
            summary.by_filename += 1

    for photo in photos:
        if photo.id not in matched:
            photo.missing = True
            summary.still_missing += 1

    project.source_dir = str(root)
    project.source_missing = summary.reattached + summary.by_filename == 0
    return summary
