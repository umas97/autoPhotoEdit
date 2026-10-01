# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Backup and restore of the catalogue (section 20.2).

**Backup** is SQLite's ``VACUUM INTO``: a consistent copy even while the server
is writing, compacted, in one statement. It holds the catalogue only -- no
cache, no proxy: those are regenerated from the RAWs, and a backup that
carried them would be too big to be made at all. The hand-painted masks are
not cache either (section 20.3) and live in their own folder, and so do the
eraser's fills: the command says where, so that they travel with the backup.

**Restore** puts the current catalogue aside, under a dated name next to it,
before copying the backup in: a restore of the wrong file is itself
recoverable. It refuses while the program is running -- the server holds the
catalogue open, and swapping the file under it is how a catalogue is
corrupted -- and refuses a file that is not an autoPhotoEdit catalogue, or one
from a newer build. An older one is migrated at the next start, as always.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import get_settings
from .safety import assert_outside_source

__all__ = ["BackupError", "backup_catalogue", "restore_catalogue"]


class BackupError(RuntimeError):
    """A backup or restore that cannot be done, in words for the terminal."""


def _schema_of(path: Path) -> int:
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = connection.execute(
                "SELECT value FROM setting WHERE key = 'schema_version'"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.DatabaseError as exc:
        raise BackupError(f"{path.name} non è un catalogo di autoPhotoEdit ({exc})") from exc
    if row is None:
        raise BackupError(f"{path.name} non è un catalogo di autoPhotoEdit (manca la versione)")
    value = row[0]
    try:
        return int(value.strip('"') if isinstance(value, str) else value)
    except (TypeError, ValueError) as exc:
        raise BackupError(f"{path.name}: versione dello schema illeggibile ({value!r})") from exc


def backup_catalogue(destination: str | os.PathLike[str], sources: Any = None) -> dict[str, Any]:
    """Write a consistent copy of the catalogue to ``destination`` (a new file)."""
    target = assert_outside_source(Path(destination).expanduser(), sources)
    if target.exists():
        raise BackupError(f"{target} esiste già: scegli un altro nome")
    database = get_settings().db_path
    if not database.exists():
        raise BackupError("non c'è ancora un catalogo da salvare")
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.execute("VACUUM INTO ?", (str(target),))
    finally:
        connection.close()
    masks = get_settings().masks_dir
    painted = sorted(masks.glob("*.png")) if masks.is_dir() else []
    retouch = get_settings().retouch_dir
    fills = sorted(retouch.glob("*.patch")) if retouch.is_dir() else []
    return {
        "path": target,
        "bytes": target.stat().st_size,
        "masks": len(painted),
        "masks_dir": masks,
        "retouch": len(fills),
        "retouch_dir": retouch,
    }


def _running() -> bool:
    from .single_instance import _pid_alive, read_lock

    info = read_lock()
    return info is not None and _pid_alive(info.pid)


def restore_catalogue(source: str | os.PathLike[str]) -> dict[str, Any]:
    """Replace the catalogue with ``source``, keeping the current one aside."""
    from .db.session import SCHEMA_VERSION, reset_engine

    origin = Path(source).expanduser().resolve()
    if not origin.is_file():
        raise BackupError(f"{origin} non esiste")
    if _running():
        raise BackupError(
            "autoPhotoEdit è in esecuzione: chiudilo («Esci dall'applicazione») e riprova"
        )
    schema = _schema_of(origin)
    if schema > SCHEMA_VERSION:
        raise BackupError(
            f"il backup viene da una versione più recente (schema {schema}, questa arriva a "
            f"{SCHEMA_VERSION}): aggiorna autoPhotoEdit prima di ripristinarlo"
        )
    database = get_settings().db_path
    reset_engine()
    aside = None
    if database.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        aside = database.with_name(f"{database.stem}.prima-del-ripristino-{stamp}{database.suffix}")
        # The WAL holds the last writes: folding it in first makes the copy put
        # aside complete on its own.
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()
        database.rename(aside)
        for suffix in ("-wal", "-shm"):
            leftover = database.with_name(database.name + suffix)
            if leftover.exists():
                leftover.unlink()
    database.parent.mkdir(parents=True, exist_ok=True)
    assert_outside_source(database)
    shutil.copyfile(origin, database)
    return {"path": database, "aside": aside, "schema": schema}
