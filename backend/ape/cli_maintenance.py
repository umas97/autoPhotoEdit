# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The maintenance commands: ``diagnose``, ``backup``, ``restore`` (sections 19, 20.2).

Each prints what it did in a sentence and returns an exit code; the work is in
``diagnostics.py`` and ``backup.py``. Every file they write goes through the
guard of section 2, with every project's source folder protected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["add_maintenance_parsers"]


def _sources() -> list[str]:
    """Every project's source folder: none of them may receive a write."""
    from sqlalchemy import select

    from .db.models import Project
    from .db.session import session_scope

    with session_scope() as session:
        return [str(p) for p in session.scalars(select(Project.source_dir)).all() if p]


def _size(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n < 1024**2 else f"{n / 1024**2:.1f} MB"


def cmd_diagnose(args: argparse.Namespace) -> int:
    from .db.session import init_db, session_scope
    from .diagnostics import build, plan
    from .safety import guarded_open

    init_db()
    with session_scope() as session:
        entries = plan(session)
        print("Il file conterrà (mai RAW, mai pixel; la cartella home è scritta come ~):")
        for entry in entries:
            print(f"  {entry['name']:<28} {_size(entry['size']):>9}  {entry['description']}")
        name, payload = build(session)
    target = Path(args.output).expanduser() if args.output else Path.cwd() / name
    if target.is_dir():
        target = target / name
    with guarded_open(target, "xb", source=_sources()) as handle:
        handle.write(payload)
    print(f"diagnostica scritta in {target} ({_size(len(payload))})")
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    from .backup import BackupError, backup_catalogue
    from .db.session import init_db

    init_db()
    try:
        result = backup_catalogue(args.destination, _sources())
    except BackupError as exc:
        print(f"errore: {exc}")
        return 1
    print(f"catalogo salvato in {result['path']} ({_size(result['bytes'])})")
    if result["masks"]:
        print(
            f"nota: {result['masks']} maschere dipinte a mano sono in {result['masks_dir']}: "
            "non sono nel catalogo, copiale insieme al backup"
        )
    if result["retouch"]:
        print(
            f"nota: {result['retouch']} riempimenti della gomma magica sono in "
            f"{result['retouch_dir']}: non sono nel catalogo, copiali insieme al backup"
        )
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    from .backup import BackupError, restore_catalogue

    try:
        result = restore_catalogue(args.source)
    except BackupError as exc:
        print(f"errore: {exc}")
        return 1
    if result["aside"] is not None:
        print(f"il catalogo precedente è stato messo da parte in {result['aside']}")
    print(f"catalogo ripristinato da {args.source} (schema {result['schema']})")
    return 0


def add_maintenance_parsers(sub) -> None:
    diagnose = sub.add_parser(
        "diagnose", help="crea lo zip di diagnostica da allegare a una segnalazione"
    )
    diagnose.add_argument(
        "-o", "--output", help="file o cartella di destinazione (default: la cartella corrente)"
    )
    diagnose.set_defaults(func=cmd_diagnose)

    backup = sub.add_parser("backup", help="salva una copia del catalogo (senza cache né proxy)")
    backup.add_argument("destination", help="file .db da creare")
    backup.set_defaults(func=cmd_backup)

    restore = sub.add_parser(
        "restore", help="sostituisce il catalogo con un backup, mettendo da parte quello attuale"
    )
    restore.add_argument("source", help="file .db creato da «autophotoedit backup»")
    restore.set_defaults(func=cmd_restore)
