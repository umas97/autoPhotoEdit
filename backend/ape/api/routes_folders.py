# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Browsing folders, for choosing a project's source (section 10, screen 1).

A browser cannot hand a server a path -- only copies of files -- but this
server runs on the same machine as the browser, so it can list the folders
itself and let the interface walk them. ``GET /api/folders`` returns the
sub-folders of one folder and how many RAWs it holds, counted the way the
import counts them (non-recursive, section 15), plus the places a photographer
starts from: home, the pictures folder, and the memory cards and disks mounted
under ``/media`` and ``/run/media``.

Read-only by construction: this module opens directories to list them and
never anything else. Hidden folders are left out unless asked for.
"""

from __future__ import annotations

import getpass
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, status

from ..importer import RAW_EXTENSION

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["folders"])

#: Most sub-folders returned for one folder. More is not a folder anyone
#: browses to find a card of photos; the answer says it was cut.
MAX_FOLDERS = 1000


def _pictures_dir(home: Path) -> Path | None:
    """``XDG_PICTURES_DIR`` from ``user-dirs.dirs``: "Immagini" on an Italian desktop."""
    config = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    try:
        text = (config / "user-dirs.dirs").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^XDG_PICTURES_DIR="([^"]*)"', text, flags=re.MULTILINE)
    if not match:
        return None
    path = Path(match.group(1).replace("$HOME", str(home)))
    return path if path.is_dir() and path != home else None


def _volumes() -> list[Path]:
    """Mounted cards and disks, where the desktop mounts them for this user."""
    user = getpass.getuser()
    found: list[Path] = []
    for base in (Path("/media") / user, Path("/run/media") / user):
        try:
            found += sorted(p for p in base.iterdir() if p.is_dir())
        except OSError:
            continue
    return found


def _places() -> list[dict]:
    home = Path.home()
    places = [{"kind": "home", "name": home.name, "path": str(home)}]
    pictures = _pictures_dir(home)
    if pictures is not None:
        places.append({"kind": "pictures", "name": pictures.name, "path": str(pictures)})
    places += [{"kind": "volume", "name": p.name, "path": str(p)} for p in _volumes()]
    places.append({"kind": "root", "name": "/", "path": "/"})
    return places


@router.get("/folders")
def list_folders(
    path: str | None = None, hidden: bool = Query(default=False)
) -> dict:
    """The sub-folders of ``path`` (home when omitted), and its RAW count.

    Raises:
        HTTPException 404: ``path`` is not a folder.
        HTTPException 403: the folder cannot be read.
    """
    folder = Path(path).expanduser() if path else Path.home()
    if not folder.is_absolute():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "serve un percorso assoluto")
    folder = folder.resolve()
    if not folder.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"la cartella {folder} non esiste")

    folders: list[dict] = []
    raws = 0
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if entry.is_dir():
                        if hidden or not entry.name.startswith("."):
                            folders.append({"name": entry.name, "path": str(folder / entry.name)})
                    elif entry.name.lower().endswith(RAW_EXTENSION) and entry.is_file():
                        raws += 1
                except OSError:
                    continue  # a broken link, a vanished entry: not worth failing the list
    except PermissionError as exc:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, f"non hai i permessi per leggere {folder}"
        ) from exc
    except OSError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{folder} non si legge: {exc}") from exc

    folders.sort(key=lambda f: (f["name"].casefold(), f["name"]))
    return {
        "path": str(folder),
        "parent": str(folder.parent) if folder.parent != folder else None,
        "folders": folders[:MAX_FOLDERS],
        "truncated": len(folders) > MAX_FOLDERS,
        "raw_count": raws,
        "places": _places(),
    }
