# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The diagnostics bundle (section 19): what to attach to an issue.

A zip of plain text: the log files, the versions of Python, of the packages and
of the native libraries, the shape and the counts of the catalogue, and the
EXIF of the failed photos with their errors. **Never a RAW, never a pixel,
never the contents of a photo** -- nothing here reads a photo file at all: the
EXIF comes from the columns the import filled. Project names, the artist and
the copyright are left out; every path has the home directory as ``~``.

:func:`plan` lists the files with what each contains and its size, *before*
anything is written: the interface shows it, so that the user sees what is
about to go into a public issue. :func:`build` makes the zip from the same
contents, in memory; where it goes -- a download, or a file chosen on the
command line -- is the caller's business.
"""

from __future__ import annotations

import io
import json
import os
import platform
import sys
import zipfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from sqlalchemy import func, inspect, select, table
from sqlalchemy.orm import Session

from . import __version__
from .config import get_settings
from .db.models import Job, JobState, Photo, PhotoStatus
from .logs import log_files
from .problems import STAGES, anonymise, plain

__all__ = ["Entry", "build", "plan"]

#: The packages whose versions say something about a bug; the rest of the
#: environment is `uv.lock`.
_PACKAGES = (
    "numpy",
    "scipy",
    "opencv-python-headless",
    "rawpy",
    "pyexiv2",
    "lensfunpy",
    "onnxruntime",
    "colour-science",
    "pillow",
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "pydantic",
)

#: The photo columns that come from EXIF. Not the path, not the hash.
_EXIF = (
    "camera",
    "lens",
    "iso",
    "aperture",
    "shutter",
    "focal_length",
    "shot_at",
    "width",
    "height",
    "orientation",
)


class Entry(dict):
    """One file of the bundle: ``name``, ``description``, ``size``, ``content``."""


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n"


def _os_release() -> str | None:
    try:
        text = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def _memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None
    return round(pages / 1024**3, 1)


def _native() -> dict[str, Any]:
    """The C libraries under the wheels, as they report themselves."""
    found: dict[str, Any] = {}
    try:
        import rawpy

        found["libraw"] = ".".join(map(str, rawpy.libraw_version))
    except Exception as exc:  # noqa: BLE001 - a missing library is itself the fact
        found["libraw"] = f"non disponibile ({type(exc).__name__})"
    # pyexiv2 is GPL and stays inside raw/metadata.py (section 24).
    from .raw.metadata import exiv2_version

    found["exiv2"] = exiv2_version() or "non disponibile"
    try:
        import lensfunpy

        found["lensfun"] = ".".join(map(str, lensfunpy.lensfun_version))
        from .lensdb import data_status

        found["lensfun_data"] = data_status()
    except Exception as exc:  # noqa: BLE001
        found["lensfun"] = f"non disponibile ({type(exc).__name__})"
    return found


def _versions() -> dict[str, Any]:
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    from .models_registry import MODELS, model_path

    return {
        "autophotoedit": __version__,
        "python": sys.version.split()[0],
        "sistema": _os_release(),
        "kernel": platform.release(),
        "architettura": platform.machine(),
        "cpu": os.cpu_count(),
        "memoria_gb": _memory_gb(),
        "pacchetti": packages,
        "librerie": _native(),
        "modelli": {entry.name: model_path(entry).is_file() for entry in MODELS if entry.sha256},
    }


def _catalogue(session: Session) -> dict[str, Any]:
    from .db.session import SCHEMA_VERSION, read_setting

    tables = {
        name: session.execute(select(func.count()).select_from(table(name))).scalar_one()
        for name in sorted(inspect(session.get_bind()).get_table_names())
    }
    photos = dict(
        (status.value if hasattr(status, "value") else str(status), n)
        for status, n in session.execute(
            select(Photo.status, func.count(Photo.id)).group_by(Photo.status)
        )
    )
    jobs: dict[str, dict[str, int]] = {}
    for kind, state, n in session.execute(
        select(Job.kind, Job.state, func.count(Job.id)).group_by(Job.kind, Job.state)
    ):
        jobs.setdefault(kind.value, {})[state.value] = int(n)
    db_path = get_settings().db_path
    return {
        "schema": read_setting(session, "schema_version"),
        "schema_di_questa_build": SCHEMA_VERSION,
        "dimensione_mb": round(db_path.stat().st_size / 1024**2, 2) if db_path.exists() else None,
        "righe_per_tabella": tables,
        "foto_per_stato": photos,
        "job_per_tipo_e_stato": jobs,
    }


def _failed(session: Session) -> list[dict[str, Any]]:
    """The EXIF, error and traceback of every failed photo."""
    rows = []
    photos = session.scalars(select(Photo).where(Photo.status == PhotoStatus.FAILED)).all()
    jobs = session.scalars(
        select(Job).where(Job.state == JobState.FAILED).order_by(Job.id.desc())
    ).all()
    last = {}
    for job in jobs:
        photo_id = (job.payload or {}).get("photo_id")
        if isinstance(photo_id, int):
            last.setdefault(photo_id, job)
    for photo in photos:
        job = last.get(photo.id)
        rows.append(
            {
                "file": photo.filename,
                "exif": {column: getattr(photo, column) for column in _EXIF},
                "errore": plain(photo.error),
                "fase": STAGES.get(job.kind) if job else None,
                "dettagli": anonymise(job.traceback or job.error) if job else None,
            }
        )
    return rows


def _settings() -> dict[str, Any]:
    settings = get_settings()
    return {
        "cache_max_gb": settings.cache_max_gb,
        "proxy_long_edge": settings.proxy_long_edge,
        "export_concurrency": settings.export_concurrency,
        "log_level": settings.log_level,
        "cartella_dati": anonymise(str(settings.data_dir)),
        "cartella_stato": anonymise(str(settings.state_dir)),
    }


def plan(session: Session) -> list[Entry]:
    """Every file the bundle will contain, with its content, before anything is written."""
    entries = [
        Entry(
            name="versioni.json",
            description="Versioni di autoPhotoEdit, Python, pacchetti e librerie di sistema; "
            "sistema operativo, processori e memoria; quali modelli sono installati.",
            content=_json(_versions()),
        ),
        Entry(
            name="catalogo.json",
            description="Schema del catalogo e conteggi: righe per tabella, foto per stato, "
            "job per tipo e stato. Nessun nome di progetto, nessun percorso.",
            content=_json(_catalogue(session)),
        ),
        Entry(
            name="foto_fallite.json",
            description="Per ogni foto fallita: nome del file, dati EXIF (fotocamera, obiettivo, "
            "esposizione, dimensioni), errore e dettagli tecnici. Mai il file né i suoi pixel.",
            content=_json(_failed(session)),
        ),
        Entry(
            name="impostazioni.json",
            description="Impostazioni tecniche (cache, proxy, export, livello dei log). "
            "Non autore né copyright.",
            content=_json(_settings()),
        ),
    ]
    for path in log_files():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        entries.append(
            Entry(
                name=f"log/{path.name}",
                description="Registro del programma (la cartella home è scritta come ~).",
                content=anonymise(text) or "",
            )
        )
    for entry in entries:
        entry["size"] = len(entry["content"].encode("utf-8"))
    return entries


def build(session: Session) -> tuple[str, bytes]:
    """The bundle as ``(file name, zip bytes)``: exactly what :func:`plan` lists."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in plan(session):
            archive.writestr(entry["name"], entry["content"])
    return f"autophotoedit-diagnostica-{stamp}.zip", buffer.getvalue()
