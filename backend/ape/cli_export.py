# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``autophotoedit export``: the batch of the Export screen, from a terminal.

Its own module because ``__main__.py`` holds every other command and section 26
caps a file at about four hundred lines.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .pipeline.colorspace import OutputSpace
from .safety import register_protected_root

__all__ = ["add_export_parser", "cmd_export"]


def add_export_parser(sub) -> None:
    parser = sub.add_parser("export", help="esporta le foto di un progetto")
    parser.add_argument("project", help="id o nome del progetto")
    parser.add_argument("-o", "--output", required=True, help="cartella di destinazione")
    parser.add_argument("-f", "--format", choices=["jpeg", "tiff8", "tiff16"])
    parser.add_argument("-q", "--quality", type=int, help="qualità JPEG 1-100")
    parser.add_argument("-s", "--space", choices=[s.value for s in OutputSpace])
    parser.add_argument("--long-edge", type=int, help="lato lungo in pixel")
    parser.add_argument(
        "--sharpening", choices=["none", "low", "standard", "high"], help="nitidezza di output"
    )
    parser.add_argument("-t", "--template", help="modello del nome, es. {basename}.{ext}")
    parser.add_argument(
        "--on-conflict",
        default="rename",
        choices=["rename", "overwrite", "skip"],
        help="file già presenti: rinomina (default), sovrascrivi, salta",
    )
    parser.add_argument(
        "--strip-gps", action="store_true", help="rimuovi i dati di posizione"
    )
    parser.add_argument("--xmp", help="sidecar da scrivere: darktable,adobe")
    parser.add_argument(
        "--xmp-only", action="store_true", help="scrivi solo i sidecar, nessuna immagine"
    )
    parser.add_argument(
        "--only-approved", action="store_true", help="solo le foto approvate in revisione"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="numero di worker (default: la regola di §12)"
    )
    parser.add_argument(
        "--timeout", type=float, default=7200.0, help="secondi massimi di attesa"
    )
    parser.set_defaults(func=cmd_export)



def cmd_export(args: argparse.Namespace) -> int:
    """Export a project's photos, the same batch the Export screen starts.

    Collisions default to ``rename`` (section 16.2): a terminal has no dialog
    to ask in, and renaming is the one choice that cannot lose a file.
    """
    from sqlalchemy import select

    from .config import get_settings
    from .db.enums import ExportConflict
    from .db.models import Project
    from .db.session import init_db, session_scope
    from .export import batch as batches
    from .export import service
    from .export.settings import ExportSettings, load_settings
    from .jobs.pool import WorkerPool

    init_db()
    with session_scope() as session:
        key = args.project
        project = session.get(Project, int(key)) if key.isdigit() else None
        if project is None:
            project = session.scalars(select(Project).where(Project.name == key)).first()
        if project is None:
            raise ValueError(f"progetto «{key}» inesistente")
        register_protected_root(project.source_dir)
        changes = {"output_dir": str(Path(args.output).expanduser().resolve())}
        for field in ("format", "quality", "long_edge", "template", "sharpening"):
            value = getattr(args, field)
            if value is not None:
                changes[field] = value
        if args.space:
            changes["output_space"] = args.space
        changes["on_conflict"] = args.on_conflict
        changes["strip_gps"] = bool(args.strip_gps)
        changes["only_approved"] = bool(args.only_approved)
        kinds = {k.strip() for k in (args.xmp or "").split(",") if k.strip()}
        unknown = kinds - {"darktable", "adobe"}
        if unknown:
            raise ValueError(f"sidecar sconosciuti: {', '.join(sorted(unknown))}")
        changes["xmp_darktable"] = "darktable" in kinds
        changes["xmp_adobe"] = "adobe" in kinds
        changes["images"] = not args.xmp_only
        settings = ExportSettings.model_validate(
            load_settings(project).model_dump(mode="json") | changes
        )
        batch = service.start(
            session, project, settings, apply_to_all=ExportConflict(args.on_conflict)
        )
        batch_id, total = batch.id, batch.total
    print(f"export {batch_id}: {total} foto in coda verso {changes['output_dir']}")

    pool = WorkerPool(db_path=get_settings().db_path, workers=args.workers)
    started = time.perf_counter()
    pool.start()
    try:
        finished = pool.drain(timeout=args.timeout)
    finally:
        pool.stop()
    elapsed = time.perf_counter() - started
    with session_scope() as session:
        from .db.models import ExportBatch

        batch = session.get(ExportBatch, batch_id)
        summary = batches.batch_summary(session, batch, with_items=True)
    counts = summary["counts"]
    print(
        f"{counts['done']} esportate, {counts['skipped']} saltate, {counts['failed']} non "
        f"riuscite in {elapsed:.1f}s ({elapsed / max(1, total):.2f}s a foto, "
        f"{pool.workers} worker, al massimo {pool.export_limit} export insieme)"
    )
    for problem in summary.get("problems", []):
        if problem["state"] == "failed":
            print(f"  {problem['filename']}: {problem['error']}")
    if not finished:
        print("tempo scaduto: l'export riprende al prossimo avvio")
    return 0 if counts["failed"] == 0 and finished else 2
