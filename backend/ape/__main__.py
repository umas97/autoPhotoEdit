# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line entry point.

``autophotoedit`` with no arguments starts the local server *and its window*,
which is what the desktop entry of section 18 runs. Running it a second time
does not start a second program: it raises the window that is already open
(section 21.1), or opens one on the instance that is already serving.

The other commands are the pipeline (``render``, ``params``, ``info``) and the
catalogue (``import``), the latter being the same code path the API uses, so a
project imported from a terminal and one imported from the interface are the
same project.

Every command that writes registers the RAW's own folder as a protected root
first, so the non-destructiveness guard is armed even from the CLI, where there
is no project to derive it from.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .pipeline.colorspace import OutputSpace
from .pipeline.params import EditParams, neutral_params
from .safety import SourceWriteError, register_protected_root

_log = logging.getLogger("ape")


def _load_params(path: str | None) -> EditParams:
    if path is None:
        return neutral_params()
    text = Path(path).expanduser().read_text(encoding="utf-8")
    return EditParams.from_json(text)


def _format_for(output: Path, explicit: str | None):
    from .export.image import ExportFormat

    if explicit:
        return ExportFormat(explicit)
    return (
        ExportFormat.JPEG
        if output.suffix.lower() in (".jpg", ".jpeg")
        else ExportFormat.TIFF16
    )


def cmd_render(args: argparse.Namespace) -> int:
    from .export.image import save_image
    from .pipeline.render import RenderOptions, render
    from .raw.decode import decode_linear

    source = Path(args.raw).expanduser()
    output = Path(args.output).expanduser()
    # Arm the guard before anything can write: the folder holding the RAW is
    # read-only for the rest of this process (docs/SPEC.md section 2).
    register_protected_root(source.parent)

    params = _load_params(args.params)
    started = time.perf_counter()
    decoded = decode_linear(source, quality=args.quality)
    decoded_at = time.perf_counter()

    options = RenderOptions(
        output_space=OutputSpace(args.space),
        long_edge=args.long_edge,
        apply_baseline_exposure=not args.no_baseline,
    )
    image = render(decoded, params, options)
    rendered_at = time.perf_counter()

    fmt = _format_for(output, args.format)
    written = save_image(
        image,
        output,
        fmt,
        output_space=options.output_space,
        quality=args.jpeg_quality,
        dither=args.dither,
    )
    finished = time.perf_counter()

    height, width = image.shape[:2]
    print(
        f"{written}  {width}x{height}  {fmt.value}  "
        f"decodifica {decoded_at - started:.2f}s  "
        f"sviluppo {rendered_at - decoded_at:.2f}s  "
        f"scrittura {finished - rendered_at:.2f}s"
    )
    return 0


def cmd_params(args: argparse.Namespace) -> int:
    text = neutral_params().to_json()
    if args.output:
        destination = Path(args.output).expanduser()
        destination.write_text(text + "\n", encoding="utf-8")
        print(f"parametri neutri scritti in {destination}")
    else:
        print(text)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .raw.decode import decode_linear
    from .raw.metadata import exiv2_available, read_metadata

    source = Path(args.raw).expanduser()
    register_protected_root(source.parent)

    meta = read_metadata(source)
    decoded = decode_linear(source, half_size=True)
    camera = decoded.camera

    rows = [
        ("file", source.name),
        ("fotocamera", meta.camera or "sconosciuta"),
        ("obiettivo", meta.lens_model or "sconosciuto"),
        ("ISO", meta.iso),
        ("diaframma", f"f/{meta.aperture:.1f}" if meta.aperture else None),
        ("tempo", f"{meta.shutter}s" if meta.shutter else None),
        ("focale", f"{meta.focal_length:.0f}mm" if meta.focal_length else None),
        ("scatto", meta.shot_at.isoformat(sep=" ") if meta.shot_at else None),
        ("GPS", "presente" if meta.has_gps else "assente"),
        ("dimensioni (half-size)", f"{decoded.rgb.shape[1]}x{decoded.rgb.shape[0]}"),
        ("WB as-shot", f"{camera.as_shot_temperature_k:.0f} K, tint {camera.as_shot_tint:+.0f}"),
        ("moltiplicatori WB", ", ".join(f"{m:.4f}" for m in camera.as_shot_multipliers)),
        ("esposizione di base", f"{decoded.baseline_exposure_ev:+.2f} EV"),
    ]
    if not exiv2_available():
        rows.append(("nota", "exiv2 non disponibile: metadati EXIF non letti"))

    width = max(len(str(label)) for label, _ in rows)
    for label, value in rows:
        if value is not None:
            print(f"{label:<{width}}  {value}")

    if args.json:
        payload = {
            "camera": meta.camera,
            "lens": meta.lens_model,
            "iso": meta.iso,
            "aperture": meta.aperture,
            "shutter": meta.shutter,
            "focal_length": meta.focal_length,
            "shot_at": meta.shot_at.isoformat() if meta.shot_at else None,
            "has_gps": meta.has_gps,
            "as_shot_temperature_k": camera.as_shot_temperature_k,
            "as_shot_tint": camera.as_shot_tint,
            "baseline_exposure_ev": decoded.baseline_exposure_ev,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the local server and its window. The work is in ``runtime.py``."""
    from .runtime import run_server

    return run_server(args)


def cmd_import(args: argparse.Namespace) -> int:
    """Create or reuse a project and import a folder into it.

    Re-running this on the same folder is the point: it must add only what is
    new and leave every edit alone (test 12 of section 13).
    """
    from sqlalchemy import select

    from .db.models import Project
    from .db.session import init_db, session_scope
    from .importer import import_folder
    from .jobs.handlers import enqueue_proxies

    folder = Path(args.folder).expanduser()
    register_protected_root(folder)
    init_db()

    with session_scope() as session:
        name = args.name or folder.name
        project = session.scalars(select(Project).where(Project.name == name)).first()
        if project is None:
            project = Project(name=name, source_dir=str(folder.resolve()))
            session.add(project)
            session.flush()
            print(f"progetto «{name}» creato")
        summary = import_folder(session, project, folder)
        queued = enqueue_proxies(session, project.id) if not args.no_proxies else 0
        project_id = project.id

    print(summary.describe())
    if queued:
        print(f"{queued} anteprime in coda")

    if args.wait and queued:
        from .config import get_settings
        from .jobs.pool import WorkerPool

        pool = WorkerPool(db_path=get_settings().db_path, workers=args.workers)
        started = time.perf_counter()
        pool.start()
        try:
            finished = pool.drain(timeout=args.timeout)
        finally:
            pool.stop()
        elapsed = time.perf_counter() - started
        print(
            f"anteprime completate in {elapsed:.1f}s con {pool.workers} worker"
            if finished
            else f"tempo scaduto dopo {elapsed:.1f}s: la coda riprende al prossimo avvio"
        )
    print(f"progetto {project_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autophotoedit",
        description="Post-produzione automatica di RAW Sony, in locale e non distruttiva.",
    )
    parser.add_argument("--version", action="version", version=f"autoPhotoEdit {__version__}")
    parser.add_argument(
        "--log-level",
        default=None,
        help="quanto scrivere nel terminale: DEBUG, INFO, WARNING, ERROR (default: WARNING); "
        "il file di log segue APE_LOG_LEVEL",
    )
    sub = parser.add_subparsers(dest="command")

    render_parser = sub.add_parser("render", help="sviluppa un RAW e salva l'immagine")
    render_parser.add_argument("raw", help="file .ARW da sviluppare (mai modificato)")
    render_parser.add_argument("-o", "--output", required=True, help="file di destinazione")
    render_parser.add_argument("-p", "--params", help="file JSON di EditParams")
    render_parser.add_argument(
        "-f", "--format", choices=["jpeg", "tiff8", "tiff16"], help="formato di uscita"
    )
    render_parser.add_argument(
        "-s",
        "--space",
        default="srgb",
        choices=[s.value for s in OutputSpace],
        help="spazio colore di uscita (default: srgb)",
    )
    render_parser.add_argument(
        "--long-edge", type=int, help="ridimensiona il lato lungo prima della nitidezza"
    )
    render_parser.add_argument(
        "--jpeg-quality", type=int, default=92, help="qualità JPEG 1-100 (default: 92)"
    )
    render_parser.add_argument(
        "--quality", action="store_true", help="demosaicing DCB anziché AHD (più lento)"
    )
    render_parser.add_argument(
        "--dither", action="store_true", help="dithering ordinato in quantizzazione"
    )
    render_parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="non applicare l'ancoraggio di esposizione ISO 12232",
    )
    render_parser.set_defaults(func=cmd_render)

    params_parser = sub.add_parser("params", help="stampa o scrive EditParams neutri")
    params_parser.add_argument("-o", "--output", help="file JSON da scrivere")
    params_parser.set_defaults(func=cmd_params)

    info_parser = sub.add_parser("info", help="mostra i metadati letti da un RAW")
    info_parser.add_argument("raw", help="file .ARW da leggere")
    info_parser.add_argument("--json", action="store_true", help="stampa anche il JSON")
    info_parser.set_defaults(func=cmd_info)

    serve_parser = sub.add_parser("serve", help="avvia il server locale")
    _add_serve_arguments(serve_parser)
    serve_parser.set_defaults(func=cmd_serve)

    import_parser = sub.add_parser("import", help="importa una cartella in un progetto")
    import_parser.add_argument("folder", help="cartella da importare (mai modificata)")
    import_parser.add_argument("-n", "--name", help="nome del progetto (default: nome cartella)")
    import_parser.add_argument(
        "--no-proxies", action="store_true", help="non generare le anteprime"
    )
    import_parser.add_argument(
        "--wait", action="store_true", help="attendi che le anteprime siano pronte"
    )
    import_parser.add_argument(
        "--workers", type=int, default=0, help="numero di worker (default: la regola di §12)"
    )
    import_parser.add_argument(
        "--timeout", type=float, default=1800.0, help="secondi massimi di attesa (default: 1800)"
    )
    import_parser.set_defaults(func=cmd_import)

    from .cli_export import add_export_parser

    add_export_parser(sub)

    from .cli_maintenance import add_maintenance_parsers

    add_maintenance_parsers(sub)

    # The bare command is the desktop entry of section 18: it starts the server.
    _add_serve_arguments(parser)
    return parser


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", type=int, help="porta del server (default: 8787)")
    parser.add_argument(
        "--workers", type=int, default=0, help="numero di worker (default: la regola di §12)"
    )
    parser.add_argument(
        "--no-workers", action="store_true", help="avvia il server senza il pool di worker"
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="apri la finestra dedicata anche senza un display rilevato",
    )
    parser.add_argument(
        "--no-window", action="store_true", help="avvia il solo server senza aprire nulla"
    )
    parser.add_argument(
        "--browser",
        help="browser da usare per la finestra dedicata (anche da APE_BROWSER)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level or "WARNING").upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not getattr(args, "func", None):
        # No subcommand: start the server, which is what the icon does.
        args.func = cmd_serve
    try:
        return int(args.func(args))
    except SourceWriteError as exc:
        print(f"errore: {exc}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        print(f"errore: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
