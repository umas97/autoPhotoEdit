# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Culling throughput, measured rather than estimated (sections 7.7 and 12).

Section 7.7: "Su 2000 file, con i soli criteri di default, su 14 worker:
analisi di cernita ≤ 3 minuti". Section 12 adds two per-photo budgets on one
core: embedded-preview extraction ≤ 60 ms, technical scoring ≤ 80 ms.

Two thousand real ARW files are 48 GB, and the fixtures folder holds 24. So the
catalogue gets two thousand rows pointing at those 24 files in turn, each with
its own identity -- the importer would, correctly, deduplicate copies by
content, and the thing measured here is the analysis, not the import. The
queue, the worker pool, the handler, the catalogue writes and the grouping are
all the real ones.

One caveat, stated rather than hidden: after the first pass the 24 files are in
the page cache, so the run measures CPU, not a cold card reader. The culling
job reads the file's first few hundred kilobytes -- the header and the embedded
JPEG -- so on an SSD the difference is small; on a USB card reader it is not,
and that is a property of the card reader.

Usage::

    uv run python tests/bench/bench_culling.py [--photos 2000] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

TARGET_SECONDS = 180.0
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _prepare_environment(root: Path) -> None:
    for variable in ("XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        os.environ[variable] = str(root / variable.split("_")[1].lower())
    os.environ["XDG_RUNTIME_DIR"] = str(root / "run")
    (root / "run").mkdir(parents=True, exist_ok=True)


def _single_core(raws: list[Path]) -> None:
    """The two budgets of section 12, one photo at a time, one thread."""
    import cv2

    cv2.setNumThreads(1)
    from ape.culling.technical import technical_scores
    from ape.raw.embedded import read_embedded_preview

    extraction, scoring = [], []
    for _ in range(2):  # the first pass warms the page cache and the imports
        extraction.clear()
        scoring.clear()
        for raw in raws:
            started = time.perf_counter()
            preview = read_embedded_preview(raw)
            extraction.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            technical_scores(preview.image)
            scoring.append((time.perf_counter() - started) * 1000)
    for label, values, target in (
        ("estrazione anteprima", extraction, 60),
        ("punteggio tecnico", scoring, 80),
    ):
        median = statistics.median(values)
        worst = max(values)
        verdict = "rispettato" if median <= target else "NON rispettato"
        print(f"{label:<24}mediana {median:5.1f} ms  max {worst:5.1f} ms  "
              f"(obiettivo §12 {target} ms: {verdict})")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv[1:])

    raws = sorted(p for p in FIXTURES.glob("*") if p.suffix.lower() == ".arw")
    if not raws:
        print(f"nessun .ARW in {FIXTURES}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="ape-bench-cull-") as temporary:
        _prepare_environment(Path(temporary))

        from sqlalchemy import insert

        from ape.config import get_settings
        from ape.culling import service
        from ape.db.models import Photo, PhotoKind, Project
        from ape.db.session import init_db, session_scope
        from ape.jobs.pool import WorkerPool
        from ape.jobs.worker import default_worker_count

        _single_core(raws)

        settings = get_settings()
        settings.ensure_dirs()
        init_db()
        with session_scope() as session:
            project = Project(name="benchmark", source_dir=str(FIXTURES))
            session.add(project)
            session.flush()
            session.execute(
                insert(Photo),
                [
                    {
                        "project_id": project.id,
                        "path": str(raws[i % len(raws)]),
                        "filename": f"{raws[i % len(raws)].stem}-{i:04d}.ARW",
                        "hash": f"q:bench:{i}",
                        "kind": PhotoKind.RAW,
                    }
                    for i in range(args.photos)
                ],
            )
            queued = service.enqueue_culling(session, project)
            project_id = project.id

        workers = args.workers or default_worker_count()
        pool = WorkerPool(db_path=settings.db_path, workers=workers)
        started = time.perf_counter()
        pool.start()
        try:
            finished = pool.drain(timeout=3600.0)
        finally:
            pool.stop()
        analysis = time.perf_counter() - started

        with session_scope() as session:
            project = session.get(Project, project_id)
            started = time.perf_counter()
            service.ensure_grouped(session, project)
            state = service.apply_selection(session, project)
            grouping = time.perf_counter() - started
            counts = service.state(session, project)

        total = analysis + grouping
        print()
        analysed, failed = counts['analysed'], counts['failed']
        print(f"{'foto analizzate':<34}{analysed} di {queued} ({failed} fallite)")
        print(f"{'analisi (' + str(workers) + ' worker)':<34}{analysis:8.1f} s   "
              f"({1000 * analysis / max(queued, 1):.1f} ms/foto di tempo reale)")
        print(f"{'raggruppamento + selezione':<34}{grouping * 1000:8.0f} ms")
        verdict = "rispettato" if total <= TARGET_SECONDS * args.photos / 2000 else "NON rispettato"
        print(f"{'totale':<34}{total:8.1f} s   (obiettivo §7.7 per 2000 foto: "
              f"{TARGET_SECONDS:.0f} s: {verdict})")
        print(f"{'selezionate (conservativa)':<34}{state.selected}")
        if not finished:
            print("ATTENZIONE: la coda non si è svuotata entro il timeout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
