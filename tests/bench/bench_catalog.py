# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Throughput of an import, measured rather than estimated (section 12).

Section 12 asks for a thousand photos imported and analysed in four minutes with
fourteen workers, and section 26 says to report measured numbers. This script
measures the two halves separately, because they are limited by different
things:

* **the scan** reads a mebibyte of each file and hashes it -- disk bound, one
  process, and the part that decides how long the user stares at a progress bar
  before the grid appears;
* **the proxies and the analysis** decode each RAW at half size, write a JPEG,
  then straighten, propose a crop, compute the scene features and the
  embedding on it (phase 5) -- CPU bound, one photo per worker, and the part
  the worker count actually helps.

``--borrow`` links the scene model and the updated lensfun data of the real
data home into the temporary one, so the analysis measured is the one the user
gets rather than the one of a fresh install.

It runs against a temporary catalogue in a temporary XDG home and never writes
anywhere near the photographs.

Usage::

    uv run python tests/bench/bench_catalog.py [folder] [--workers N] [--repeat K]

The folder defaults to ``tests/fixtures/``. With fewer photos than a real card,
the per-photo figures are what to read; the extrapolation to a thousand is
printed as an extrapolation and labelled as one.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

TARGET_PHOTOS = 1000
TARGET_SECONDS = 240.0


def _prepare_environment(root: Path) -> None:
    for variable in ("XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
        os.environ[variable] = str(root / variable.split("_")[1].lower())
    os.environ["XDG_RUNTIME_DIR"] = str(root / "run")
    (root / "run").mkdir(parents=True, exist_ok=True)


def _idle_footprint(pids: list[int], settle: float = 20.0, window: float = 10.0):
    """RSS and PSS in bytes of each worker, and the CPU they burn together, once idle.

    PSS is the honest figure: RSS counts in full, in every worker, the pages
    they all share with the forkserver that preloaded NumPy, OpenCV and rawpy.

    ``settle`` lets the batch's tail die out: each worker trims its heap
    (``limits.trim_heap``) and spends a few scattered tens of milliseconds for
    up to about 15 s after the last job -- measured in 3 s windows, then 0.00 in
    every worker. ``window`` is how long the idle CPU is averaged over, which
    section 26 wants at about zero. Linux only: ``None`` elsewhere.
    """
    ticks = os.sysconf("SC_CLK_TCK")

    def cpu_seconds() -> float:
        total = 0.0
        for pid in pids:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            total += (int(fields[11]) + int(fields[12])) / ticks  # utime + stime
        return total

    try:
        time.sleep(settle)
        before, at = cpu_seconds(), time.perf_counter()
        time.sleep(window)
        cpu = (cpu_seconds() - before) / (time.perf_counter() - at)
        rss, pss = [], []
        for pid in pids:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Rss:"):
                    rss.append(int(line.split()[1]) * 1024)
                elif line.startswith("Pss:"):
                    pss.append(int(line.split()[1]) * 1024)
        return rss, pss, cpu
    except (OSError, ValueError, IndexError):
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "fixtures"),
        help="cartella di RAW da importare (mai modificata)",
    )
    parser.add_argument("--workers", type=int, default=0, help="worker (default: regola di §12)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="duplica la cartella N volte in una copia temporanea, per un carico realistico",
    )
    parser.add_argument(
        "--borrow",
        action="store_true",
        help="usa il modello di scena e i dati lensfun della home reale, se presenti",
    )
    args = parser.parse_args(argv[1:])
    home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    real_data = home / "autophotoedit"

    source = Path(args.folder).expanduser()
    raws = sorted(p for p in source.glob("*") if p.suffix.lower() == ".arw")
    if not raws:
        print(f"nessun .ARW in {source}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="ape-bench-") as temporary:
        root = Path(temporary)
        _prepare_environment(root)
        if args.borrow:
            data = root / "data" / "autophotoedit"
            model = real_data / "models" / "clip-vit-b32-visual-int8.onnx"
            if model.is_file():
                (data / "models").mkdir(parents=True, exist_ok=True)
                os.symlink(model, data / "models" / model.name)
            lensfun = real_data / "lensfun" / "version_1"
            if lensfun.is_dir():
                shutil.copytree(lensfun, data / "lensfun" / "version_1")

        folder = source
        if args.repeat > 1:
            # Copies, because the importer deduplicates by content: the same
            # bytes under another name is one photo, which is the right answer
            # and the wrong benchmark. Each copy is padded by a different number
            # of bytes, so the copies differ in *size* and the quick key alone
            # separates them -- padding them all by one byte instead would leave
            # every copy with the same quick key and send the importer down the
            # full-hash path, which measures hashing rather than importing.
            folder = root / "card"
            folder.mkdir()
            for repetition in range(args.repeat):
                for raw in raws:
                    target = folder / f"{raw.stem}-{repetition:02d}{raw.suffix}"
                    shutil.copy2(raw, target)
                    with open(target, "ab") as handle:
                        handle.write(b"\0" * (repetition + 1))
            print(f"copia di lavoro: {args.repeat} x {len(raws)} file in {folder}")

        from ape.config import get_settings
        from ape.db.models import Project
        from ape.db.session import init_db, session_scope
        from ape.importer import import_folder
        from ape.jobs.handlers import enqueue_proxies
        from ape.jobs.pool import WorkerPool
        from ape.jobs.worker import default_worker_count

        settings = get_settings()
        settings.ensure_dirs()
        init_db()

        started = time.perf_counter()
        with session_scope() as session:
            project = Project(name="benchmark", source_dir=str(folder))
            session.add(project)
            session.flush()
            summary = import_folder(session, project)
            project_id = project.id
        scan_seconds = time.perf_counter() - started

        with session_scope() as session:
            queued = enqueue_proxies(session, project_id)

        workers = args.workers or default_worker_count()
        pool = WorkerPool(db_path=settings.db_path, workers=workers)
        started = time.perf_counter()
        pool.start()
        try:
            finished = pool.drain(timeout=3600.0)
            proxy_seconds = time.perf_counter() - started
            idle = _idle_footprint([p.pid for p in pool._processes])
        finally:
            pool.stop()

        count = summary.imported
        print()
        print(f"{'foto importate':<34}{count}")
        print(f"{'scansione + catalogo':<34}{scan_seconds:8.2f} s   "
              f"({1000 * scan_seconds / max(count, 1):.1f} ms/foto)")
        print(f"{'anteprime+analisi (' + str(workers) + ' w)':<34}{proxy_seconds:8.2f} s   "
              f"({1000 * proxy_seconds / max(queued, 1):.1f} ms/foto)")
        total = scan_seconds + proxy_seconds
        print(f"{'totale':<34}{total:8.2f} s")
        if not finished:
            print("ATTENZIONE: la coda non si è svuotata entro il timeout")
        if idle is not None:
            rss, pss, cpu = idle
            print(f"{'worker a riposo (RSS)':<34}{sum(rss) / 2**20:8.0f} MB totali, "
                  f"{max(rss) / 2**20:.0f} MB il più grande")
            print(f"{'worker a riposo (PSS)':<34}{sum(pss) / 2**20:8.0f} MB totali, "
                  f"{max(pss) / 2**20:.0f} MB il più grande")
            print(f"{'worker a riposo (CPU)':<34}{100 * cpu:8.2f} % di un core, tutti insieme")

        per_photo = total / max(count, 1)
        projected = per_photo * TARGET_PHOTOS
        verdict = "rispettato" if projected <= TARGET_SECONDS else "NON rispettato"
        print()
        measured = count >= TARGET_PHOTOS
        label = "riportato a" if measured else "estrapolazione a"
        print(
            f"{label} {TARGET_PHOTOS} foto: {projected:.0f} s "
            f"({projected / 60:.1f} min) — obiettivo §12 {TARGET_SECONDS:.0f} s: {verdict}"
        )
        if not measured:
            print("(estrapolazione lineare da un campione piccolo: va riverificata su "
                  f"{TARGET_PHOTOS} file, per esempio con --repeat 42)")

        with session_scope() as session:
            from ape.jobs.queue import counts_by_state

            print(f"stato dei job: {counts_by_state(session)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
