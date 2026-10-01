# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The merge targets of section 12, measured on real 24 MP decodes.

* HDR of 3 frames at 24 MP: at most 25 s on 2 cores;
* focus stack of 8 frames: at most 60 s;
* panorama of 6 frames: at most 180 s and 3 GB of memory.

Every case runs in its own process, pinned like a worker (one thread for
OpenMP and BLAS, two for OpenCV during the merge, ``merge/rebuild.py``), and
reports its wall time and peak RSS. The frames are real: every member is a
full decode of a fixture ARW, through the program's own decoder. What is not
real is the scene's variety -- an HDR and a stack of one file repeated, a
panorama of pans cut out of one frame (``tests/pano_synth.py``) -- which
changes what the merge finds, not what it costs. The time spent cutting the
panorama's pans is measured and subtracted.

Usage::

    uv run python tests/bench/bench_merge.py [hdr|stack|pano|all] [--raw 16]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
TARGETS = {"hdr": (3, 25.0), "stack": (8, 60.0), "pano": (6, 180.0)}


def _case(kind: str, raw: Path, queue) -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    home = Path(tempfile.mkdtemp(prefix="ape-bench-"))
    os.environ["XDG_DATA_HOME"] = str(home / "data")
    os.environ["XDG_STATE_HOME"] = str(home / "state")
    sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "tests")]

    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    from ape.merge import engine
    from ape.merge.engine import Member, Recipe
    from ape.merge.rebuild import run_recipe

    count = TARGETS[kind][0]
    folder = home / "card"
    folder.mkdir()
    members = []
    for index in range(count):
        link = folder / f"DSC{index:05d}.ARW"
        link.symlink_to(raw)
        members.append(Member(
            photo_id=index, path=str(link), filename=link.name, hash=f"bench-{kind}-{index}",
            ev_offset=[-2.0, 0.0, 2.0][index] if kind == "hdr" else None,
            reference=index == count // 2 if kind != "hdr" else index == 1,
        ))
    options = {"output_scale": 1.0} if kind == "pano" else {}
    recipe = Recipe(group_id=1, kind={"hdr": "hdr", "stack": "focus_stack",
                                      "pano": "panorama"}[kind],
                    members=tuple(members), options=options)

    synthetic = [0.0]
    if kind == "pano":
        from pano_synth import rotation

        real = engine.decode_linear

        def decode(path, **kwargs):
            decoded = real(path, **kwargs)
            start = time.perf_counter()
            index = int(Path(path).stem[3:])
            height, width = decoded.rgb.shape[:2]
            # The frame is the world (143 degrees across), seen by a 58 degree
            # camera turning 16 degrees a shot: six 24 MP frames, 138 degrees.
            f_world, f_shot = width / 3.0, width * 0.9
            k_world = np.array([[f_world, 0, width / 2], [0, f_world, height / 2], [0, 0, 1]])
            k_shot = np.array([[f_shot, 0, width / 2], [0, f_shot, height / 2], [0, 0, 1]])
            homography = k_world @ rotation(-40 + 16 * index, 0.3, 0.2) @ np.linalg.inv(k_shot)
            decoded.rgb = cv2.warpPerspective(
                decoded.rgb, homography, (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT,
            )
            synthetic[0] += time.perf_counter() - start
            return decoded

        engine.decode_linear = decode

    # Peak RSS counts the pages of the memory-mapped frames too, which are
    # file-backed and the kernel may drop; the anonymous memory is what the
    # worker budget of section 26 is about. Sampled, since the kernel keeps
    # no high-water mark of it.
    import threading

    anon = [0]
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("RssAnon:"):
                    anon[0] = max(anon[0], int(line.split()[1]))
            done.wait(0.02)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    path, report, size = run_recipe(recipe)
    elapsed = time.perf_counter() - start - synthetic[0]
    done.set()
    sampler.join()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    queue.put({"kind": kind, "seconds": round(elapsed, 1), "synthetic_s": round(synthetic[0], 1),
               "peak_mb": round(peak), "anon_peak_mb": round(anon[0] / 1024),
               "size": size, "file_mb": round(path.stat().st_size / 1e6),
               "ram_estimate_mb": report.get("ram_estimate_mb")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="all", choices=["hdr", "stack", "pano", "all"])
    parser.add_argument("--raw", type=int, default=16, help="indice del file ARW in tests/fixtures")
    args = parser.parse_args()
    raw = sorted(FIXTURES.glob("*.ARW"))[args.raw]
    kinds = list(TARGETS) if args.which == "all" else [args.which]
    context = mp.get_context("spawn")
    for kind in kinds:
        queue = context.Queue()
        process = context.Process(target=_case, args=(kind, raw, queue))
        process.start()
        result = queue.get()
        process.join()
        frames, target = TARGETS[kind]
        verdict = "✓" if result["seconds"] <= target else "✗"
        print(f"{kind:5s} {frames} scatti: {result['seconds']:6.1f} s (target {target:.0f} s) "
              f"{verdict}  picco RSS {result['peak_mb']} MB, anonima {result['anon_peak_mb']} MB")


if __name__ == "__main__":
    main()
