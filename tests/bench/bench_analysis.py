# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The analysis of phase 5 on one core, stage by stage (section 12).

Section 12 budgets the analysis -- "embedding + feature + geometria" -- at
0.6 s per photo, and the lens correction adds to the full-resolution render of
the export (3.5 s per photo). Both are measured here on the user's files, one
thread, the way a worker runs them:

* the proxy is rendered once per file, outside the timings, as the proxy job
  would have left it;
* each stage of the analysis is then timed on it;
* the lens correction is timed on the full-resolution decode, and the whole
  export render with lens correction and a rotation, against the neutral one.

Usage::

    uv run python tests/bench/bench_analysis.py [folder] [--full N]

``--full`` sets how many files get the full-resolution timings (slow).
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import cv2
import numpy as np

cv2.setNumThreads(1)


def _ms(samples: list[float]) -> str:
    return f"{1000 * statistics.median(samples):7.1f} ms mediana, {1000 * max(samples):7.1f} max"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=str(Path(__file__).parents[1] / "fixtures"))
    parser.add_argument("--full", type=int, default=3)
    args = parser.parse_args(argv[1:])

    from ape.analysis import embed, straighten
    from ape.analysis.crop import propose_crop
    from ape.analysis.scene import SceneInputs, scene_features
    from ape.export.image import ExportFormat, encode_image
    from ape.pipeline import geometry, lens
    from ape.pipeline.params import GeometryParams, neutral_params
    from ape.pipeline.render import RenderOptions, render
    from ape.raw.decode import decode_linear
    from ape.raw.proxy import editing_proxy

    raws = sorted(p for p in Path(args.folder).glob("*") if p.suffix.lower() == ".arw")
    if not raws:
        print("nessun .ARW", file=sys.stderr)
        return 1
    print(f"modello di scena: {'presente' if embed.available() else 'ASSENTE'}")

    stages: dict[str, list[float]] = {
        k: [] for k in ("lettura", "raddrizzamento", "crop", "feature", "embedding", "totale")
    }
    for raw in raws:
        proxy = render(editing_proxy(raw), neutral_params(), RenderOptions(long_edge=2048))
        data = encode_image(proxy, ExportFormat.JPEG, quality=90)
        embed.embed(proxy)  # the first call loads the model; a worker pays it once

        start = time.perf_counter()
        image = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR),
                             cv2.COLOR_BGR2RGB)
        t1 = time.perf_counter()
        level = straighten.estimate(image)
        t2 = time.perf_counter()
        frame = image
        if level.rotation_deg:
            frame = geometry.apply(image.astype(np.float32) / 255,
                                   GeometryParams(rotation_deg=level.rotation_deg))
        h, w = frame.shape[:2]
        size = (512, round(512 * h / w)) if w >= h else (round(512 * w / h), 512)
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        propose_crop(small)
        t3 = time.perf_counter()
        scene_features(image, SceneInputs(100, 2.8, 1 / 250, 50, 5500, 0))
        t4 = time.perf_counter()
        embed.embed(image)
        t5 = time.perf_counter()
        for key, value in zip(
            stages, (t1 - start, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - start), strict=True
        ):
            stages[key].append(value)

    print(f"\nanalisi su {len(raws)} proxy 2048 px, 1 core (obiettivo §12: 600 ms)")
    for key, samples in stages.items():
        print(f"  {key:<16}{_ms(samples)}")

    lens_times, neutral_times, full_times = [], [], []
    for raw in raws[: args.full]:
        decoded = decode_linear(raw)
        start = time.perf_counter()
        lens.apply(decoded.rgb, decoded.lens)
        lens_times.append(time.perf_counter() - start)
        plain = neutral_params()
        plain.geometry.lens_correction = False
        start = time.perf_counter()
        render(decoded, plain)
        neutral_times.append(time.perf_counter() - start)
        params = neutral_params()
        params.geometry.rotation_deg = 1.2
        start = time.perf_counter()
        render(decoded, params)
        full_times.append(time.perf_counter() - start)
    if lens_times:
        print(f"\nrisoluzione piena, {len(lens_times)} file, 1 core")
        print(f"  correzione lente   {_ms(lens_times)}")
        print(f"  render neutro      {_ms(neutral_times)}")
        print(f"  + lente + rotaz.   {_ms(full_times)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
