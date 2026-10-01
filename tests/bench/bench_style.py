# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The acceptance of phase 6, measured on the user's own edits (section 14).

"Given a profile trained on 30 pairs, on a validation set of 10 unseen photos
the mean ΔE2000 against the user's manual edit is < 6 and no photo exceeds 12."

This script does exactly that with the production code: pairing
(``style/pairing.py``), preparation and inversion (``sample.py``,
``learn.py``) in a pool of workers, training (``train.py``), prediction
(``model.py``), and the ΔE of the predicted rendering against the user's JPEG
on the registered area at the comparison size of section 8.2 (512 px).

One split is not a measurement, so it reports a *canonical* split (the first
permutation of seed 0, for a number that can be quoted and reproduced) and the
distribution over ``--splits`` random ones. Two ΔE figures are given:
``strict``, per pixel at 512 px, and ``colour``, after the 1 px blur the
inversion compares at (``loss.COMPARE_SIGMA``) -- the difference is detail
(output sharpening, resampling, two lens profiles), not colour.

It also measures the timings of section 12: inversion per pair (budget
10-25 s) and prediction per photo (target 5 ms).

Nothing is written near the photographs; the prepared pairs are cached, if
asked, in a directory of the caller's choosing.

Usage::

    uv run python tests/bench/bench_style.py RAW_DIR EDITED_DIR [--splits 50]
        [--train 30] [--validate 10] [--workers 14] [--cache DIR]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import numpy as np



#: Carries the parameters' version: a pickle of version 1 holds its vectors and
#: contexts in the old tint sign, and would be read back silently wrong.
CACHE_SUFFIX = ".p2.pkl"


def _work(job: tuple[str, str, str | None]) -> dict:
    import cv2

    cv2.setNumThreads(1)
    from ape.style import learn, sample

    raw, reference, cache = job
    target = Path(cache) / (Path(raw).stem + CACHE_SUFFIX) if cache else None
    if target is not None and target.is_file():
        with target.open("rb") as handle:
            return pickle.load(handle)
    started = time.perf_counter()
    prepared = sample.prepare(raw, reference)
    prepared_s = time.perf_counter() - started
    result = learn.invert(prepared.images)
    out = {
        "raw": raw,
        "reference": reference,
        "vector": result.vector,
        "delta_e": result.delta_e,
        "strict": result.delta_e_strict,
        "evaluations": result.evaluations,
        "invert_s": result.seconds,
        "prepare_s": prepared_s,
        "features": prepared.scene.features,
        "embedding": prepared.scene.embedding,
        "context": prepared.scene.context,
        "images": prepared.images,
    }
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(out, handle)
    return out


def _evaluate(pair: dict, vector: np.ndarray) -> tuple[float, float]:
    import cv2

    from ape.pipeline.render import render
    from ape.style import vector as sv
    from ape.style.colordiff import delta_e_2000, srgb_to_lab

    images = pair["images"]
    ours = render(images.decoded, sv.to_params(vector, pair["context"]))
    mask = images.mask
    strict = float(
        delta_e_2000(srgb_to_lab(ours[mask]), srgb_to_lab(images.reference[mask])).mean()
    )
    sigma = 1.0
    a = cv2.GaussianBlur(ours, (0, 0), sigma)
    b = cv2.GaussianBlur(images.reference, (0, 0), sigma)
    colour = float(delta_e_2000(srgb_to_lab(a[mask]), srgb_to_lab(b[mask])).mean())
    return strict, colour


def _split(pairs: list[dict], train_idx, test_idx) -> tuple[np.ndarray, float]:
    from ape.style.model import TrainingSample
    from ape.style.train import train

    model = train(
        [
            TrainingSample(
                int(i),
                pairs[i]["vector"],
                pairs[i]["features"],
                pairs[i]["embedding"],
                pairs[i]["context"].exposure_anchor_ev,
            )
            for i in train_idx
        ]
    )
    rows = []
    predict_s = []
    for i in test_idx:
        started = time.perf_counter()
        prediction = model.predict(
            pairs[i]["features"], pairs[i]["embedding"], pairs[i]["context"].exposure_anchor_ev
        )
        predict_s.append(time.perf_counter() - started)
        rows.append(_evaluate(pairs[i], prediction.vector))
    return np.array(rows), float(np.median(predict_s))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("edited_dir", type=Path)
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--train", type=int, default=30)
    parser.add_argument("--validate", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    import multiprocessing as mp

    from ape.style.pairing import RAW_SUFFIXES, list_references, pair_files

    raws = sorted(p for p in args.raw_dir.iterdir() if p.suffix.lower() in RAW_SUFFIXES)
    started = time.perf_counter()
    pairing = pair_files(raws, list_references(args.edited_dir))
    print(
        f"accoppiamento: {len(pairing.pairs)} coppie in {time.perf_counter() - started:.2f} s "
        f"({ {m: sum(p.method == m for p in pairing.pairs) for m in ('name', 'xmp', 'time')} }), "
        f"{len(pairing.unpaired_references)} editate non accoppiate"
    )

    jobs = [
        (str(p.raw), str(p.reference), str(args.cache) if args.cache else None)
        for p in pairing.pairs
    ]
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        pairs = pool.map(_work, jobs)
    wall = time.perf_counter() - started
    pairs.sort(key=lambda p: p["raw"])
    invert = np.array([p["invert_s"] for p in pairs])
    residual = np.array([p["delta_e"] for p in pairs])
    strict = np.array([p["strict"] for p in pairs])
    print(
        f"inversione: {len(pairs)} coppie in {wall:.0f} s con {args.workers} processi; "
        f"per coppia (1 processo) mediana {np.median(invert):.1f} s, max {invert.max():.1f} s; "
        f"render al massimo {max(p['evaluations'] for p in pairs)}"
    )
    print(
        f"residuo dell'inversione: colore media {residual.mean():.2f} max {residual.max():.2f}; "
        f"stretto media {strict.mean():.2f} max {strict.max():.2f}"
    )

    n = len(pairs)
    need = args.train + args.validate
    if n < need:
        print(f"servono almeno {need} coppie, ce ne sono {n}")
        return
    rng = np.random.default_rng(0)
    results = []
    predict_times = []
    for split in range(args.splits):
        order = rng.permutation(n)
        rows, predict_s = _split(pairs, order[: args.train], order[args.train : need])
        predict_times.append(predict_s)
        results.append(rows)
        if split == 0:
            names = [Path(pairs[i]["reference"]).name for i in order[args.train : need]]
            print(
                "\nsplit canonico (seed 0, primo): "
                f"stretto media {rows[:, 0].mean():.2f} max {rows[:, 0].max():.2f}; "
                f"colore media {rows[:, 1].mean():.2f} max {rows[:, 1].max():.2f}"
            )
            for name, (s, c) in zip(names, rows, strict=True):
                print(f"   {name:40s} stretto {s:5.2f}  colore {c:5.2f}")
    stacked = np.array(results)  # (splits, validate, 2)
    means, maxima = stacked[:, :, 0].mean(axis=1), stacked[:, :, 0].max(axis=1)
    passed = int(np.sum((means < 6.0) & (maxima <= 12.0)))
    print(
        f"\n{args.splits} split casuali "
        f"({args.train} addestramento / {args.validate} validazione), ΔE stretto:"
    )
    print(
        f"   media per split: mediana {np.median(means):.2f}, 10°-90° percentile "
        f"{np.percentile(means, 10):.2f}-{np.percentile(means, 90):.2f}; media < 6 in "
        f"{int(np.sum(means < 6))}/{args.splits}"
    )
    print(
        f"   foto peggiore per split: mediana {np.median(maxima):.2f}, max {maxima.max():.2f}; "
        f"nessuna sopra 12 in {int(np.sum(maxima <= 12))}/{args.splits}"
    )
    print(f"   entrambi i criteri: {passed}/{args.splits}")
    colour_means, colour_max = stacked[:, :, 1].mean(axis=1), stacked[:, :, 1].max(axis=1)
    print(
        f"   (a scala di colore: media {np.median(colour_means):.2f}, peggiore mediana "
        f"{np.median(colour_max):.2f}, max {colour_max.max():.2f})"
    )
    print(f"predizione per foto: mediana {1000 * np.median(predict_times):.2f} ms (target 5 ms)")


if __name__ == "__main__":
    main()
