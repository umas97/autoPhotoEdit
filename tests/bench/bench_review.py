# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The acceptance of phase 7, measured on the user's own event (section 14).

"On a project of 500 photos, less than 20% needs individual review and the
whole validation closes in under 15 minutes of the user's work."

The user's folder of one event is the project: every RAW is described exactly
as the program describes a project photo -- proxy, scene features, embedding,
exposure anchor, the measurements of ``review/measure.py``, straightening, lens
-- and predicted by a profile learned from the user's delivered edits, then
scored by the confidence of section 9.1. A photo whose own edit is among the
pairs is predicted **leave-one-out** (by a profile trained on the other pairs),
so that no photo is judged by a model that has seen it; the others by the
profile of all the pairs. Coherence is not applied: it moves photos towards
their scene, which only narrows what is measured here.

On the paired photos there is a ground truth, the user's own JPEG, so the
script also says whether the confidence is right: how far the predictions it
escalates are from the user's edit, against the ones it lets through.

It reports: the fraction escalated and why; the scenes (the clustering of
section 9.2) and their sizes; and the minutes of review work under a stated
model of how long a scene and a queued photo take, which is an estimate --
the only honest measure of a person's time is a person.

Usage::

    uv run python tests/bench/bench_review.py RAW_DIR EDITED_DIR [--cache DIR]
        [--also DIR ...] [--workers 14] [--scene-seconds 12] [--photo-seconds 8]
        [--grid-seconds 3]

``--also`` adds the RAWs of other folders to the project, unpaired: another
shoot of the same photographer, judged by the profile of all the pairs.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from bench_style import _evaluate, _work  # noqa: E402


def _describe(job: tuple[str, str]) -> dict:
    """What the ``proxy``, ``analyze`` and ``predict`` jobs store for one photo."""
    import cv2

    cv2.setNumThreads(1)
    # The proxies go to a temporary data directory, which borrows the user's
    # scene model and lensfun data by symbolic link: nothing is written to the
    # user's own cache, and the photos are described as the program would.
    os.environ["XDG_DATA_HOME"] = job[1]
    from PIL import Image

    from ape.analysis import embed, straighten
    from ape.analysis.scene import SceneInputs, scene_features
    from ape.config import get_settings
    from ape.lensdb import LensIdentity, resolve
    from ape.raw.metadata import read_metadata, read_optics
    from ape.raw.proxy import build_proxy
    from ape.style.predict import _measure

    get_settings.cache_clear()
    raw = Path(job[0])
    result = build_proxy(raw, raw.stem)
    with Image.open(result.path) as image:
        pixels = np.asarray(image.convert("RGB"))
    meta = read_metadata(raw)
    optics = read_optics(raw)
    as_shot = result.as_shot or {"temperature_k": 5500.0, "tint": 0.0}
    exif = SceneInputs(
        iso=meta.iso, aperture=meta.aperture, shutter=meta.shutter,
        focal_length=meta.focal_length,
        as_shot_temperature_k=as_shot["temperature_k"], as_shot_tint=as_shot["tint"],
    )
    lens = resolve(
        LensIdentity(
            optics.camera_make, optics.camera_model, optics.lens_model,
            optics.focal_length, optics.aperture,
        )
    )
    return {
        "raw": str(raw),
        "shot_at": meta.shot_at,
        "features": scene_features(pixels, exif),
        "embedding": embed.embed(pixels),
        "measured": _measure(result.path, as_shot),
        "as_shot": as_shot,
        "straighten": straighten.estimate(pixels).as_json(),
        "lens": lens is not None,
    }


def _sample_clipping(pair: dict, description: dict) -> tuple[float, float]:
    """As ``review/reference.py`` measures a sample: its edit against its own frame.

    The tails are those of the photo's proxy, which is what the sample's own
    neutral frame measures to (``measure.neutral_tails``) within a JPEG level.
    """
    from ape.review.tonal import added_clipping
    from ape.style import vector as sv

    params = sv.to_params(pair["vector"], pair["context"])
    return added_clipping(
        description["measured"]["tails"], params, pair["context"].exposure_anchor_ev
    )


def _score(model, description: dict, clipping: dict[int, tuple[float, float]]):
    from ape.pipeline.params import EditParams
    from ape.review import confidence as cf
    from ape.review import tonal
    from ape.review.reference import style_clipping
    from ape.style import vector as sv
    from ape.style.profile import LoadedProfile

    measured = description["measured"]
    context = sv.StyleContext(
        description["as_shot"]["temperature_k"],
        description["as_shot"]["tint"],
        measured["exposure_anchor_ev"],
    )
    prediction = model.predict(
        description["features"], description["embedding"], context.exposure_anchor_ev
    )
    params = sv.to_params(prediction.vector, context, base=EditParams())
    params.geometry.rotation_deg = description["straighten"]["rotation_deg"]
    calibration = LoadedProfile(0, "", False, None, None, model).calibration
    burnt, crushed = tonal.added_clipping(measured["tails"], params, context.exposure_anchor_ev)
    style = style_clipping(
        [(n.sample_id, n.weight) for n in prediction.neighbours], clipping
    ) or (None, None)
    inputs = cf.ConfidenceInputs(
        nearest=prediction.nearest, nearest_p90=calibration.nearest_p90,
        dispersion=prediction.dispersion, dispersion_p90=calibration.dispersion_p90,
        exposure_anchor_ev=context.exposure_anchor_ev,
        anchor_range=(calibration.anchor_min, calibration.anchor_max),
        wb_spread_mired=measured["wb_spread_mired"], burnt=burnt, crushed=crushed,
        burnt_style=style[0], crushed_style=style[1],
        straighten=description["straighten"], lens_profile=description["lens"],
    )
    return cf.score(inputs), prediction, context


def _train(pairs: list[dict], indices) -> object:
    from ape.style.model import TrainingSample
    from ape.style.train import train

    return train(
        [
            TrainingSample(
                int(i), pairs[i]["vector"], pairs[i]["features"], pairs[i]["embedding"],
                pairs[i]["context"].exposure_anchor_ev,
            )
            for i in indices
        ]
    )


def main() -> None:  # noqa: PLR0915 - a report, read top to bottom
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("edited_dir", type=Path)
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--also", type=Path, action="append", default=[])
    parser.add_argument("--scene-seconds", type=float, default=12.0)
    parser.add_argument("--photo-seconds", type=float, default=8.0)
    parser.add_argument("--grid-seconds", type=float, default=3.0)
    args = parser.parse_args()

    import multiprocessing as mp

    from ape.analysis.cluster import ClusterItem, cluster_photos
    from ape.review.confidence import DEFAULT_THRESHOLD
    from ape.style.pairing import RAW_SUFFIXES, list_references, pair_files

    raws = sorted(p for p in args.raw_dir.iterdir() if p.suffix.lower() in RAW_SUFFIXES)
    pairing = pair_files(raws, list_references(args.edited_dir))
    for folder in args.also:
        raws += sorted(p for p in folder.iterdir() if p.suffix.lower() in RAW_SUFFIXES)
    cache = tempfile.mkdtemp(prefix="bench-review-")
    from ape.config import get_settings

    own = get_settings().data_dir
    borrowed = Path(cache) / own.name
    borrowed.mkdir(parents=True)
    for name in ("models", "lensfun"):
        if (own / name).exists():
            (borrowed / name).symlink_to(own / name)
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        pairs = pool.map(
            _work,
            [(str(p.raw), str(p.reference), str(args.cache) if args.cache else None)
             for p in pairing.pairs],
        )
        described = pool.map(_describe, [(str(r), cache) for r in raws])
    print(f"{len(pairs)} coppie e {len(described)} foto descritte in "
          f"{time.perf_counter() - started:.0f} s")
    pairs.sort(key=lambda p: p["raw"])
    by_raw = {d["raw"]: d for d in described}
    by_name = {Path(d["raw"]).name: d for d in described}
    clipping = {i: _sample_clipping(p, by_name[Path(p["raw"]).name]) for i, p in enumerate(pairs)}

    rows = []  # (raw, paired, confidence, reasons, strict, colour)
    everything = list(range(len(pairs)))
    for i, pair in enumerate(pairs):
        model = _train(pairs, [j for j in everything if j != i])
        result, prediction, context = _score(model, by_raw[pair["raw"]], clipping)
        strict, colour = _evaluate({**pair, "context": context}, prediction.vector)
        rows.append((pair["raw"], True, result.value, [t.code for t in result.reasons()],
                     strict, colour))
    full = _train(pairs, everything)
    paired = {p["raw"] for p in pairs}
    for raw in raws:
        if str(raw) in paired:
            continue
        result, _p, _c = _score(full, by_raw[str(raw)], clipping)
        rows.append((str(raw), False, result.value, [t.code for t in result.reasons()],
                     None, None))

    threshold = DEFAULT_THRESHOLD
    queued = [r for r in rows if r[2] < threshold]
    print(f"\nsoglia {threshold}: in coda {len(queued)} su {len(rows)} "
          f"({100 * len(queued) / len(rows):.1f}%); "
          f"accoppiate {sum(r[1] for r in queued)}/{len(pairs)}, "
          f"non accoppiate {sum(not r[1] for r in queued)}/{len(rows) - len(pairs)}")
    print("  primo motivo:", dict(Counter(r[3][0] for r in queued if r[3])))
    print("  tutti i motivi (anche sopra soglia):",
          dict(Counter(code for r in rows for code in r[3])))

    truth = [r for r in rows if r[1]]
    strict = np.array([r[4] for r in truth])
    low = np.array([r[2] < threshold for r in truth])
    print("\nsulle coppie (ΔE contro il JPEG dell'utente, leave-one-out, senza coerenza):")
    print(f"  ΔE stretto: tutte media {strict.mean():.2f}; in coda media "
          f"{strict[low].mean() if low.any() else float('nan'):.2f}; "
          f"passate media {strict[~low].mean():.2f}, peggiore {strict[~low].max():.2f}")
    for limit in (8.0, 10.0, 12.0):
        bad = strict > limit
        print(f"  ΔE > {limit:.0f}: {bad.sum()} foto, di cui in coda {(bad & low).sum()}")
    for raw, _paired, value, reasons, s, _c in sorted(truth, key=lambda r: -r[4])[:8]:
        print(f"    {Path(raw).name}  ΔE {s:5.2f}  confidenza {value:.2f}  {reasons}")

    items = [
        ClusterItem(i, by_raw[r[0]]["embedding"], by_raw[r[0]]["features"],
                    by_raw[r[0]]["shot_at"])
        for i, r in enumerate(rows)
    ]
    clusters = cluster_photos(items)
    sizes = Counter(clusters.labels.values())
    print(f"\nscene ({clusters.basis}): {len(sizes)}; dimensioni "
          f"{sorted(sizes.values(), reverse=True)}")
    seconds = len(sizes) * args.scene_seconds + len(queued) * args.photo_seconds
    per_photo = seconds / len(rows)
    print(f"lavoro stimato ({args.scene_seconds:.0f} s a scena, {args.photo_seconds:.0f} s a "
          f"foto in coda): {seconds / 60:.1f} min per {len(rows)} foto; "
          f"a 500 foto nella stessa proporzione {per_photo * 500 / 60:.1f} min")
    # The grid: every scene judged at a glance of its developed representative
    # and members, approved together; the queue as before. Scenes that need a
    # correction cost what a scene costs above, and how many there are is the
    # one number only the user can give.
    grid = len(sizes) * args.grid_seconds + len(queued) * args.photo_seconds
    print(f"con la griglia ({args.grid_seconds:.0f} s a scena guardata, nessuna da correggere): "
          f"{grid / 60:.1f} min per {len(rows)} foto; "
          f"a 500 foto {grid / len(rows) * 500 / 60:.1f} min; "
          f"ogni scena da correggere aggiunge {args.scene_seconds:.0f} s")


if __name__ == "__main__":
    main()
