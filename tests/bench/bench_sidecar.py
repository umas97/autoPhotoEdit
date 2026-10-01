# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How close darktable's development of our sidecar comes to our own render.

Test 7 only asks that darktable open the XMP without errors. This measures
what the user actually gets: for real ARW files and real looks (the inversions
of the user's own Lightroom edits, cached by ``bench_style.py --cache``), the
sidecar is written, ``darktable-cli`` develops the RAW with it at 1024 px, and
the result is registered on our render of the same parameters (affine ECC, the
two lens corrections crop slightly differently) and compared in ΔE2000 after a
1 px blur, on the frame minus a 2% border.

Usage::

    uv run python tests/bench/bench_sidecar.py [--photos 8] [--looks 3]
        [--cache DIR_OF_bench_style_PICKLES] [--jobs 4]
"""

from __future__ import annotations

import argparse
import contextlib
import pickle
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import cv2
import numpy as np

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EDGE = 1024


def _command(workdir: Path) -> list[str]:
    native = shutil.which("darktable-cli")
    if native:
        return [native]
    return [
        "flatpak",
        "run",
        f"--filesystem={workdir}",
        "--command=darktable-cli",
        "org.darktable.Darktable",
    ]


def _lab(rgb: np.ndarray) -> np.ndarray:
    import colour

    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb, 0, 1)))


def _register(ours: np.ndarray, theirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = ours.shape[:2]
    resized = cv2.resize(theirs, (width, height), interpolation=cv2.INTER_AREA)

    def grey(img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return cv2.equalizeHist(g).astype(np.float32)

    matrix = np.eye(2, 3, dtype=np.float32)
    with contextlib.suppress(cv2.error):
        _, matrix = cv2.findTransformECC(
            grey(ours),
            grey(resized),
            matrix,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
            None,
            5,
        )
    warped = cv2.warpAffine(
        resized,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderValue=(-1, -1, -1),
    )
    mask = warped[..., 0] >= 0
    border = int(0.02 * max(width, height))
    inner = np.zeros_like(mask)
    inner[border:-border, border:-border] = True
    return warped, mask & inner


def _case(job: tuple[str, dict, str]) -> dict:
    from ape.export import xmp_darktable
    from ape.export.sidecar import read_sidecar_context
    from ape.pipeline.filters import resize_long_edge
    from ape.pipeline.params import EditParams
    from ape.pipeline.render import RenderOptions, render
    from ape.raw.decode import decode_linear

    raw_path, spec, label = job
    if "vector" in spec:
        # A style's look as the program applies it to *this* photo: white
        # balance as a shift from this photo's as-shot, exposure on its anchor.
        from ape.style import vector as style_vector

        own = read_sidecar_context(raw_path)
        context = style_vector.StyleContext(
            as_shot_temperature_k=own.as_shot_temperature_k,
            as_shot_tint=own.as_shot_tint,
            exposure_anchor_ev=spec["anchor"],
        )
        params = style_vector.to_params(np.asarray(spec["vector"]), context)
    else:
        params = EditParams.from_dict(spec)
    with tempfile.TemporaryDirectory(prefix="ape-dt-") as tmp:
        work = Path(tmp)
        raw = work / Path(raw_path).name
        shutil.copy2(raw_path, raw)
        ctx = read_sidecar_context(raw)
        sidecar = work / f"{raw.name}.xmp"
        sidecar.write_bytes(xmp_darktable.build(params, ctx))
        out = work / "dt.jpg"
        run = subprocess.run(
            [
                *_command(work),
                str(raw),
                str(sidecar),
                str(out),
                "--width",
                str(EDGE),
                "--height",
                str(EDGE),
                "--core",
                "--library",
                ":memory:",
                "--configdir",
                str(work / "conf"),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if run.returncode != 0 or not out.is_file():
            return {"label": label, "raw": raw.name, "error": run.stderr[-500:]}
        theirs = cv2.imread(str(out))[..., ::-1].astype(np.float32) / 255.0

    decoded = decode_linear(raw_path, half_size=True)
    decoded.rgb = np.ascontiguousarray(resize_long_edge(decoded.rgb, EDGE))
    ours = render(decoded, params, RenderOptions(long_edge=EDGE))
    warped, mask = _register(ours, theirs)
    a = cv2.GaussianBlur(ours, (0, 0), 1.0)
    b = cv2.GaussianBlur(np.clip(warped, 0, 1), (0, 0), 1.0)
    import colour

    lab_a, lab_b = _lab(a), _lab(b)
    delta = colour.delta_E(lab_a, lab_b, method="CIE 2000")[mask]
    shift = (lab_b - lab_a)[mask].mean(axis=0)
    return {
        "label": label,
        "raw": Path(raw_path).name,
        "mean": float(delta.mean()),
        "p90": float(np.percentile(delta, 90)),
        "dL": float(shift[0]),
        "da": float(shift[1]),
        "db": float(shift[2]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--photos", type=int, default=8)
    parser.add_argument("--looks", type=int, default=3)
    parser.add_argument("--cache", type=Path, help="pickles of bench_style.py --cache")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    from ape.pipeline.params import EditParams, neutral_params

    raws = sorted(p for p in FIXTURES.iterdir() if p.suffix.lower() == ".arw")
    step = max(1, len(raws) // args.photos)
    chosen = raws[::step][: args.photos]
    looks: list[tuple[str, dict]] = [("neutro", neutral_params().model_dump(mode="json"))]
    if args.cache and args.cache.is_dir():
        from ape.style import vector as style_vector

        from bench_style import CACHE_SUFFIX

        pickles = sorted(args.cache.glob("*" + CACHE_SUFFIX))
        for path in pickles[:: max(1, len(pickles) // args.looks)][: args.looks]:
            with path.open("rb") as handle:
                data = pickle.load(handle)
            context = data["context"]
            if isinstance(context, dict):
                context = style_vector.StyleContext(**context)
            spec = {
                "vector": list(map(float, data["vector"])),
                "anchor": context.exposure_anchor_ev,
            }
            looks.append((f"look {Path(data['raw']).stem}", spec))
    geometry = EditParams()
    geometry.geometry.rotation_deg = -2.0
    looks.append(("rotazione -2°", geometry.model_dump(mode="json")))

    jobs = [(str(raw), params, label) for raw in chosen for label, params in looks]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_case, jobs))

    by_label: dict[str, list[dict]] = {}
    for result in results:
        by_label.setdefault(result["label"], []).append(result)
    for label, rows in by_label.items():
        good = [r for r in rows if "error" not in r]
        failed = len(rows) - len(good)
        if not good:
            print(f"{label:22s} tutti falliti: {rows[0]['error']}")
            continue
        means = np.array([r["mean"] for r in good])
        shift = np.mean([[r["dL"], r["da"], r["db"]] for r in good], axis=0)
        print(
            f"{label:22s} ΔE2000 media {means.mean():5.2f} "
            f"(foto: {means.min():.2f}-{means.max():.2f}), "
            f"p90 medio {np.mean([r['p90'] for r in good]):5.2f}, "
            f"spostamento L* {shift[0]:+.2f} a* {shift[1]:+.2f} b* {shift[2]:+.2f}"
            + (f", {failed} falliti" if failed else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
