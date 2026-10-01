# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured render timings against the targets of docs/SPEC.md section 12.

Run it, do not guess it:

```sh
uv run python tests/bench/bench_render.py            # synthetic 24 MP frame
uv run python tests/bench/bench_render.py file.ARW   # a real file, decode included
```

Single-threaded on purpose: section 12 states its per-photo targets for one
core, because the real parallelism is one photo per worker across fourteen of
them, and a benchmark that quietly used all sixteen threads would report a
number that does not survive a batch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

import cv2  # noqa: E402
from ape.pipeline.params import EditParams, neutral_params  # noqa: E402
from ape.pipeline.render import STAGES, RenderOptions, _Context, render  # noqa: E402
from ape.raw.decode import decoded_from_array  # noqa: E402

#: Section 12's per-photo budget for a full-resolution export, one core.
TARGET_EXPORT_SECONDS = 3.5

#: Everything switched on, to bound the worst case rather than the typical one.
HEAVY = EditParams.model_validate(
    {
        "exposure": {"ev": 0.3},
        "noise": {"luminance": 0.4, "chrominance": 0.6},
        "tone": {"contrast": 1.35, "chroma_preservation": 0.5},
        "tone_shaping": {"shadows": 0.3, "highlights": -0.3},
        "tone_curve": {"points": [(0.0, 0.0), (0.5, 0.55), (1.0, 1.0)]},
        "color": {
            "saturation": 0.15,
            "vibrance": 0.3,
            "hsl": {"blue": {"saturation": 0.3}, "orange": {"hue": -0.1}},
        },
        "local_contrast": {"clarity": 0.4},
        "sharpen": {"amount": 0.8},
    }
)


def _synthetic(megapixels: float) -> np.ndarray:
    """A frame with real structure: a flat noise field is not a photograph."""
    height = int(round(np.sqrt(megapixels * 1e6 / 1.5)))
    width = int(round(height * 1.5))
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = 0.02 + 0.6 * (x / width) * (0.4 + 0.6 * (y / height))
    tint = np.array([1.05, 1.0, 0.9], dtype=np.float32)
    grain = rng.normal(0.0, 0.01, size=(height, width, 3)).astype(np.float32)
    return np.clip(base[..., None] * tint + grain, 1e-4, None).astype(np.float32)


def _time_stages(decoded, params, options, repeats: int = 2) -> list[tuple[str, float]]:
    """Best-of-N per stage.

    The first pass through a 24 MP pipeline pays for page faults on a quarter of
    a gigabyte of fresh buffers, which lands on whichever stage happened to
    allocate first and makes the breakdown add up to three times the render it
    is breaking down. Timing it twice and keeping the faster pass measures the
    arithmetic instead of the allocator.
    """
    ctx = _Context(decoded=decoded, params=params, options=options)
    best: dict[str, float] = {}
    for _ in range(max(2, repeats)):
        img = decoded.rgb
        for name, function, _key in STAGES:
            started = time.perf_counter()
            img = function(img, ctx)
            elapsed = time.perf_counter() - started
            best[name] = min(best.get(name, elapsed), elapsed)
    return [(name, best[name]) for name, _function, _key in STAGES]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", nargs="?", help="file ARW da misurare (opzionale)")
    parser.add_argument("--megapixels", type=float, default=24.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    cv2.setNumThreads(1)

    if args.raw:
        from ape.raw.decode import decode_linear
        from ape.safety import register_protected_root

        source = Path(args.raw).expanduser()
        register_protected_root(source.parent)
        started = time.perf_counter()
        decoded = decode_linear(source)
        print(f"decodifica {source.name}: {time.perf_counter() - started:.2f}s")
    else:
        decoded = decoded_from_array(_synthetic(args.megapixels))

    height, width = decoded.rgb.shape[:2]
    print(f"immagine {width}x{height} = {width * height / 1e6:.1f} MP, 1 core\n")

    options = RenderOptions()
    for label, params in (("neutri", neutral_params()), ("tutto attivo", HEAVY)):
        timings = _time_stages(decoded, params, options)
        total = sum(value for _name, value in timings)
        print(f"  parametri {label}: somma degli stadi {total:.2f}s")
        for name, value in timings:
            if value >= 0.05:
                print(f"    {name:20s} {value:6.2f}s")

        best = min(
            _timed(render, decoded, params, options) for _ in range(args.repeats)
        )
        verdict = "OK" if best <= TARGET_EXPORT_SECONDS else "OLTRE IL TARGET"
        print(
            f"    {'render completo':20s} {best:6.2f}s  "
            f"(target {TARGET_EXPORT_SECONDS}s: {verdict})\n"
        )
    return 0


def _timed(function, *args) -> float:
    started = time.perf_counter()
    function(*args)
    return time.perf_counter() - started


if __name__ == "__main__":
    raise SystemExit(main())
