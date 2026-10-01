# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 1 of docs/SPEC.md section 13: resolution invariance.

"For 10 images and 20 random ``EditParams``, the 1024 px render and the
full-resolution render downscaled to 1024 px must have a mean dE2000 below 1.5."

This is the test that keeps the promise the UI makes: the preview is computed on
a 2048 px proxy, the export on 24 MP, and the user is told they are the same
picture. Every spatial filter in the pipeline takes its radius as a fraction of
the long edge precisely so that this holds.

The "full resolution" here is 2048 px rather than 6000. The property under test
is scale invariance, which a 2:1 ratio exercises exactly as well as a 6:1 one,
and 200 renders at 24 MP would put this test out of reach of a normal test run.
The fixture-backed variant at the bottom does check a real file, at real size.

Two hundred pairs of renders is still minutes of arithmetic, so they run across a
process pool -- the same shape section 12 gives the real worker pool, one image
per process with the thread count pinned to one, which is also the configuration
the timings in the benchmark are quoted for. The pool is started with ``spawn``:
by the time this file runs, earlier tests have had OpenCV start its own threads,
and a ``fork`` from a process with threads gives children that inherit locks
held by threads that do not exist in them. The result is a worker asleep on a
futex for ever, which is exactly what this test did when it was forked.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from ape.pipeline.colorspace import delta_e_2000, display_decode, to_lab
from ape.pipeline.filters import resize_long_edge
from ape.pipeline.params import HSL_BANDS, EditParams
from ape.pipeline.render import RenderOptions, render
from conftest import SCENE_COUNT, as_decoded, build_scene

PREVIEW_EDGE = 1024
FULL_EDGE = 2048
TOLERANCE = 1.5

#: Border to ignore when comparing. Every spatial filter reflects at the edge,
#: and reflection at 2048 px and at 1024 px cannot produce the same pixels --
#: that is a boundary condition, not a scale dependency.
_MARGIN = 24

#: Compare every second pixel on each axis. The statistic under test is a mean
#: over roughly a million pixels; a quarter of them estimates it to well under
#: the third decimal, and CIEDE2000 in float64 is the slowest thing here.
_STRIDE = 2


def _random_params(rng: np.random.Generator) -> EditParams:
    """Draw plausible parameters, with every spatial radius in play."""
    bands = {
        name: {
            "hue": float(rng.uniform(-0.4, 0.4)),
            "saturation": float(rng.uniform(-0.5, 0.5)),
            "luminance": float(rng.uniform(-0.3, 0.3)),
        }
        for name in rng.choice(HSL_BANDS, size=3, replace=False)
    }
    return EditParams.model_validate(
        {
            "exposure": {"ev": float(rng.uniform(-1.5, 1.5))},
            "highlight_recovery": {
                "strength": float(rng.uniform(0.0, 1.0)),
                "threshold": float(rng.uniform(0.85, 0.99)),
            },
            "noise": {
                "luminance": float(rng.uniform(0.0, 0.8)),
                "chrominance": float(rng.uniform(0.0, 1.0)),
                # Lower bound chosen so the filter spans at least two pixels at
                # 1024 px: below that the radius quantises to a single pixel at
                # both sizes and the comparison stops being about scale.
                "radius": float(rng.uniform(0.002, 0.02)),
            },
            "tone": {
                "black_point_ev": float(rng.uniform(-12.0, -5.0)),
                "white_point_ev": float(rng.uniform(2.5, 7.0)),
                "contrast": float(rng.uniform(0.7, 2.0)),
                "pivot": float(rng.uniform(-0.8, 0.8)),
                "toe": float(rng.uniform(0.6, 3.0)),
                "shoulder": float(rng.uniform(0.6, 3.0)),
                "chroma_preservation": float(rng.uniform(0.0, 1.0)),
            },
            "tone_shaping": {
                "shadows": float(rng.uniform(-1, 1)),
                "highlights": float(rng.uniform(-1, 1)),
                "whites": float(rng.uniform(-0.6, 0.6)),
                "blacks": float(rng.uniform(-0.6, 0.6)),
            },
            "tone_curve": {
                "highlights": float(rng.uniform(-0.6, 0.6)),
                "lights": float(rng.uniform(-0.6, 0.6)),
                "darks": float(rng.uniform(-0.6, 0.6)),
                "shadows": float(rng.uniform(-0.6, 0.6)),
            },
            "color": {
                "saturation": float(rng.uniform(-0.6, 0.6)),
                "vibrance": float(rng.uniform(-0.6, 0.6)),
                "hsl": bands,
                "split_toning": {
                    "shadow_hue": float(rng.uniform(0, 360)),
                    "shadow_saturation": float(rng.uniform(0, 0.5)),
                    "highlight_hue": float(rng.uniform(0, 360)),
                    "highlight_saturation": float(rng.uniform(0, 0.5)),
                    "balance": float(rng.uniform(-0.5, 0.5)),
                },
            },
            "local_contrast": {
                "clarity": float(rng.uniform(-0.8, 0.8)),
                "radius": float(rng.uniform(0.006, 0.06)),
            },
            "sharpen": {
                "amount": float(rng.uniform(0.0, 1.5)),
                "radius": float(rng.uniform(0.0008, 0.005)),
                "threshold": float(rng.uniform(0.0, 0.05)),
            },
        }
    )


def _mean_delta_e(scene: np.ndarray, params: EditParams) -> float:
    """dE2000 between the proxy render and the full-size render of one scene.

    ``scene`` is the full-size frame; the proxy is derived from it by halving,
    exactly as the real program derives a proxy from a decoded RAW. Building the
    two sizes independently would compare two samplings of the scene rather than
    one scene at two sizes, and on fine detail that difference alone is worth
    several dE.

    Both renders are asked for the *same* output size, so the downscale happens
    inside the pipeline, at the point section 6.2 puts it: after local contrast
    and before sharpening. That matters. Sharpening is output-referred and is
    not scale invariant in any program -- sharpening at 2048 px and then
    shrinking to 1024 measures how much a resize attenuates an unsharp mask,
    which is not the question. The question is whether the proxy the UI shows
    matches the export *at the size it is exported*, and this is that question.
    """
    options = RenderOptions(stop_before_output=True, long_edge=PREVIEW_EDGE)
    preview = render(as_decoded(resize_long_edge(scene, PREVIEW_EDGE)), params, options)
    full = render(as_decoded(scene), params, options)

    # Sizes can differ by a pixel after two independent roundings.
    height = min(preview.shape[0], full.shape[0]) - _MARGIN
    width = min(preview.shape[1], full.shape[1]) - _MARGIN

    sampled = (slice(_MARGIN, height, _STRIDE), slice(_MARGIN, width, _STRIDE))
    lab_preview = to_lab(display_decode(preview[sampled]))
    lab_full = to_lab(display_decode(full[sampled]))
    return float(np.mean(delta_e_2000(lab_preview, lab_full)))


#: Section 12's rule, capped: each worker holds several full buffers plus the
#: float64 Lab arrays, and sixteen of those at once would not fit in 8 GB.
WORKERS = max(2, min(8, (os.cpu_count() or 4) - 2))


def _one_combination(job: tuple[int, str]) -> float:
    """One (scene, parameters) pair, in a worker process."""
    import cv2

    cv2.setNumThreads(1)
    scene_index, payload = job
    scene = build_scene(scene_index, FULL_EDGE)
    return _mean_delta_e(scene, EditParams.from_json(payload))


def test_preview_matches_full_resolution():
    """The headline assertion: 10 scenes x 20 parameter sets, mean dE2000 < 1.5."""
    rng = np.random.default_rng(1024)
    jobs = [
        (scene_index, _random_params(rng).to_json(indent=None))
        for scene_index in range(SCENE_COUNT)
        for _ in range(20)
    ]

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        deltas = list(pool.map(_one_combination, jobs, chunksize=2))

    failures = [
        (jobs[index][0], index % 20, delta)
        for index, delta in enumerate(deltas)
        if delta >= TOLERANCE
    ]
    worst = max(deltas)
    assert not failures, (
        f"{len(failures)} combinazioni su {len(jobs)} oltre {TOLERANCE} dE2000; "
        f"peggiore {worst:.3f}"
    )
    print(f"\ndE2000 medio {np.mean(deltas):.3f}, peggiore {worst:.3f} su {len(jobs)} combinazioni")


def test_sharpening_radius_scales_with_the_image():
    """A dedicated check on the one operation that is purely about pixel size."""
    params = EditParams.model_validate({"sharpen": {"amount": 1.5, "radius": 0.003}})
    for index in range(3):
        assert _mean_delta_e(build_scene(index, FULL_EDGE), params) < TOLERANCE


def test_clarity_radius_scales_with_the_image():
    params = EditParams.model_validate({"local_contrast": {"clarity": 0.8, "radius": 0.04}})
    for index in range(3):
        assert _mean_delta_e(build_scene(index, FULL_EDGE), params) < TOLERANCE


def test_the_checkerboard_survives_a_guided_filter_at_both_sizes():
    """The scene that used to fail, on the operations that made it fail.

    Clarity and denoising both run a guided filter, and a checkerboard is the
    worst thing one can be given: every pixel is next to an edge. If the filter
    ever goes back to picking its working resolution from the input size rather
    than from the radius, this is where it shows first.
    """
    params = EditParams.model_validate(
        {
            "local_contrast": {"clarity": 0.8, "radius": 0.02},
            "noise": {"luminance": 0.6, "chrominance": 1.0, "radius": 0.01},
        }
    )
    assert _mean_delta_e(build_scene(4, FULL_EDGE), params) < TOLERANCE


@pytest.mark.fixtures
@pytest.mark.slow
def test_preview_matches_full_resolution_on_a_real_raw(raw_fixtures):
    """Same property, on a real ARW at its native resolution."""
    from ape.raw.decode import decode_linear

    rng = np.random.default_rng(7)
    decoded = decode_linear(raw_fixtures[0])
    options = RenderOptions(stop_before_output=True, long_edge=PREVIEW_EDGE)

    preview_source = as_decoded(resize_long_edge(decoded.rgb, PREVIEW_EDGE))
    preview_source.baseline_exposure_ev = decoded.baseline_exposure_ev
    preview_source.camera = decoded.camera
    # Without the optics the full-size render is lens-corrected (vignetting
    # included) and the preview is not: a 9.8 dE2000 difference that is about
    # the test, not about scale.
    preview_source.lens = decoded.lens

    worst = 0.0
    for _ in range(3):
        params = _random_params(rng)
        full = render(decoded, params, options)
        preview = render(preview_source, params, options)

        height = min(preview.shape[0], full.shape[0]) - _MARGIN
        width = min(preview.shape[1], full.shape[1]) - _MARGIN
        crop = (slice(_MARGIN, height), slice(_MARGIN, width))
        delta = float(
            np.mean(
                delta_e_2000(
                    to_lab(display_decode(preview[crop])),
                    to_lab(display_decode(full[crop])),
                )
            )
        )
        worst = max(worst, delta)
        assert delta < TOLERANCE, f"dE2000 medio {delta:.3f}"
    print(f"\n{raw_fixtures[0].name}: dE2000 medio peggiore {worst:.3f}")
