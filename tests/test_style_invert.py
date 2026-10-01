# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The inversion of section 8.2, and the pieces it is made of.

Test 3 of section 13 is here: develop a RAW with known parameters, save the
JPEG, invert, and the recovered parameters must render within ΔE 2.0 of it. It
runs on a synthetic scene always, and on one of the user's ARWs when the
fixtures are there.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from ape.pipeline.ops import local_contrast
from ape.pipeline.params import EditParams, LocalContrastParams
from ape.pipeline.render import RenderOptions, render
from ape.style import cmaes, learn
from ape.style import vector as sv
from ape.style.colordiff import delta_e_2000, srgb_to_lab
from conftest import as_decoded, available_raws, build_scene


def _jpeg_round_trip(image: np.ndarray, quality: int = 95) -> np.ndarray:
    """What "save the JPEG" means: 8 bits and a lossy code, like a real export."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        buffer, format="JPEG", quality=quality
    )
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB")).astype(np.float32) / 255.0


def _known_vector() -> np.ndarray:
    """A plausible edit that uses every family of controls the inversion searches."""
    v = sv.neutral_vector()
    for name, value in {
        "wb_mired_shift": 18.0,
        "wb_tint_shift": -6.0,
        "exposure_offset": 0.45,
        "contrast": 1.45,
        "white_point_ev": 5.2,
        "shadows": 0.25,
        "highlights": -0.2,
        "vibrance": 0.3,
        "saturation": -0.1,
        "local_highlights": -0.4,
        "local_shadows": 0.2,
        "hsl_blue_saturation": 0.3,
        "hsl_green_hue": 0.2,
        "clarity": 0.2,
    }.items():
        v[sv.index(name)] = value
    return v


def test_colordiff_matches_colour_science():
    import colour

    rng = np.random.default_rng(3)
    a = rng.random((2000, 3)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.06, a.shape), 0, 1).astype(np.float32)
    lab_a = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(a.astype(np.float64)))
    lab_b = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(b.astype(np.float64)))
    assert np.abs(srgb_to_lab(a) - lab_a).max() < 0.05
    reference = colour.difference.delta_E_CIE2000(lab_a, lab_b)
    ours = delta_e_2000(srgb_to_lab(a), srgb_to_lab(b))
    assert np.abs(ours - reference).max() < 0.01


def test_cmaes_finds_a_shifted_quadratic():
    target = np.array([0.2, 0.7, 0.4, 0.9])

    def f(x):
        return float(np.sum((x - target) ** 2 * np.array([1, 4, 9, 16])))

    result = cmaes.minimise(f, np.full(4, 0.5), sigma=0.2, budget=400, seed=1)
    assert result.value < 1e-4
    assert result.evaluations <= 400
    # Seeded: the same search every time.
    again = cmaes.minimise(f, np.full(4, 0.5), sigma=0.2, budget=400, seed=1)
    assert np.array_equal(result.x, again.x)


def test_local_tone_at_zero_is_bit_identical():
    rng = np.random.default_rng(0)
    image = rng.random((120, 160, 3), dtype=np.float32)
    params = LocalContrastParams(clarity=0.0)
    assert local_contrast.apply(image, params) is image
    edited = local_contrast.apply(image, LocalContrastParams(highlights=-1.0, shadows=1.0))
    assert not np.array_equal(edited, image)


def test_local_tone_keeps_regions_in_order():
    """Full travel darkens the bright regions and lifts the dark ones, and never
    swaps them: the mapping of the base layer is monotonic (see TONE_TRAVEL)."""
    base = np.linspace(0.0, 1.0, 2001, dtype=np.float32)
    for shadows, highlights in ((1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0)):
        mapped = base * local_contrast.local_tone_gain(base, shadows, highlights)
        assert np.all(np.diff(mapped) >= -1e-6), (shadows, highlights)


def test_old_params_without_local_tone_still_load():
    payload = EditParams().model_dump(mode="json")
    del payload["local_contrast"]["shadows"], payload["local_contrast"]["highlights"]
    loaded = EditParams.from_dict(payload)
    assert loaded.local_contrast.shadows == 0.0 and loaded.local_contrast.highlights == 0.0


def test_style_vector_round_trips_through_params():
    context = sv.StyleContext(4300.0, 7.0, 0.8)
    vector = _known_vector()
    vector[sv.index("split_shadow_x")] = 0.2
    vector[sv.index("split_shadow_y")] = -0.1
    back = sv.from_params(sv.to_params(vector, context), context)
    assert np.allclose(back, vector, atol=1e-6)


def _invert_synthetic(index: int) -> tuple[float, learn.InversionResult]:
    decoded = as_decoded(build_scene(index, long_edge=512))
    context = sv.StyleContext(decoded.camera.as_shot_temperature_k, decoded.camera.as_shot_tint)
    truth = sv.to_params(_known_vector(), context)
    reference = _jpeg_round_trip(render(decoded, truth, RenderOptions()))
    mask = np.ones(reference.shape[:2], dtype=bool)
    result = learn.invert(learn.PairImages(decoded, reference, mask, context))
    recovered = render(decoded, sv.to_params(result.vector, context), RenderOptions())
    error = float(delta_e_2000(srgb_to_lab(recovered), srgb_to_lab(reference)).mean())
    return error, result


@pytest.mark.parametrize("index", [0, 3])
def test_inversion_converges_on_a_synthetic_scene(index: int):
    """Test 3 of section 13, on a scene whose every pixel we know."""
    error, result = _invert_synthetic(index)
    assert error < 2.0, f"ΔE residuo {error:.2f}"
    assert result.evaluations <= learn.BUDGET


def test_inversion_is_deterministic():
    decoded = as_decoded(build_scene(1, long_edge=256))
    context = sv.StyleContext(6504.0, 0.0)
    reference = render(decoded, sv.to_params(_known_vector(), context), RenderOptions())
    mask = np.ones(reference.shape[:2], dtype=bool)
    pair = learn.PairImages(decoded, reference, mask, context)
    first = learn.invert(pair, budget=120)
    second = learn.invert(pair, budget=120)
    assert np.array_equal(first.vector, second.vector)


@pytest.mark.fixtures
def test_inversion_converges_on_a_real_raw():
    """Test 3 on one of the user's ARWs, at the comparison size of section 8.2."""
    from dataclasses import replace

    from ape.pipeline.filters import resize_long_edge
    from ape.raw.decode import decode_linear

    raws = available_raws()
    if not raws:
        pytest.skip("nessun ARW in tests/fixtures/")
    decoded = decode_linear(raws[len(raws) // 2], half_size=True)
    decoded = replace(decoded, rgb=np.ascontiguousarray(resize_long_edge(decoded.rgb, 512)))
    context = sv.StyleContext(decoded.camera.as_shot_temperature_k, decoded.camera.as_shot_tint)
    reference = _jpeg_round_trip(render(decoded, sv.to_params(_known_vector(), context)))
    mask = np.ones(reference.shape[:2], dtype=bool)
    result = learn.invert(learn.PairImages(decoded, reference, mask, context))
    recovered = render(decoded, sv.to_params(result.vector, context))
    error = float(delta_e_2000(srgb_to_lab(recovered), srgb_to_lab(reference)).mean())
    assert error < 2.0, f"ΔE residuo {error:.2f}"
