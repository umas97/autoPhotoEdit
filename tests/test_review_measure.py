# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The pixel measurements behind the confidence (``review/measure.py``, ``tonal.py``).

* Mixed light: a frame whose half is under a warmer light spreads the cells'
  white balance by far more than the same frame under one light.
* Clipping: the estimate from the neutral proxy agrees with a real render of
  the RAW on the thresholds of section 9.1 (2% burnt, 5% black); what counts
  is what the edit adds to the frame's own clipping.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.pipeline.params import EditParams
from ape.review import measure, tonal
from conftest import available_raws


def _srgb(linear: np.ndarray) -> np.ndarray:
    gamma = 1.055 * np.power(np.clip(linear, 0, 1), 1 / 2.4) - 0.055
    encoded = np.where(linear <= 0.0031308, linear * 12.92, gamma)
    return (np.clip(encoded, 0, 1) * 255 + 0.5).astype(np.uint8)


def _scene(seed: int = 0) -> np.ndarray:
    """Linear sRGB patches: mostly greys and muted colours, like a real frame."""
    rng = np.random.default_rng(seed)
    tiles = rng.uniform(0.05, 0.6, size=(24, 36, 1)) * rng.uniform(0.85, 1.15, size=(24, 36, 3))
    return np.kron(tiles, np.ones((16, 16, 1)))  # 384 x 576


def test_one_light_has_little_spread_and_mixed_light_a_lot():
    scene = _scene()
    single = measure.wb_spread_mired(
        measure._display(_srgb(scene)), as_shot_temperature_k=5500.0, as_shot_tint=0.0
    )
    mixed = scene.copy()
    # The left half under tungsten: about 3000 K against the 5500 K of the rest,
    # some 150 mired apart.
    mixed[:, : mixed.shape[1] // 2] *= np.array([2.0, 1.0, 0.45])
    spread = measure.wb_spread_mired(
        measure._display(_srgb(mixed)), as_shot_temperature_k=5500.0, as_shot_tint=0.0
    )
    assert single is not None and spread is not None
    assert single < 20.0
    assert spread > 100.0  # past the ramp of ``confidence._WB_MIRED``: full risk


def test_a_frame_too_dark_to_judge_has_no_spread():
    dark = np.full((384, 576, 3), 0.004)
    assert (
        measure.wb_spread_mired(
            measure._display(_srgb(dark)), as_shot_temperature_k=5500.0, as_shot_tint=0.0
        )
        is None
    )


def test_brightening_burns_more_and_darkening_crushes_more():
    tails = measure.tails(measure._display(_srgb(_scene() * 1.6)))
    base = EditParams()
    brighter = base.model_copy(deep=True)
    brighter.exposure.ev = 2.0
    darker = base.model_copy(deep=True)
    darker.exposure.ev = -5.0
    burnt, crushed = tonal.clipped_after(tails, base)
    assert tonal.clipped_after(tails, brighter)[0] > burnt
    assert tonal.clipped_after(tails, darker)[1] > crushed
    assert tonal.clipped_after(None, base) is None


def test_lifted_local_shadows_crush_less():
    """The local shadows of a learned style lift the regions the estimate reads."""
    tails = measure.tails(measure._display(_srgb(_scene() * 0.08)))
    base = EditParams()
    base.exposure.ev = -2.0
    lifted = base.model_copy(deep=True)
    lifted.local_contrast.shadows = 0.8
    assert tonal.clipped_after(tails, lifted)[1] < tonal.clipped_after(tails, base)[1]


def test_only_the_clipping_the_edit_adds_counts():
    """A frame white in the scene is not the edit's doing; brightening it further is."""
    scene = _scene()
    scene[: scene.shape[0] // 4] = 1.0  # a quarter of sky, burnt as shot
    tails = measure.tails(measure._display(_srgb(scene)))
    anchor = 0.0
    neutral = tonal.reference_params(anchor)
    assert tonal.clipped_after(tails, neutral)[0] > 0.2
    assert tonal.added_clipping(tails, neutral, anchor) == (0.0, 0.0)
    brighter = neutral.model_copy(deep=True)
    brighter.exposure.ev += 1.5
    assert tonal.added_clipping(tails, brighter, anchor)[0] > 0.02
    assert tonal.added_clipping(tails, neutral, None) is None
    assert tonal.added_clipping(None, neutral, anchor) is None


def test_neutral_tails_quantise_a_float_render_like_the_proxy():
    image = _srgb(_scene())
    assert measure.neutral_tails(image.astype(np.float32) / 255.0) == measure.neutral_tails(image)


@pytest.mark.fixtures
@pytest.mark.slow
def test_the_clipping_estimate_agrees_with_real_renders(xdg_home):
    """On real ARWs, proxy-based estimate vs the render of the RAW itself."""
    from PIL import Image

    from ape.pipeline.render import render
    from ape.raw.proxy import build_proxy, editing_proxy
    from ape.style import auto
    from ape.style import vector as sv

    raws = {p.stem: p for p in available_raws()}
    names = [n for n in ("DSC05618", "DSC05631", "DSC05638", "DSC05641") if n in raws]
    if len(names) < 3:
        pytest.skip("servono i file ARW di tests/fixtures/")
    agree = total = 0
    for name in names:
        proxy = build_proxy(raws[name], name, force=True).path
        with Image.open(proxy) as image:
            pixels = np.asarray(image.convert("RGB"))
        decoded = editing_proxy(raws[name], long_edge=1024)
        camera = decoded.camera
        tails = measure.tails(measure._display(pixels))
        anchor = auto.measure(
            pixels,
            as_shot_temperature_k=camera.as_shot_temperature_k,
            as_shot_tint=camera.as_shot_tint,
            white_balance=False,
        ).exposure_anchor_ev
        context = sv.StyleContext(camera.as_shot_temperature_k, camera.as_shot_tint, anchor)
        for offset in (-1.0, 0.0, 1.0, 2.0):
            vector = sv.neutral_vector()
            vector[sv.index("exposure_offset")] = offset
            params = sv.to_params(vector, context)
            image = (np.clip(render(decoded, params), 0, 1) * 255 + 0.5).astype(np.uint8)
            actual = ((image.max(axis=2) >= 250).mean(), (image.max(axis=2) <= 4).mean())
            estimate = tonal.clipped_after(tails, params)
            assert abs(estimate[0] - actual[0]) < 0.05, (name, offset, estimate, actual)
            agree += int((estimate[0] > 0.02) == (actual[0] > 0.02))
            agree += int((estimate[1] > 0.05) == (actual[1] > 0.05))
            total += 2
    assert agree >= 0.85 * total, (agree, total)
