# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The memory work of phase 8 changes no pixel.

The full-resolution render runs its per-pixel stages band by band
(``pipeline/bands.py``), the guided filter works one channel at a time and can
write into its input, and an export releases the decoded frame early. Each of
these is only acceptable because it gives the same bits as before; these tests
say so on frames big enough to take the banded path.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.pipeline import bands
from ape.pipeline.filters import guided_filter
from ape.pipeline.params import CropRect, EditParams, HSLBand
from ape.pipeline.render import render


def _look() -> EditParams:
    params = EditParams()
    params.exposure.ev = 0.4
    params.noise.luminance = 0.3
    params.tone_shaping.shadows = 0.2
    params.tone_curve.points = [(0.0, 0.0), (0.3, 0.25), (1.0, 1.0)]
    params.color.vibrance = 0.2
    params.color.saturation = 0.1
    params.color.hsl = {"green": HSLBand(hue=0.2, saturation=-0.3)}
    params.color.split_toning.highlight_hue = 40.0
    params.color.split_toning.highlight_saturation = 0.2
    params.local_contrast.shadows = 0.3
    params.local_contrast.clarity = 0.2
    params.geometry.rotation_deg = 1.2
    params.geometry.crop = CropRect(x=0.05, y=0.05, width=0.9, height=0.9)
    return params


@pytest.fixture(scope="module")
def big_scene() -> np.ndarray:
    """4.3 MP of texture, highlights past white and saturated colours."""
    rng = np.random.default_rng(8)
    scene = rng.uniform(0.0, 0.6, size=(1800, 2400, 3)).astype(np.float32)
    scene[:300, :, :] *= 3.0  # a blown sky: highlight recovery and gamut work
    scene[900:1100, 1000:1400, 1] *= 0.05  # a saturated magenta patch
    return scene


def test_banded_render_is_identical_to_the_per_stage_one(big_scene, monkeypatch):
    from conftest import as_decoded

    assert big_scene.shape[0] * big_scene.shape[1] >= bands.BAND_MIN_PIXELS
    params = _look()
    banded = render(as_decoded(big_scene), params)
    monkeypatch.setattr(bands, "BAND_MIN_PIXELS", 10**12)
    whole = render(as_decoded(big_scene), params)
    assert np.array_equal(banded, whole)


def test_consume_releases_the_frame_and_changes_nothing(big_scene):
    from conftest import as_decoded

    params = _look()
    kept = as_decoded(big_scene.copy())
    consumed = as_decoded(big_scene.copy())
    expected = render(kept, params)
    assert kept.rgb.shape == big_scene.shape  # untouched without the flag
    assert np.array_equal(render(consumed, params, consume=True), expected)
    assert consumed.rgb.size == 0
    # The context the sidecars need survives the release.
    assert consumed.camera is not None
    assert consumed.baseline_exposure_ev == kept.baseline_exposure_ev


@pytest.mark.parametrize("subsample", [True, False])
def test_guided_filter_can_write_into_its_input(subsample):
    rng = np.random.default_rng(3)
    guide = rng.uniform(0, 1, size=(300, 400)).astype(np.float32)
    src = rng.uniform(-0.2, 0.2, size=(300, 400, 3)).astype(np.float32)
    expected = guided_filter(guide, src.copy(), 0.03, 0.01, subsample=subsample)
    in_place = src.copy()
    result = guided_filter(guide, in_place, 0.03, 0.01, subsample=subsample, out=in_place)
    assert result is in_place
    assert np.array_equal(result, expected)
    # Channel by channel is the same as the channels one call at a time.
    for index in range(3):
        alone = guided_filter(guide, np.ascontiguousarray(src[..., index]), 0.03, 0.01,
                              subsample=subsample)
        assert np.array_equal(alone, expected[..., index])
