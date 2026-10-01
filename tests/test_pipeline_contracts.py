# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The invariants each operation promises in its docstring, checked.

Section 26 asks every pipeline operation to be a pure function with a declared
input space, range and output. These are the properties that follow from that
and that the rest of the pipeline silently relies on:

* an operation at its neutral setting is the identity, so a slider at zero costs
  nothing and changes nothing;
* no operation mutates its input, because the stage cache hands the same buffer
  to more than one consumer;
* the display-referred stages stay inside [0, 1], because the next stage assumes
  it;
* the tone mapping is monotonic and lands middle grey where it says it does.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.pipeline.colorspace import DISPLAY_GREY, MIDDLE_GREY
from ape.pipeline.ops import (
    color,
    exposure,
    highlight_recovery,
    local_contrast,
    noise,
    sharpen,
    tone,
    tone_curve,
    white_balance,
)
from ape.pipeline.params import (
    ColorParams,
    ExposureParams,
    HighlightRecoveryParams,
    LocalContrastParams,
    NoiseParams,
    SharpenParams,
    ToneCurveParams,
    ToneParams,
    ToneShapingParams,
    WhiteBalanceParams,
)

_SCENE_OPS = [
    (noise.apply, NoiseParams(luminance=0.0, chrominance=0.0)),
    (exposure.apply, ExposureParams()),
    (highlight_recovery.apply, HighlightRecoveryParams(strength=0.0)),
]
_DISPLAY_OPS = [
    (tone.apply_shaping, ToneShapingParams()),
    (tone_curve.apply, ToneCurveParams()),
    (color.apply, ColorParams()),
    (local_contrast.apply, LocalContrastParams()),
    (sharpen.apply, SharpenParams()),
]


@pytest.fixture
def scene() -> np.ndarray:
    rng = np.random.default_rng(5)
    base = rng.uniform(0.001, 1.4, size=(64, 96, 3)).astype(np.float32)
    return np.ascontiguousarray(base)


@pytest.fixture
def display(scene) -> np.ndarray:
    return tone.apply_sigmoid(scene, ToneParams())


@pytest.mark.parametrize("function,params", _SCENE_OPS + _DISPLAY_OPS)
def test_neutral_parameters_are_the_identity(function, params, scene, display):
    source = scene if (function, params) in _SCENE_OPS else display
    assert np.array_equal(function(source, params), source)


@pytest.mark.parametrize("function,params", _SCENE_OPS + _DISPLAY_OPS)
def test_operations_do_not_mutate_their_input(function, params, scene, display):
    source = (scene if (function, params) in _SCENE_OPS else display).copy()
    original = source.copy()
    function(source, params)
    assert np.array_equal(source, original)


def test_white_balance_at_the_as_shot_illuminant_is_the_identity(scene):
    params = WhiteBalanceParams(mode="custom", temperature_k=5200.0, tint=4.0)
    result = white_balance.apply(
        scene, params, as_shot_temperature_k=5200.0, as_shot_tint=4.0
    )
    assert np.array_equal(result, scene)


def test_raising_the_temperature_warms_the_picture(scene):
    """The direction every other raw processor uses: higher kelvin, warmer image."""
    warmer = white_balance.apply(
        scene,
        WhiteBalanceParams(mode="custom", temperature_k=8000.0),
        as_shot_temperature_k=5000.0,
        as_shot_tint=0.0,
    )
    ratio_before = float(scene[..., 0].mean() / scene[..., 2].mean())
    ratio_after = float(warmer[..., 0].mean() / warmer[..., 2].mean())
    assert ratio_after > ratio_before


def test_white_balance_preserves_the_brightness_of_a_neutral():
    """Moving the temperature must not double as an exposure change."""
    from ape.pipeline.colorspace import luminance

    grey = np.full((8, 8, 3), 0.4, dtype=np.float32)
    for kelvin in (2800.0, 4000.0, 6500.0, 9000.0):
        result = white_balance.apply(
            grey,
            WhiteBalanceParams(mode="custom", temperature_k=kelvin),
            as_shot_temperature_k=5500.0,
            as_shot_tint=0.0,
        )
        assert luminance(result).mean() == pytest.approx(luminance(grey).mean(), rel=0.02)


@pytest.mark.parametrize("function,params", _DISPLAY_OPS)
def test_display_operations_stay_inside_the_unit_range(function, params, display):
    from ape.pipeline.params import HSLBand

    pushed = {
        ToneShapingParams: ToneShapingParams(shadows=1, highlights=-1, whites=1, blacks=-1),
        ToneCurveParams: ToneCurveParams(highlights=1, lights=-1, darks=1, shadows=-1),
        ColorParams: ColorParams(
            saturation=1.0, vibrance=1.0, hsl={"red": HSLBand(saturation=1.0, luminance=1.0)}
        ),
        LocalContrastParams: LocalContrastParams(clarity=1.0),
        SharpenParams: SharpenParams(amount=3.0),
    }[type(params)]
    result = function(display, pushed)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_the_sigmoid_lands_middle_grey_where_it_claims():
    """A scene-linear 0.1845 must come out near the display's middle grey."""
    grey = np.full((4, 4, 3), MIDDLE_GREY, dtype=np.float32)
    result = tone.apply_sigmoid(grey, ToneParams())
    # The black and white points crop the curve, which shifts the pivot slightly;
    # a tenth of a stop of drift is the most that should survive.
    assert float(result.mean()) == pytest.approx(DISPLAY_GREY, abs=0.04)


@pytest.mark.parametrize("contrast", [0.5, 1.0, 1.5, 2.5])
@pytest.mark.parametrize("toe,shoulder", [(0.3, 0.3), (1.2, 1.5), (5.0, 5.0)])
def test_the_sigmoid_is_monotonic(contrast, toe, shoulder):
    params = ToneParams(contrast=contrast, toe=toe, shoulder=shoulder)
    response = tone.sigmoid_response(np.linspace(-20.0, 14.0, 2000), params)
    assert np.all(np.diff(response) >= -1e-7)


def test_the_shaping_curve_is_monotonic_at_every_extreme():
    for shadows in (-1.0, 0.0, 1.0):
        for highlights in (-1.0, 0.0, 1.0):
            for whites in (-1.0, 1.0):
                for blacks in (-1.0, 1.0):
                    lut = tone.build_shaping_lut(
                        ToneShapingParams(
                            shadows=shadows,
                            highlights=highlights,
                            whites=whites,
                            blacks=blacks,
                        )
                    )
                    assert np.all(np.diff(lut) >= -1e-7), (
                        shadows, highlights, whites, blacks
                    )


def test_the_tone_curve_is_monotonic_at_every_extreme():
    for values in ((1, 1, 1, 1), (-1, -1, -1, -1), (1, -1, 1, -1), (-1, 1, -1, 1)):
        lut = tone_curve.build_lut(
            ToneCurveParams(
                highlights=values[0], lights=values[1], darks=values[2], shadows=values[3]
            )
        )
        assert np.all(np.diff(lut) >= -1e-7), values


def test_a_user_spline_cannot_fold_back_on_itself():
    """PCHIP plus a running maximum: even a non-monotonic input stays usable."""
    lut = tone_curve.build_lut(
        ToneCurveParams(points=[(0.1, 0.8), (0.4, 0.2), (0.9, 0.95)])
    )
    assert np.all(np.diff(lut) >= -1e-7)


def test_highlight_recovery_does_not_darken_anything(scene):
    """Recovery rebuilds clipped channels; it must never take light away."""
    recovered = highlight_recovery.apply(scene, HighlightRecoveryParams(strength=1.0))
    from ape.pipeline.colorspace import luminance

    assert float(luminance(recovered).min()) >= float(luminance(scene).min()) - 1e-6


def test_highlight_recovery_neutralises_a_clipped_channel():
    """A blown sky must not come out cyan."""
    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[..., 0] = 1.0  # red clipped
    image[..., 1] = 0.99
    image[..., 2] = 0.985
    recovered = highlight_recovery.apply(image, HighlightRecoveryParams(strength=1.0))
    spread_before = float(image.max(axis=-1).mean() - image.min(axis=-1).mean())
    spread_after = float(recovered.max(axis=-1).mean() - recovered.min(axis=-1).mean())
    assert spread_after < spread_before
