# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 4 of docs/SPEC.md section 13: determinism.

"The same photo with the same parameters produces identical bytes."

Identical bytes, not similar pixels: the export cache, the resume of an
interrupted batch and any future content-addressed storage all depend on it, and
so does the user's ability to tell whether anything actually changed.

The usual sources of drift are a timestamp in a header (the ICC profile writes a
fixed one for exactly this reason), a dictionary iterating in a different order,
and a filter seeded from the clock. All three are checked here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.export.icc import build_profile
from ape.export.image import ExportFormat, encode_image
from ape.pipeline.colorspace import OutputSpace
from ape.pipeline.params import EditParams
from ape.pipeline.render import RenderOptions, StageRenderer, render
from conftest import as_decoded

_PARAMS = EditParams.model_validate(
    {
        "exposure": {"ev": 0.4},
        "noise": {"luminance": 0.4, "chrominance": 0.7},
        "tone": {"contrast": 1.35, "chroma_preservation": 0.5},
        "tone_shaping": {"shadows": 0.3, "highlights": -0.4},
        "tone_curve": {"points": [(0.0, 0.0), (0.4, 0.45), (1.0, 1.0)]},
        "color": {"saturation": 0.2, "vibrance": 0.3,
                  "hsl": {"blue": {"saturation": 0.4}, "orange": {"hue": -0.2}},
                  "split_toning": {"shadow_hue": 220.0, "shadow_saturation": 0.2}},
        "local_contrast": {"clarity": 0.5},
        "sharpen": {"amount": 0.9},
    }
)


def test_render_is_bit_identical_across_calls(synthetic_scenes):
    for scene in synthetic_scenes[:4]:
        first = render(as_decoded(scene), _PARAMS)
        second = render(as_decoded(scene), _PARAMS)
        assert np.array_equal(first, second)


def test_render_is_bit_identical_from_a_reloaded_params_object(synthetic_scenes):
    """Serialisation must not perturb a single float."""
    reloaded = EditParams.from_json(_PARAMS.to_json())
    scene = synthetic_scenes[5]
    assert np.array_equal(render(as_decoded(scene), _PARAMS), render(as_decoded(scene), reloaded))


@pytest.mark.parametrize(
    "fmt", [ExportFormat.JPEG, ExportFormat.TIFF8, ExportFormat.TIFF16]
)
def test_encoded_files_are_byte_identical(synthetic_scenes, fmt):
    image = render(as_decoded(synthetic_scenes[7]), _PARAMS)
    assert encode_image(image, fmt) == encode_image(image, fmt)


def test_icc_profiles_are_byte_identical():
    """A clock in the profile header would leak into every exported file."""
    for space in OutputSpace:
        assert build_profile(space) == build_profile(space)


def test_the_stage_cache_returns_what_a_fresh_render_returns(synthetic_scenes):
    """The cached path and the stateless path must not diverge.

    Exercised by walking a slider: each step changes one late-stage parameter,
    which is exactly the case the cache is built for and the case where a stale
    entry would go unnoticed.
    """
    scene = synthetic_scenes[8]
    decoded = as_decoded(scene)
    cached = StageRenderer(decoded=decoded)
    options = RenderOptions()

    for saturation in (0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0):
        params = _PARAMS.model_copy(deep=True)
        params.color.saturation = saturation
        assert np.array_equal(cached.render(params, options), render(decoded, params, options))


def test_the_stage_cache_notices_an_early_change(synthetic_scenes):
    """Changing exposure must invalidate everything downstream of it."""
    decoded = as_decoded(synthetic_scenes[9])
    cached = StageRenderer(decoded=decoded)
    options = RenderOptions()

    for ev in (0.0, 1.0, -1.0, 0.0):
        params = _PARAMS.model_copy(deep=True)
        params.exposure.ev = ev
        assert np.array_equal(cached.render(params, options), render(decoded, params, options))
