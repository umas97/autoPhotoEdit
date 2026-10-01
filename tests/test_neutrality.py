# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 5 of docs/SPEC.md section 13: neutrality.

"A colour checker RAW developed with neutral parameters and a correct white
balance must give the grey patches a*, b* within +/-2."

Split into three assertions, because the one sentence covers three different
claims that fail for different reasons:

1. **Neutrals stay neutral.** A spectrally flat grey chart must come out with no
   chroma at all. The tolerance here is 0.5, not 2: the maths says the answer is
   exactly zero, so anything bigger is a bug, not a tolerance.
2. **Neutrals stay neutral under any light.** Same chart under tungsten. This is
   what actually tests the white balance path; under D65 a broken adaptation
   would pass by accident.
3. **Colours are reproduced.** A real ColorChecker pushed through the camera
   must land where the same chart lands without one, within 2 dE -- which is the
   claim the spec is really making about the camera matrix. The comparison is
   render against render, not render against scene: the tone mapping moves
   lightness and chroma by design, and comparing across it would measure the
   tone curve rather than the colour path.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.pipeline.colorspace import display_decode, to_lab
from ape.pipeline.params import EditParams, neutral_params
from ape.pipeline.render import RenderOptions, render
from conftest import (
    D65_XYZ,
    colour_checker_xyz,
    neutral_grey_xyz,
    patch_centre,
    simulate_capture,
)

#: Neutrality is exact arithmetic; this is a guard against drift, not a tolerance.
NEUTRAL_TOLERANCE = 0.5
#: The spec's own number, used where a real chart's own chromaticity is in play.
CHART_TOLERANCE = 2.0

# Nothing in this test may push the picture around: a tone curve or a saturation
# offset would make "neutral in, neutral out" meaningless.
_FLAT = EditParams.model_validate(
    {
        **neutral_params().model_dump(),
        "tone": {"contrast": 1.0, "chroma_preservation": 0.0},
        "highlight_recovery": {"strength": 0.0},
        "noise": {"luminance": 0.0, "chrominance": 0.0},
    }
)

_OPTIONS = RenderOptions(stop_before_output=True)


def _lab_of(decoded, params=_FLAT):
    """Render and convert to Lab, staying in the working space throughout."""
    rendered = render(decoded, params, _OPTIONS)
    return to_lab(display_decode(rendered))


@pytest.mark.parametrize("cct", [6504.0, 2856.0, 4000.0])
def test_flat_grey_stays_neutral(synthetic_camera, cct):
    """A spectrally flat chart comes out with no chroma, whatever the light."""
    scene = neutral_grey_xyz(illuminant_cct=cct)
    decoded = simulate_capture(scene, synthetic_camera, illuminant_cct=cct)
    lab = _lab_of(decoded)

    worst = 0.0
    for index in range(6):
        _lightness, a_star, b_star = patch_centre(lab, index, columns=6)
        worst = max(worst, abs(a_star), abs(b_star))
        assert abs(a_star) < NEUTRAL_TOLERANCE, f"patch {index} a*={a_star:.3f} a {cct:.0f} K"
        assert abs(b_star) < NEUTRAL_TOLERANCE, f"patch {index} b*={b_star:.3f} a {cct:.0f} K"
    assert worst < NEUTRAL_TOLERANCE


def test_grey_lightness_is_monotonic(synthetic_camera):
    """The grey ramp keeps its order: a tone curve that folds would show up here."""
    scene = neutral_grey_xyz()
    decoded = simulate_capture(scene, synthetic_camera)
    lab = _lab_of(decoded)
    lightness = [patch_centre(lab, index, columns=6)[0] for index in range(6)]
    assert all(a > b for a, b in zip(lightness, lightness[1:], strict=False)), lightness


def test_colour_checker_is_reproduced(synthetic_camera):
    """Under the light the camera matrix was built for, colour comes back intact.

    The whole chain is exercised -- camera matrix, white balance multipliers,
    renormalisation, inversion, working space -- against a reference computed
    directly from the scene's XYZ. Only the tone mapping stands between the two,
    and it is applied to the reference as well.
    """
    scene = colour_checker_xyz()
    # Shot under D65 exactly, the illuminant the camera matrix is normalised on.
    decoded = simulate_capture(scene, synthetic_camera, illuminant_override=D65_XYZ)

    from ape.pipeline.colorspace import XYZ_TO_REC2020, apply_matrix
    from ape.raw.decode import decoded_from_array

    reference = decoded_from_array(
        apply_matrix(scene.astype(np.float32), XYZ_TO_REC2020.astype(np.float32))
    )
    reference.baseline_exposure_ev = 0.0

    lab_actual = _lab_of(decoded)
    lab_expected = _lab_of(reference)

    from ape.pipeline.colorspace import delta_e_2000

    differences = [
        float(
            delta_e_2000(
                patch_centre(lab_actual, index)[None, :],
                patch_centre(lab_expected, index)[None, :],
            )[0]
        )
        for index in range(24)
    ]
    # Exact arithmetic again: under the matrix's own illuminant the camera path
    # is invertible, so the tolerance is float32 noise, not a colour tolerance.
    assert max(differences) < 0.2, (
        f"dE2000 massimo {max(differences):.3f} sulla patch {int(np.argmax(differences))}"
    )


def test_off_locus_illuminant_costs_little(synthetic_camera):
    """Shooting under a light the matrix was not built for stays inside 2 dE.

    6504 K on the Planckian locus is not D65 -- D65 sits a little above it -- so
    this measures the adaptation error a real, slightly-wrong illuminant
    estimate costs. It is the spec's +/-2 that applies here.
    """
    from ape.pipeline.colorspace import XYZ_TO_REC2020, apply_matrix, delta_e_2000
    from ape.raw.decode import decoded_from_array

    scene = colour_checker_xyz()
    decoded = simulate_capture(scene, synthetic_camera, illuminant_cct=6504.0)
    reference = decoded_from_array(
        apply_matrix(scene.astype(np.float32), XYZ_TO_REC2020.astype(np.float32))
    )
    reference.baseline_exposure_ev = 0.0

    lab_actual = _lab_of(decoded)
    lab_expected = _lab_of(reference)
    differences = [
        float(
            delta_e_2000(
                patch_centre(lab_actual, index)[None, :],
                patch_centre(lab_expected, index)[None, :],
            )[0]
        )
        for index in range(24)
    ]
    assert np.mean(differences) < CHART_TOLERANCE, f"dE2000 medio {np.mean(differences):.2f}"
