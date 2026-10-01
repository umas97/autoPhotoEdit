# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared fixtures: synthetic scenes and a synthetic camera.

Phase 1 has to be testable without a single RAW file on disk, because the
acceptance criteria are about the pipeline's mathematics and those do not need a
real sensor to be checked. A synthetic camera -- a plausible colour matrix and a
chosen illuminant -- exercises exactly the same code path a real ARW takes
through ``raw/decode.py``, with the advantage that the correct answer is known
in advance.

Tests that genuinely need a real file are marked ``fixtures`` and skip when
``tests/fixtures/`` is empty. See ``tests/fixtures/README.md``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from ape.pipeline.colorspace import (
    REC2020_TO_XYZ,
    XYZ_TO_REC2020,
    apply_matrix,
)
from ape.raw.decode import CameraColor, DecodedRaw
from ape.raw.whitepoint import illuminant_xyz, normalise_camera_matrix, xyz_to_cct_tint

#: The catalogue fixtures live in their own module: this file is about the
#: pipeline, and mixing a synthetic sensor with a synthetic memory card in one
#: conftest makes both harder to find.
pytest_plugins = ["conftest_catalog", "merge_bracketing"]

FIXTURES = Path(__file__).parent / "fixtures"

#: The working space's white point, exported so tests can shoot under it exactly.
D65_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

_D65_XYZ = D65_XYZ


def available_raws() -> list[Path]:
    """Real ARW files the user has dropped into ``tests/fixtures/``."""
    return sorted(p for p in FIXTURES.glob("*") if p.suffix.lower() == ".arw")


@pytest.fixture(scope="session")
def raw_fixtures() -> list[Path]:
    files = available_raws()
    if not files:
        pytest.skip("nessun file ARW in tests/fixtures/ (vedi il README lì dentro)")
    return files


@pytest.fixture(scope="session")
def synthetic_camera() -> np.ndarray:
    """A plausible XYZ -> camera matrix for a synthetic body.

    Built by perturbing the sRGB primaries rather than copying a real camera's
    published coefficients: the point is to have a matrix that is not the
    identity and not diagonal, so that a bug in the normalisation or in the
    inversion actually shows up.
    """
    base = np.asarray(XYZ_TO_REC2020, dtype=np.float64)
    crosstalk = np.array(
        [
            [1.00, 0.06, -0.04],
            [-0.05, 1.00, 0.03],
            [0.02, -0.07, 1.00],
        ],
        dtype=np.float64,
    )
    return crosstalk @ base


def simulate_capture(
    xyz_scene: np.ndarray,
    cam_from_xyz: np.ndarray,
    illuminant_cct: float = 6504.0,
    illuminant_tint: float = 0.0,
    illuminant_override: np.ndarray | None = None,
) -> DecodedRaw:
    """Push an XYZ scene through a camera and back out of ``decode_linear``'s maths.

    Reproduces, step for step, what happens to a real file: the sensor sees
    ``cam_from_xyz @ XYZ``, the camera's white balance multipliers neutralise the
    illuminant, and the decoder renormalises the matrix on D65 and inverts it.
    Anything this fixture gets wrong, the real decoder gets wrong too.

    ``illuminant_override`` shoots under an exact XYZ rather than a point on the
    Planckian locus. It matters for the colour-reproduction test: D65 sits
    slightly off the locus, so "6504 K, Duv 0" is a *different* light from D65
    and the camera matrix, which is normalised on D65, pays a small adaptation
    error for the difference.
    """
    illuminant = (
        np.asarray(illuminant_override, dtype=np.float64)
        if illuminant_override is not None
        else illuminant_xyz(illuminant_cct, illuminant_tint)
    )
    exact = 1.0 / (cam_from_xyz @ illuminant)
    # Cameras record the multipliers relative to green; LibRaw then rescales the
    # data so the white level lands on 1.0, which cancels that normalisation
    # again. Using the unnormalised multipliers here reproduces the combination
    # in one step, while the camera context still carries the green-relative
    # numbers a real file would report.
    multipliers = exact / exact[1]

    sensor = np.einsum("ij,...j->...i", cam_from_xyz, xyz_scene.astype(np.float64))
    balanced = sensor * exact

    xyz_from_cam = np.linalg.inv(normalise_camera_matrix(cam_from_xyz, _D65_XYZ))
    rgb = apply_matrix(
        balanced.astype(np.float32), (XYZ_TO_REC2020 @ xyz_from_cam).astype(np.float32)
    )

    temperature, tint = xyz_to_cct_tint(illuminant)
    camera = CameraColor(
        cam_from_xyz=cam_from_xyz,
        xyz_from_cam=xyz_from_cam,
        as_shot_multipliers=multipliers,
        as_shot_temperature_k=temperature,
        as_shot_tint=tint,
    )
    return DecodedRaw(rgb=rgb, camera=camera, baseline_exposure_ev=0.0)


def colour_checker_xyz(illuminant_cct: float = 6504.0, scale: float = 0.9) -> np.ndarray:
    """A 24-patch ColorChecker as an XYZ image, lit by the given illuminant.

    The published chromaticities are measured under the chart's own illuminant.
    Dividing them by that illuminant's XYZ gives a per-channel reflectance, and
    multiplying by another illuminant re-lights the chart -- a von Kries
    approximation, exact for the neutral column, which is the column the
    neutrality test actually reads.

    Patches are 32x32 so every spatial filter in the pipeline has room to settle
    well away from the edges.
    """
    import colour

    chart = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    reference = np.asarray(colour.xy_to_XYZ(np.asarray(chart.illuminant)), dtype=np.float64)
    illuminant = illuminant_xyz(illuminant_cct)

    patch = 32
    rows, columns = 4, 6
    image = np.zeros((rows * patch, columns * patch, 3), dtype=np.float64)
    for index, xy_y in enumerate(chart.data.values()):
        xyz = np.asarray(colour.xyY_to_XYZ(np.asarray(xy_y)), dtype=np.float64)
        reflectance = xyz / reference
        row, column = divmod(index, columns)
        image[row * patch : (row + 1) * patch, column * patch : (column + 1) * patch] = (
            reflectance * illuminant * scale
        )
    return image


#: Index of the six neutral patches, bottom row of a ColorChecker24.
NEUTRAL_PATCHES = tuple(range(18, 24))


def patch_centre(image: np.ndarray, index: int, patch: int = 32, columns: int = 6) -> np.ndarray:
    """Mean of the central quarter of one ColorChecker patch."""
    row, column = divmod(index, columns)
    margin = patch // 4
    block = image[
        row * patch + margin : (row + 1) * patch - margin,
        column * patch + margin : (column + 1) * patch - margin,
    ]
    return block.reshape(-1, block.shape[-1]).mean(axis=0)


#: Long edge the scenes are built at by default. Anything that only needs
#: *some* image uses this; the resolution-invariance test asks for its own size.
SCENE_EDGE = 576


def scene_shape(long_edge: int) -> tuple[int, int]:
    """Height and width for a 3:2-ish frame whose height is always even.

    Even height matters: the resolution-invariance test derives its preview by
    halving the full-size scene, and an odd dimension would make that halving a
    resampling with a fractional factor instead of an exact box average.
    """
    return 2 * int(round(long_edge / 3.0)), int(long_edge)


#: How many scenes there are. Test 1 of section 13 asks for ten.
SCENE_COUNT = 10


def _grid(long_edge: int) -> tuple[np.ndarray, ...]:
    height, width = scene_shape(long_edge)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    return x, y, x / (width - 1), y / (height - 1)


@lru_cache(maxsize=4)
def build_scene(index: int, long_edge: int = SCENE_EDGE) -> np.ndarray:
    """One linear Rec.2020 scene, built at the requested size.

    The ten scenes between them cover flat gradients, saturated colour, deep
    shadow, blown highlight and fine high-frequency detail -- the four things
    that break a tone mapping in different ways, plus the texture that breaks a
    spatial filter.

    Every feature is sized as a fraction of the frame, so the same scene at two
    resolutions is the same picture and not merely a similar one. That is what
    lets the invariance test compare a 2048 px render with a 1024 px one without
    measuring, instead, the difference between two interpolations of a small
    bitmap. One scene at a time, and seeded per scene, because at full size ten
    of them at once in every worker process is a gigabyte that buys nothing.
    """
    height, width = scene_shape(long_edge)
    x, y, u, v = _grid(long_edge)

    if index == 0:
        # Horizontal luminance ramp over eight stops.
        ramp = (0.005 * np.power(2.0, u * 8.0)).astype(np.float32)
        scene = np.repeat(ramp[..., None], 3, axis=-1)

    elif index == 1:
        # Saturated hue sweep at constant luminance: the hue-stability case.
        hue = np.stack(
            [np.cos(2 * np.pi * (u - phase)) for phase in (0.0, 1 / 3, 2 / 3)], axis=-1
        )
        scene = (0.25 * (1.0 + 0.85 * hue)).astype(np.float32)

    elif index == 2:
        # Deep shadows with a small bright window: tests the toe.
        scene = np.full((height, width, 3), 0.004, dtype=np.float32)
        scene[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 0.6

    elif index == 3:
        # Blown highlights with coloured clipping: the recovery case.
        blown = np.stack(
            [np.full_like(u, 1.4), np.full_like(u, 1.05), np.full_like(u, 0.45)], axis=-1
        )
        blown[: height // 2] *= 0.18
        scene = blown.astype(np.float32)

    elif index == 4:
        # Checkerboard: the spatial-filter case. The block is a fixed fraction of
        # the frame and stays several pixels wide even at preview size -- a
        # pattern at the sampling limit would only measure how the two
        # resolutions alias it, which is a property of the pattern, not of us.
        block = max(2, long_edge // 128)
        checker = (((x // block).astype(int) + (y // block).astype(int)) % 2).astype(np.float32)
        scene = np.repeat((0.05 + 0.5 * checker)[..., None], 3, axis=-1)

    elif 5 <= index < SCENE_COUNT:
        # Noisy, tinted, randomised natural-looking mixes. The noise is shot
        # noise -- standard deviation proportional to the square root of the
        # signal -- because that is what a sensor produces and what noise.py is
        # built around. A constant-sigma noise would be several stops above the
        # signal in the shadows, and at that point the test would be measuring
        # how a nonlinearity and an average fail to commute on a pathological
        # input rather than whether the pipeline is scale invariant.
        step = index - 5
        rng = np.random.default_rng((20240918, index))
        base = 0.02 + 0.55 * np.power(v, 1.0 + 0.5 * step)
        tint = np.array([1.0 + 0.2 * step * 0.1, 1.0, 1.3 - 0.1 * step], dtype=np.float32)
        signal = (base[..., None] * tint).astype(np.float32)
        noise = rng.normal(0.0, 1.0, size=(height, width, 3)).astype(np.float32)
        scene = np.clip(signal + noise * np.float32(0.02) * np.sqrt(signal), 1e-4, None)

    else:
        raise IndexError(f"scena {index}: ne esistono {SCENE_COUNT}")

    return np.ascontiguousarray(scene, dtype=np.float32)


def build_synthetic_scenes(long_edge: int = SCENE_EDGE) -> tuple[np.ndarray, ...]:
    """All ten scenes. Convenient for the tests that just want some pixels."""
    return tuple(build_scene(index, long_edge) for index in range(SCENE_COUNT))


@pytest.fixture(scope="session")
def synthetic_scenes() -> list[np.ndarray]:
    """The scenes above, as a fixture. Built by a plain cached function so that
    worker processes can rebuild them identically without a pytest session."""
    return list(build_synthetic_scenes())


def as_decoded(scene_rec2020: np.ndarray) -> DecodedRaw:
    """Wrap a linear Rec.2020 scene as a decoded frame with an identity camera."""
    from ape.raw.decode import decoded_from_array

    return decoded_from_array(scene_rec2020, temperature_k=6504.0, tint=0.0)


def rec2020_from_xyz(xyz: np.ndarray) -> np.ndarray:
    return apply_matrix(xyz.astype(np.float32), XYZ_TO_REC2020.astype(np.float32))


def xyz_from_rec2020(rgb: np.ndarray) -> np.ndarray:
    return apply_matrix(rgb.astype(np.float32), REC2020_TO_XYZ.astype(np.float32))


def neutral_grey_xyz(illuminant_cct: float = 6504.0, scale: float = 0.9) -> np.ndarray:
    """A chart of six perfectly neutral patches under the given illuminant.

    A real ColorChecker's grey column is not spectrally flat -- its patches carry
    a measured chromaticity of their own -- so it cannot be used to answer "does
    the pipeline keep neutrals neutral?" without conflating the pipeline's error
    with the chart's. This chart is flat by construction, so any chroma in the
    output is ours.
    """
    illuminant = illuminant_xyz(illuminant_cct)
    reflectances = (0.90, 0.59, 0.36, 0.19, 0.09, 0.03)

    patch = 32
    image = np.zeros((patch, patch * len(reflectances), 3), dtype=np.float64)
    for index, reflectance in enumerate(reflectances):
        image[:, index * patch : (index + 1) * patch] = illuminant * reflectance * scale
    return image
