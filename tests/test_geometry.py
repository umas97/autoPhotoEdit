# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rotation and crop in the renderer (section 6.4).

What is checked: the sign convention, the minimal implicit crop (no empty
corner, nothing more lost than necessary), the crop normalised to the
straightened frame, and resolution independence -- the proxy and the export
must frame the picture identically.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from ape.pipeline import geometry
from ape.pipeline.params import CropRect, EditParams, GeometryParams


def _line_angle(img: np.ndarray) -> float:
    """Angle in degrees of the bright line in ``img``, counter-clockwise positive."""
    mask = img[..., 1] > 0.5
    ys, xs = np.nonzero(mask)
    slope = np.polyfit(xs.astype(float), ys.astype(float), 1)[0]
    return -math.degrees(math.atan(slope))  # y points down


def _tilted_line(width: int, height: int, degrees: float) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.float32)
    theta = math.radians(degrees)
    centre = (width / 2, height / 2)
    half = width  # long enough to cross the frame
    start = (centre[0] - half * math.cos(theta), centre[1] + half * math.sin(theta))
    end = (centre[0] + half * math.cos(theta), centre[1] - half * math.sin(theta))
    cv2.line(img, tuple(map(round, start)), tuple(map(round, end)), (1.0, 1.0, 1.0), 3)
    return img


@pytest.mark.parametrize("tilt", [-6.0, -1.5, 0.7, 4.0])
def test_rotation_levels_a_tilted_line(tilt):
    """A line rising by ``tilt`` degrees is levelled by ``rotation_deg = -tilt``."""
    img = _tilted_line(1200, 800, tilt)
    assert _line_angle(img) == pytest.approx(tilt, abs=0.1)
    levelled = geometry.apply(img, GeometryParams(rotation_deg=-tilt))
    assert abs(_line_angle(levelled)) < 0.1


@pytest.mark.parametrize("degrees", [0.3, 2.0, 8.0, -20.0])
def test_implicit_crop_is_minimal_and_leaves_no_empty_corner(degrees):
    """The corners of the straightened frame lie inside the rotated picture.

    Checked on the geometry directly: map the four corners of the output back
    through the inverse rotation and require them inside the source, and
    require that a slightly larger crop would not fit -- "minimal".
    """
    width, height = 6000.0, 4000.0
    k = geometry.inscribed_scale(width, height, degrees)
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)

    def corners_fit(scale: float) -> bool:
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = sx * width * scale / 2, sy * height * scale / 2
                u, v = c * x - s * y, s * x + c * y
                if abs(u) > width / 2 + 1e-6 or abs(v) > height / 2 + 1e-6:
                    return False
        return True

    assert corners_fit(k)
    assert not corners_fit(k * 1.001)


def test_rotated_constant_image_has_no_border_artefacts():
    img = np.full((600, 900, 3), 0.42, dtype=np.float32)
    out = geometry.apply(img, GeometryParams(rotation_deg=5.0))
    assert out.shape[0] < 600 and out.shape[1] < 900
    np.testing.assert_allclose(out, 0.42, atol=1e-5)
    # Same aspect ratio as the frame it came from.
    assert out.shape[1] / out.shape[0] == pytest.approx(1.5, abs=0.01)


def test_crop_without_rotation_is_an_exact_slice():
    img = np.random.default_rng(1).random((400, 600, 3)).astype(np.float32)
    crop = CropRect(x=0.25, y=0.1, width=0.5, height=0.5)
    out = geometry.apply(img, GeometryParams(crop=crop))
    np.testing.assert_array_equal(out, img[40:240, 150:450])


def test_identity_returns_the_same_array():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    assert geometry.apply(img, GeometryParams()) is img


def test_crop_is_relative_to_the_straightened_frame():
    """A crop keeps its meaning when the rotation changes underneath it."""
    img = np.zeros((800, 1200, 3), dtype=np.float32)
    crop = CropRect(x=0.0, y=0.0, width=0.5, height=1.0)
    for angle in (0.0, 3.0, -6.0):
        params = GeometryParams(rotation_deg=angle, crop=crop)
        sw, sh = geometry.straightened_size(1200, 800, angle)
        assert geometry.apply(img, params).shape[:2] == (round(sh), round(sw * 0.5))


def test_rotation_and_crop_are_resolution_independent():
    """Render at two sizes, bring the large one down: same framing (section 6.2)."""
    rng = np.random.default_rng(7)
    base = cv2.GaussianBlur(rng.random((1600, 2400, 3)).astype(np.float32), (0, 0), 6)
    params = GeometryParams(rotation_deg=3.5, crop=CropRect(x=0.1, y=0.2, width=0.6, height=0.5))
    large = geometry.apply(base, params)
    small = geometry.apply(cv2.resize(base, (600, 400), interpolation=cv2.INTER_AREA), params)
    reduced = cv2.resize(large, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    assert float(np.abs(reduced - small)[4:-4, 4:-4].mean()) < 4e-3


def test_renderer_applies_geometry_instead_of_refusing_it():
    from ape.pipeline.render import render
    from ape.raw.decode import decoded_from_array

    params = EditParams.model_validate(
        {"geometry": {"rotation_deg": 2.0, "crop": {"x": 0, "y": 0, "width": 0.5, "height": 0.5}}}
    )
    decoded = decoded_from_array(np.full((80, 120, 3), 0.2, dtype=np.float32))
    out = render(decoded, params)
    expected = geometry.output_size(120, 80, params.geometry)
    assert (out.shape[1], out.shape[0]) == expected
