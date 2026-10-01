# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic straightening (section 6.4) and the acceptance of phase 5.

"Su 50 foto di test il raddrizzamento non peggiora mai un orizzonte già
corretto." The user's 24 photos are mostly forest, snow, logs and portraits --
almost nothing in them is supposed to be level -- so the 50 are those 24 plus
26 synthetic scenes built so that the right answer is known: seascapes and
facades, level and tilted, with and without perspective, and pure texture.

Two properties are checked, in this order of importance:

1. **Never worse.** On a level frame the detector does nothing; on any frame,
   the residual tilt after its correction is never larger than before.
2. **Useful.** On a frame with a clear horizon or clear verticals, a tilt
   between 0.5 and 8 degrees is found to within a quarter of a degree.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from ape.analysis import straighten
from ape.pipeline import geometry
from ape.pipeline.params import GeometryParams


def _rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    return geometry.apply(img, GeometryParams(rotation_deg=degrees))


def _seascape(seed: int = 0, width: int = 1500, height: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.empty((height, width, 3), dtype=np.float32)
    horizon = int(height * (0.35 + 0.3 * rng.random()))
    ramp = np.linspace(0.85, 0.55, horizon, dtype=np.float32)[:, None]
    img[:horizon] = np.stack([ramp * 0.6, ramp * 0.75, ramp], axis=-1)
    sea = np.linspace(0.25, 0.4, height - horizon, dtype=np.float32)[:, None]
    img[horizon:] = np.stack([sea * 0.3, sea * 0.5, sea * 0.8], axis=-1)
    # Waves: short, bright, nearly horizontal strokes of random slope.
    for _ in range(120):
        y = rng.integers(horizon + 20, height)
        x = rng.integers(0, width)
        length = rng.integers(10, 35)
        slope = rng.normal(0, 0.15)
        cv2.line(img, (x, y), (x + length, int(y + slope * length)), (0.5, 0.6, 0.7), 1)
    # A boat and a headland break the line, as they do.
    boat = ((int(width * 0.3), horizon - 30), (int(width * 0.33), horizon))
    cv2.rectangle(img, *boat, (0.1,) * 3, -1)
    cv2.ellipse(img, (width, horizon), (int(width * 0.2), 60), 0, 180, 360, (0.2, 0.25, 0.2), -1)
    noise = rng.normal(0, 0.01, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1)


def _facade(seed: int = 0, keystone: float = 0.0, width: int = 1500, height: int = 1000):
    """A wall of windows; ``keystone`` > 0 narrows the top, as a camera tilted up does."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 0.62, dtype=np.float32)
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    cols, rows = 7, 5
    for i in range(cols):
        for j in range(rows):
            x0 = int(width * (0.06 + i * 0.13))
            y0 = int(height * (0.08 + j * 0.18))
            cv2.rectangle(img, (x0, y0), (x0 + int(width * 0.07), y0 + int(height * 0.11)),
                          (0.15, 0.17, 0.2), -1)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0.3, 0.3, 0.3), 12)
    if keystone:
        shift = keystone * width
        src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst = np.float32([[shift, 0], [width - shift, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return np.clip(img, 0, 1)


def _texture(seed: int = 0, width: int = 1500, height: int = 1000) -> np.ndarray:
    """Grass and bark: plenty of edges, none of them structure."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((height, width, 3)).astype(np.float32), (0, 0), 3)
    for _ in range(400):
        x, y = rng.integers(0, width), rng.integers(0, height)
        angle = rng.normal(90, 25)
        length = rng.integers(15, 60)
        dx = length * math.cos(math.radians(angle))
        dy = -length * math.sin(math.radians(angle))
        cv2.line(img, (x, y), (int(x + dx), int(y + dy)), (0.1, 0.4, 0.1), 2)
    return img


def _residual(true_tilt: float, result: straighten.StraightenResult) -> float:
    """Tilt left in the frame after applying the detector's correction."""
    return abs(true_tilt + result.rotation_deg)


SYNTHETIC = (
    [("seascape", seed, tilt) for seed, tilt in enumerate([0.0, 0.0, 0.8, -1.7, 3.0, -5.5, 7.0])]
    + [("facade", seed, tilt) for seed, tilt in enumerate([0.0, 0.0, 0.6, -2.2, 4.5, -6.0])]
    + [("keystone", seed, tilt) for seed, tilt in enumerate([0.0, 0.0, 1.2, -3.0, 5.0])]
    + [("texture", seed, tilt) for seed, tilt in enumerate([0.0, 0.0, 2.0, -4.0])]
    + [("seascape", 10 + seed, tilt) for seed, tilt in enumerate([11.0, -14.0])]
    + [("facade", 10 + seed, tilt) for seed, tilt in enumerate([0.1, -0.1])]
)
assert len(SYNTHETIC) == 26


def _scene(kind: str, seed: int) -> np.ndarray:
    if kind == "seascape":
        return _seascape(seed)
    if kind == "facade":
        return _facade(seed)
    if kind == "keystone":
        return _facade(seed, keystone=0.06)
    return _texture(seed)


@pytest.mark.parametrize(("kind", "seed", "tilt"), SYNTHETIC)
def test_synthetic_scene_is_never_made_worse(kind, seed, tilt):
    result = straighten.estimate(_rotate(_scene(kind, seed), tilt))
    assert _residual(tilt, result) <= abs(tilt) + 1e-9
    if abs(tilt) < straighten.MIN_ROTATION_DEG:
        assert result.rotation_deg == 0.0, result.as_json()
    if kind == "texture":
        assert result.rotation_deg == 0.0, result.as_json()
    if abs(tilt) > straighten.MAX_ROTATION_DEG:
        assert result.rotation_deg == 0.0


@pytest.mark.parametrize(
    ("kind", "seed", "tilt"),
    [s for s in SYNTHETIC if s[0] != "texture" and 0.5 <= abs(s[2]) <= 8.0],
)
def test_clear_structure_is_levelled(kind, seed, tilt):
    result = straighten.estimate(_rotate(_scene(kind, seed), tilt))
    assert result.outcome == "rotated", result.as_json()
    assert _residual(tilt, result) < 0.25, result.as_json()


def test_result_is_serialisable_for_the_catalogue():
    import json

    result = straighten.estimate(_rotate(_seascape(1), 2.0))
    data = json.loads(json.dumps(result.as_json()))
    assert data["outcome"] == "rotated"
    assert set(data["families"]) >= {"horizon"}


@pytest.mark.fixtures
def test_real_photos_are_never_made_worse(raw_fixtures):
    """The user's 24 frames, as the analysis sees them: the lens-corrected proxy.

    None of them has a visibly crooked horizon, so the correction applied to
    each must stay below half a degree -- anything more would be the detector
    inventing a tilt. And rotated by a known angle, a frame the detector reads
    must be read consistently: the angle it finds minus the one added must stay
    within half a degree of what it found on the original.
    """
    from ape.pipeline.params import neutral_params
    from ape.pipeline.render import RenderOptions, render
    from ape.raw.proxy import editing_proxy

    for path in raw_fixtures:
        proxy = render(editing_proxy(path, long_edge=1024), neutral_params(),
                       RenderOptions(long_edge=1024))
        original = straighten.estimate(proxy)
        assert abs(original.rotation_deg) <= 0.5, (path.name, original.as_json())
        base = original.measured_deg if original.outcome in ("rotated", "level") else None
        for added in (-3.0, 1.5):
            turned = straighten.estimate(_rotate(proxy, added))
            if turned.outcome != "rotated":
                continue
            reference = base if base is not None else 0.0
            assert abs(turned.measured_deg - added - reference) <= 0.5 + 1e-9, (
                path.name, added, turned.as_json()
            )


@pytest.mark.fixtures
@pytest.mark.parametrize("name", ["DSC05635", "DSC05636"])
def test_ground_lines_never_become_a_horizon(raw_fixtures, tmp_path, name):
    """Two straight lines on the ground that must stay what they are.

    05635's far edge of the frozen pond leans 1.7 degrees in perspective while
    the poles and benches stand straight; the snow ridge on the ice of 05636 is
    long and straight but has the same ice on both sides. Checked on the proxy
    exactly as the analysis job reads it -- 2048 px JPEG -- at the qualities
    around the one in use, because the pieces LSD finds along a faint line
    change with the compression: a change that "stabilised" the far shore of
    05634 across qualities turned the edge of 05635 into a horizon at three of
    four.
    """
    from PIL import Image

    from ape.export.image import ExportFormat, save_image
    from ape.pipeline.params import neutral_params
    from ape.pipeline.render import RenderOptions, render
    from ape.raw.proxy import editing_proxy

    path = next((p for p in raw_fixtures if p.stem == name), None)
    if path is None:
        pytest.skip(f"{name} non è tra i file di tests/fixtures/")
    image = render(editing_proxy(path, long_edge=2048), neutral_params(),
                   RenderOptions(long_edge=2048))
    for quality in (85, 90, 92, 95):
        destination = tmp_path / f"{quality}.jpg"
        save_image(image, destination, ExportFormat.JPEG, quality=quality)
        with Image.open(destination) as jpeg:
            proxy = np.asarray(jpeg.convert("RGB"))
        result = straighten.estimate(proxy)
        assert "horizon" not in result.families, (name, quality, result.as_json())
        assert abs(result.rotation_deg) <= 0.5, (name, quality, result.as_json())
