# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focus stacking and its coverage map (section 25.4).

The stack is synthetic and built from a real A7 III frame: three frames, each
sharp over a third of the picture and increasingly blurred away from it, with
the focus breathing of a real lens -- a change of scale of about one per cent
per frame -- and a small shift. The truth is the frame itself, sharp
everywhere, so what the merge recovers can be measured.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ape.merge.coverage import coverage_map
from ape.merge.engine import Member, Recipe
from ape.merge.errors import MergeFailure
from ape.merge.focus_stack import run
from ape.merge.pyramid import fuse_stack
from ape.merge.warp import shrink
from ape.raw.decode import decode_linear

#: Magnification and shift of each frame relative to the middle one.
_BREATHING = [(0.99, (3.0, -2.0)), (1.0, (0.0, 0.0)), (1.011, (-2.0, 1.5))]


@pytest.fixture(scope="module")
def scene(raw_fixtures) -> tuple[np.ndarray, object]:
    decoded = decode_linear(raw_fixtures[10], half_size=True)  # DSC05628, a woodland
    small, _ = shrink(decoded.rgb, 1200)
    return np.ascontiguousarray(small, dtype=np.float32), decoded


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma)


def _defocus(image: np.ndarray, sigma_map: np.ndarray) -> np.ndarray:
    """Blur that varies over the picture, blended from a few fixed widths."""
    out = image.copy()
    for sigma in (1, 2, 3, 4):
        weight = np.clip(sigma_map - (sigma - 1), 0, 1)[..., None]
        out = out * (1 - weight) + _blur(image, sigma) * weight
    return out.astype(np.float32)


def _stack(truth: np.ndarray, *, soft_top: float = 0.0, moved: bool = True) -> list[np.ndarray]:
    """Frame k sharp around x = (2k + 1) / 6; the top quarter ``soft_top`` everywhere."""
    height, width = truth.shape[:2]
    xs = np.linspace(0, 1, width, dtype=np.float32)[None, :].repeat(height, 0)
    frames = []
    for k, (magnification, (dx, dy)) in enumerate(_BREATHING):
        sigma = np.clip((np.abs(xs - (2 * k + 1) / 6) - 1 / 6) * 40, 0, 4)
        if soft_top:
            sigma[: height // 4] = soft_top
        frame = _defocus(truth, sigma)
        if moved and k != 1:
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0.0, magnification)
            matrix[:, 2] += (dx, dy)
            frame = cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REFLECT)
        frames.append(frame)
    return frames


def _detail(image: np.ndarray) -> float:
    green = image[..., 1]
    return float(np.mean(np.abs(green - _blur(green, 1.0))))


def _recipe() -> Recipe:
    members = tuple(
        Member(photo_id=i, path=f"/card/DSC0000{i}.ARW", filename=f"DSC0000{i}.ARW",
               hash=f"h{i}", ev_offset=None, reference=i == 1)
        for i in range(3)
    )
    return Recipe(group_id=1, kind="focus_stack", members=members)


def _decoder(frames: list[np.ndarray], like):
    def decode(member, *, preview, white_balance=None):
        return SimpleNamespace(rgb=frames[member.photo_id].copy(), camera=like.camera,
                               baseline_exposure_ev=like.baseline_exposure_ev)
    return decode


def _resampled_twice(truth: np.ndarray) -> np.ndarray:
    """The truth through the two resamplings a moved frame goes through here."""
    height, width = truth.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0.0, _BREATHING[2][0])
    matrix[:, 2] += _BREATHING[2][1]
    moved = cv2.warpAffine(truth, matrix, (width, height), flags=cv2.INTER_CUBIC)
    return cv2.warpAffine(moved, cv2.invertAffineTransform(matrix), (width, height),
                          flags=cv2.INTER_CUBIC)


def test_the_stack_is_sharp_wherever_some_frame_was(scene):
    truth, like = scene
    frames = _stack(truth)
    outcome = run(_recipe(), preview=True, progress=None, decode=_decoder(frames, like))
    merged = outcome.decoded.rgb
    inner = (slice(40, -40), slice(40, -40))  # the edges only the middle frame covers
    # Two thirds of the result come from frames resampled twice, which no
    # merge can make sharper than the truth resampled twice.
    ceiling = (_detail(truth[inner]) + 2 * _detail(_resampled_twice(truth)[inner])) / 3
    recovered = _detail(merged[inner]) / ceiling
    best_single = max(_detail(f[inner]) for f in frames) / ceiling
    error = float(np.mean(np.abs(merged[inner] - truth[inner])) / np.mean(truth[inner]))
    assert recovered > 0.9, recovered
    assert best_single < 0.6, best_single  # no frame alone comes close
    assert error < 0.03, error
    report = outcome.report
    scales = [f.get("scale") for f in report["frames"]]
    assert scales == pytest.approx([0.99, 1.0, 1.011], abs=0.002)
    assert not report["misaligned"] and report["residual_px"] < 1.0
    # What little is flagged is the left and right edge, which the magnified
    # frames do not reach and the reference has out of focus: true.
    assert report["uncovered_fraction"] < 0.02


def test_where_no_frame_is_in_focus_the_map_says_so(scene):
    truth, like = scene
    frames = _stack(truth, soft_top=3.0)
    outcome = run(_recipe(), preview=True, progress=None, decode=_decoder(frames, like))
    overlay = outcome.report["coverage_image"]
    assert overlay.dtype == np.uint8 and overlay.shape[2] == 4
    flagged = overlay[..., 3] > 0
    quarter = flagged.shape[0] // 4
    assert flagged[: quarter - 20].mean() > 0.3  # the soft band, where it has edges to measure
    assert flagged[quarter + 20 :].mean() < 0.02  # and not the rest
    assert outcome.report["uncovered_fraction"] > 0.05


def test_a_sharp_stack_raises_no_flags(scene):
    truth, _ = scene
    fraction, overlay = coverage_map([truth, _blur(truth, 3.0)])
    assert fraction < 0.01 and not overlay[..., 3].any()


def test_tiles_do_not_show(scene):
    """The same stack fused in one tile and in many is the same picture."""
    truth, _ = scene
    frames = _stack(truth, moved=False)
    one = fuse_stack(frames, [None] * 3, np.empty_like(truth), reference=1, budget_mb=4000)
    many = fuse_stack(frames, [None] * 3, np.empty_like(truth), reference=1, budget_mb=12)
    assert np.abs(one - many).max() <= 1e-6 * float(truth.max())


def test_the_full_merge_keeps_its_frames_on_disk_and_cleans_up(scene, xdg_home):
    from ape.config import get_settings

    truth, like = scene
    frames = _stack(truth)
    first = run(_recipe(), preview=False, progress=None, decode=_decoder(frames, like))
    again = run(_recipe(), preview=False, progress=None, decode=_decoder(frames, like))
    # Deterministic: a lost intermediate is rebuilt identical (section 25.1).
    assert np.array_equal(first.decoded.rgb, again.decoded.rgb)
    assert "coverage_image" not in first.report  # the full report is JSON
    tmp = get_settings().cache_dir / "tmp"
    assert not tmp.exists() or not any(tmp.iterdir())


def test_frames_that_are_not_a_stack_fail_with_a_reason(scene):
    truth, like = scene
    frames = _stack(truth)
    frames[2] = np.random.default_rng(1).uniform(0, 0.5, truth.shape).astype(np.float32)
    with pytest.raises(MergeFailure, match="DSC00002.ARW"):
        run(_recipe(), preview=True, progress=None, decode=_decoder(frames, like))
