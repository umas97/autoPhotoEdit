# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Panoramas (section 25.5), on panning shots of a real frame (``pano_synth``).

What is checked: the backward maps are OpenCV's; a pan comes back as the scene
it was cut from, exposure differences included; the bands do not show; the
refusals of 25.5.4 and the failures of 25.5.6 say what is wrong; the full merge
lands in an intermediate, deterministically, and leaves nothing behind.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ape.analysis.borders import largest_rectangle
from ape.merge.align import align, for_matching, residual_shift
from ape.merge.engine import Member, Recipe
from ape.merge.errors import MergeFailure
from ape.merge.pano_compose import compose, layout
from ape.merge.pano_estimate import estimate
from ape.merge.pano_geometry import Camera, backward_map, warp_roi
from ape.merge.pano_photometry import NO_VIGNETTING, falloff, radius2
from ape.merge.panorama import _refuse_if_too_large, run, run_full
from ape.merge.warp import shrink, warp_whole
from ape.raw.decode import decode_linear
from ape.raw.intermediate import frame_map, full_size
from pano_synth import rotation, shoot

_WORLD_FOCAL, _FOCAL, _SIZE = 1000.0, 1374.0, (1000, 667)
_PAN = [(-24, 0.5, 0.3), (-8, -0.4, -0.2), (8, 0.3, 0.4), (24, -0.2, -0.3)]
_GAINS = [1.0, 1.0, 0.7, 1.0]  # one frame shot 0.5 EV darker


@pytest.fixture(scope="module")
def world(raw_fixtures) -> tuple[np.ndarray, object]:
    decoded = decode_linear(raw_fixtures[16], half_size=True)  # DSC05634, a frozen pond
    small, _ = shrink(decoded.rgb, 3000)
    return np.ascontiguousarray(small, dtype=np.float32), decoded


def _recipe(count: int, options: dict | None = None) -> Recipe:
    members = tuple(
        Member(photo_id=i, path=f"/card/DSC0000{i}.ARW", filename=f"DSC0000{i}.ARW",
               hash=f"h{i}", ev_offset=None, reference=i == 1)
        for i in range(count)
    )
    return Recipe(group_id=1, kind="panorama", members=members, options=options or {})


def _decoder(frames, like):
    def decode(member, *, preview, white_balance=None):
        return SimpleNamespace(rgb=frames[member.photo_id].copy(), camera=like.camera, lens=None,
                               baseline_exposure_ev=like.baseline_exposure_ev)
    return decode


@pytest.mark.parametrize("projection", ["cylindrical", "spherical", "plane", "transverseMercator"])
def test_the_backward_map_is_opencvs(projection):
    k = np.array([[900.0, 0, 500], [0, 900, 330], [0, 0, 1]])
    camera = Camera(K=k, R=rotation(12, 3, 1))
    warper = cv2.PyRotationWarper(projection, 850.0)
    roi, map_x, map_y = warper.buildMaps((1000, 660), k.astype(np.float32),
                                         camera.R.astype(np.float32))
    v, u = np.mgrid[roi[1] : roi[1] + map_x.shape[0], roi[0] : roi[0] + map_x.shape[1]]
    ours_x, ours_y = backward_map(projection, 850.0, camera, u.astype(float), v.astype(float))
    inside = (map_x >= 0) & (map_x < 1000) & (map_y >= 0) & (map_y < 660)
    assert np.abs(ours_x - map_x)[inside].max() < 0.01
    assert np.abs(ours_y - map_y)[inside].max() < 0.01
    assert warp_roi(projection, 850.0, camera, (1000, 660))[2:] == (map_x.shape[1], map_x.shape[0])


def test_a_pan_is_planned_as_it_was_shot(world):
    scene, _ = world
    frames = shoot(scene, _WORLD_FOCAL, _SIZE, _FOCAL, _PAN, gains=_GAINS)
    plan = estimate(frames, [f"F{i}" for i in range(4)], 1)
    assert plan.projection == "cylindrical"  # 88 degrees across, a horizontal sweep
    assert not plan.vertical
    assert plan.span_deg[0] == pytest.approx(88.0, abs=2.0)
    assert plan.scale == pytest.approx(_FOCAL, rel=0.02)
    assert plan.gains == pytest.approx([1.0, 1.0, 1 / 0.7, 1.0], rel=0.02)
    assert plan.crop is not None
    # A lens without falloff is not given one.
    edge = float(falloff(np.float32(plan.vignetting[3]), plan.vignetting))
    assert edge == pytest.approx(1, abs=0.03)


def test_the_lens_falloff_is_found_and_undone(world):
    """Without a lens profile every seam of a smooth sky is a step: the user's
    07319-07335 at 28 mm f/2.8 had steps of 5-6 per cent. Here each shot loses
    1.3 stops towards its corners; the overlaps must find it, and the gains
    must come out as they were shot, not bent by it."""
    scene, _ = world
    frames = shoot(scene, _WORLD_FOCAL, _SIZE, _FOCAL, _PAN, gains=_GAINS)
    height, width = frames[0].shape[:2]
    truth = (1.0 - 0.6 * radius2(width, height)).astype(np.float32)  # 0.4 at the corners
    plan = estimate([f * truth[..., None] for f in frames], [f"F{i}" for i in range(4)], 1)
    assert plan.gains == pytest.approx([1.0, 1.0, 1 / 0.7, 1.0], rel=0.03)
    for r2 in (0.1, 0.3, 0.5):
        found = float(falloff(np.float32(r2), plan.vignetting))
        assert found == pytest.approx(1.0 - 0.6 * r2, abs=0.04), (r2, plan.vignetting)

    # And undone as the frames are read: through the same plan, the shaded
    # frames compose to what the clean ones compose to without it.
    out = layout(plan, _SIZE, 0.5)

    def composed(sources, used):
        canvas = np.zeros((out.size[1], out.size[0], 3), np.float32)

        def sink(top, band):
            canvas[top : top + band.shape[0]] = band

        compose(sources, used, out, sink, levels=4)
        return canvas

    shaded = composed([f * truth[..., None] for f in frames], plan)
    clean = composed(frames, dataclasses.replace(plan, vignetting=NO_VIGNETTING))
    covered = cv2.erode((clean.max(axis=-1) > 0).astype(np.uint8), np.ones((9, 9))) > 0
    assert np.abs(shaded - clean)[covered].mean() / clean[covered].mean() < 0.03


#: A sweep that tilts instead of turning, like the user's 07336-07354.
_TILT = [(0.5, -24, 0.3), (-0.4, -8, -0.2), (0.3, 8, 0.4), (-0.2, 24, -0.3)]


def test_a_tilt_is_planned_upright(world):
    """OpenCV's horizontal wave correction on a tilt looks for the normal of a
    plane of camera X axes that are all the same, and picks a vertical at
    random: the user's panorama came out spherical, 360 by 180 degrees, mostly
    black. A tilt gets the cylinder lying along it, and stands up."""
    scene, like = world
    frames = shoot(scene, _WORLD_FOCAL, _SIZE, _FOCAL, _TILT)
    plan = estimate(frames, [f"F{i}" for i in range(4)], 1)
    assert plan.vertical and plan.projection == "cylindrical"
    assert plan.warper == "transverseMercator"
    assert plan.span_deg[1] == pytest.approx(48 + 27, abs=3)  # the tilt plus one frame
    assert plan.span_deg[0] == pytest.approx(40, abs=3)  # one frame across
    out = layout(plan, _SIZE, 1.0)
    assert out.size[1] > 1.5 * out.size[0]

    # Upright and not mirrored: the panorama maps onto the scene with no turn.
    outcome = run(_recipe(4), preview=True, progress=None, decode=_decoder(frames, like))
    pano = for_matching(outcome.decoded.rgb)
    small = for_matching(cv2.resize(scene, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(pano, None)
    kb, db = orb.detectAndCompute(small, None)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    src = np.float32([ka[m.queryIdx].pt for m in matches])
    dst = np.float32([kb[m.trainIdx].pt for m in matches])
    similarity, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    assert int(inliers.sum()) > 40
    angle = np.degrees(np.arctan2(similarity[1, 0], similarity[0, 0]))
    assert abs(angle) < 3.0, angle


def test_the_panorama_is_the_scene_it_was_cut_from(world):
    """Through the plane projection, rotation-only shots of a plane rebuild it."""
    scene, like = world
    turns = [(-12, 0.3, 0.2), (0, 0, 0), (12, -0.3, -0.2)]
    frames = shoot(scene, _WORLD_FOCAL, _SIZE, _FOCAL, turns, gains=[1.0, 1.0, 0.7])
    outcome = run(_recipe(3, {"projection": "plane"}), preview=True, progress=None,
                  decode=_decoder(frames, like))
    pano = outcome.decoded.rgb
    assert outcome.report["projection"] == "plane" and outcome.decoded.lens is None
    assert outcome.keep_lens is False

    # Register the panorama to the scene and compare where both have pixels.
    a, b = for_matching(pano), for_matching(scene)
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    src = np.float32([ka[m.queryIdx].pt for m in matches])
    dst = np.float32([kb[m.trainIdx].pt for m in matches])
    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    height, width = pano.shape[:2]
    rough = cv2.warpPerspective(scene, homography, (width, height),
                                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)
    # The feature-point registration drifts by pixels across the frame; ECC
    # takes it the rest of the way, so what is left is the merge's own error.
    refined = align(for_matching(pano), for_matching(rough))
    truth = np.nan_to_num(warp_whole(rough, refined.matrix, (width, height)))
    covered = cv2.erode((pano.max(axis=-1) > 0).astype(np.uint8), np.ones((15, 15))) > 0
    error = np.abs(pano - truth)[covered].mean() / truth[covered].mean()
    assert error < 0.04, error  # exposure compensated: no step at the darker frame
    # No double edges at the seams. Measured inside a 128 px border: towards
    # the corners the registration of the test itself drifts smoothly (1-3 px
    # on the textureless ice), which is not the merge's error.
    inner = (slice(128, -128), slice(128, -128))
    assert residual_shift(for_matching((truth * covered[..., None])[inner]),
                          for_matching((pano * covered[..., None])[inner])) < 1.2


def test_bands_do_not_show(world):
    scene, _ = world
    frames = shoot(scene, _WORLD_FOCAL, _SIZE, _FOCAL, _PAN)
    plan = estimate(frames, [f"F{i}" for i in range(4)], 1)
    out = layout(plan, _SIZE, 0.6)
    results = []
    for rows in (64, 1024):
        canvas = np.zeros((out.size[1], out.size[0], 3), np.float32)

        def sink(top, band, canvas=canvas):
            canvas[top : top + band.shape[0]] = band

        compose(frames, plan, out, sink, levels=4, rows=rows)
        results.append(canvas)
    assert np.array_equal(results[0], results[1])


def test_frames_that_do_not_overlap_fail_with_a_reason(world):
    scene, like = world
    frames = shoot(scene, _WORLD_FOCAL, _SIZE, _FOCAL, [(-30, 0, 0), (0, 0, 0), (32, 0, 0)])
    frames[2] = np.random.default_rng(3).uniform(0, 0.4, frames[2].shape).astype(np.float32)
    with pytest.raises(MergeFailure, match="DSC00002.ARW non si sovrappone"):
        run(_recipe(3), preview=True, progress=None, decode=_decoder(frames, like))


def test_too_much_is_refused_before_it_starts():
    with pytest.raises(MergeFailure, match="tetto di 12"):
        run(_recipe(13), preview=True, progress=None, decode=None)
    huge = SimpleNamespace(size=(26000, 9000))
    with pytest.raises(MergeFailure, match=r"234 MP .*GB di memoria.*riduci la risoluzione al 90%"):
        _refuse_if_too_large(huge, (6000, 4000), 1.0)
    _refuse_if_too_large(SimpleNamespace(size=(20000, 9000)), (6000, 4000), 1.0)  # 180 MP: fine


def test_the_full_panorama_goes_to_an_intermediate_and_cleans_up(world, xdg_home, tmp_path):
    from ape.config import get_settings

    scene, like = world
    frames = shoot(scene, _WORLD_FOCAL, _SIZE, _FOCAL, _PAN, gains=_GAINS)
    recipe = _recipe(4, {"output_scale": 0.5})

    def context(decoded, width, height):
        return {"merge": {"kind": "panorama"}, "size": [width, height]}

    paths = []
    for name in ("a.tif", "b.tif"):
        path, report, size = run_full(recipe, tmp_path / name, context, reduced_long_edge=512,
                                      progress=None, decode=_decoder(frames, like))
        paths.append(path)
    assert full_size(paths[0]) == size and report["output"]["scale"] == 0.5
    assert report["ram_estimate_mb"] > 0 and report["crop"]["width"] > 0.8
    assert np.array_equal(np.asarray(frame_map(paths[0])), np.asarray(frame_map(paths[1])))
    tmp = get_settings().cache_dir / "tmp"
    assert not tmp.exists() or not any(tmp.iterdir())


def test_the_largest_rectangle_is_found():
    mask = np.zeros((20, 30), bool)
    mask[2:15, 3:25] = True
    mask[5, 10] = False  # a hole cuts it
    x, y, w, h = largest_rectangle(mask)
    assert mask[y : y + h, x : x + w].all()
    assert w * h == max(22 * 9, 22 * 3, 15 * 13, 7 * 13)
