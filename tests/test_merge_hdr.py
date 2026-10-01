# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""HDR merge and the intermediate it writes (sections 25.1 and 25.3).

Test 17 of section 13: "la fusione HDR di 3 esposizioni sintetiche generate
dallo stesso RAW ricostruisce la scena originale con ΔE2000 medio < 2.0". The
scene here is a real A7 III frame scaled so that a third of it lies *above*
sensor saturation -- the reference frame clips it, and only the merge can give
it back -- and the three exposures are that scene at -2, 0 and +2 EV, clipped
at saturation, with photon and read noise, and shifted and rotated as a
hand-held bracketing is.
"""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np
import pytest

from ape.merge.align import align, for_matching, residual_shift
from ape.merge.errors import MergeFailure
from ape.merge.hdr import HdrOptions, merge_hdr
from ape.merge.warp import shrink
from ape.pipeline.colorspace import delta_e_2000, display_decode, to_lab
from ape.pipeline.params import neutral_params
from ape.pipeline.render import RenderOptions, render
from ape.raw import intermediate
from ape.raw.decode import decode_linear, decoded_from_array
from ape.raw.metadata import PhotoMetadata, read_metadata, read_optics

#: The scene is scaled so that this fraction of it lies above saturation in the
#: reference frame, out of its reach; the -2 EV frame still sees all of it.
_ABOVE_WHITE = 0.2

#: Photon noise (full well of ~40 000 e- at base ISO) and read noise, in units
#: of saturation.
_SHOT, _READ = 2.5e-5, 7.5e-5


@pytest.fixture(scope="module")
def scene(raw_fixtures) -> tuple[np.ndarray, object]:
    decoded = decode_linear(raw_fixtures[16], half_size=True)  # DSC05634, a frozen pond
    small, _ = shrink(decoded.rgb, 1500)
    scale = 1.0 / float(np.percentile(small.max(axis=-1), 100 * (1 - _ABOVE_WHITE)))
    return np.ascontiguousarray(small) * np.float32(scale), decoded


def _capture(scene: np.ndarray, ev: float, motion, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = scene * np.float32(2.0**ev)
    sigma = np.sqrt(np.maximum(frame, 0) * _SHOT + _READ**2).astype(np.float32)
    frame = np.clip(frame + rng.normal(size=frame.shape).astype(np.float32) * sigma, None, 1.0)
    if motion is not None:
        dx, dy, angle = motion
        height, width = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        matrix[:, 2] += (dx, dy)
        frame = cv2.warpAffine(
            frame, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT
        )
    return frame.astype(np.float32)


def _lab(rgb: np.ndarray, like) -> np.ndarray:
    """Rendered two stops down, the way the highlights of an HDR are looked at.

    At the neutral exposure the sigmoid puts everything past saturation near
    white whether it was recovered or not; two stops down, a clipped sky is a
    flat grey and a recovered one has its clouds.
    """
    decoded = decoded_from_array(rgb, baseline_exposure_ev=like.baseline_exposure_ev)
    decoded.camera = like.camera
    params = neutral_params().model_copy(deep=True)
    params.exposure.ev = -2.0
    image = render(decoded, params, RenderOptions(stop_before_output=True))
    return to_lab(display_decode(image[16:-16:2, 16:-16:2]))


def test_three_exposures_rebuild_the_scene(scene):
    truth, like = scene
    reference = _capture(truth, 0.0, None, 1)
    dark = _capture(truth, -2.0, (6.3, -4.1, 0.3), 2)
    bright = _capture(truth, 2.0, (-3.2, 5.7, -0.2), 3)
    assert (reference.max(axis=-1) >= 1.0).mean() > 0.15  # the reference really clips

    merged, report = merge_hdr(
        reference.copy(), [(lambda: dark, -2.0), (lambda: bright, 2.0)]
    )

    lab_truth = _lab(truth, like)
    error = float(delta_e_2000(lab_truth, _lab(merged, like)).mean())
    reference_only = float(delta_e_2000(lab_truth, _lab(reference, like)).mean())
    assert error < 2.0, (error, report.as_json())
    assert error < reference_only / 2  # the merge is what brought the highlights back
    assert merged.max() > 1.2  # linear, not tone-mapped: past saturation where the scene is
    summary = report.as_json()
    assert not summary["misaligned"] and summary["residual_px"] < 1.0
    assert [f["ev_used"] for f in summary["frames"]] == pytest.approx([0, -2, 2], abs=0.05)


def test_the_residual_sees_a_misalignment_and_not_a_moving_subject(scene):
    """What the report calls misaligned, on the scene's texture at the alignment size.

    A frame left 1.5 px off everywhere -- 6 px at full resolution -- is what
    section 25.3.2's "dichiaralo e lascia decidere" is for. A patch of an
    eighth of the frame moved by 6 px is a subject that moved, the
    deghosting's business: it must not make a bracketing that is sharp at 1:1
    read as misaligned, as the user's own did (``tests/fixtures/fase11``).
    """
    truth, _ = scene
    grey = for_matching(_capture(truth, 0.0, None, 4))
    other = for_matching(_capture(truth, 0.0, None, 5))  # the same scene, other noise
    height, width = other.shape
    assert residual_shift(grey, other) < 0.3

    shifted = cv2.warpAffine(other, np.float32([[1, 0, 1.5], [0, 1, 0]]), (width, height),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    assert residual_shift(grey, shifted) == pytest.approx(1.5, abs=0.3)

    moved = other.copy()
    rows, cols = slice(height // 3, 2 * height // 3), slice(width // 4, width * 5 // 8)
    moved[rows, cols] = other[rows, cols.start - 6 : cols.stop - 6]
    assert residual_shift(grey, moved) < 0.3


def test_alignment_lands_on_the_true_transform(scene):
    truth, _ = scene
    reference = _capture(truth, 0.0, None, 4)
    moved = _capture(truth, -2.0, (6.3, -4.1, 0.3), 5)
    small_ref, scale = shrink(reference, 750)
    small, _ = shrink(moved, 750)
    found = align(for_matching(small_ref), for_matching(small, 0.25))
    assert found is not None
    height, width = reference.shape[:2]
    true = cv2.getRotationMatrix2D((width / 2, height / 2), 0.3, 1.0)
    true[:, 2] += (6.3, -4.1)
    true = np.vstack([true, [0, 0, 1]])
    corners = np.float32([[0, 0], [width, 0], [0, height], [width, height]]).reshape(-1, 1, 2)
    error = cv2.perspectiveTransform(corners, found.to_full(scale)) - cv2.perspectiveTransform(
        corners, true
    )
    # Estimated at half the size: within a fifth of a pixel of the truth.
    assert float(np.abs(error).max()) < 0.2


def test_a_moving_subject_is_taken_from_the_reference(scene):
    truth, _ = scene
    base = truth / 4.0
    reference = _capture(base, 0.0, None, 6)
    reference[400:520, 300:420] = 0.15
    elsewhere = base.copy()
    elsewhere[400:520, 500:620] = 0.15  # the same subject, 200 px to the right
    dark, bright = _capture(elsewhere, -2.0, None, 7), _capture(elsewhere, 2.0, None, 8)

    merged, report = merge_hdr(reference.copy(), [(lambda: dark, -2.0), (lambda: bright, 2.0)])
    assert report.ghost_fraction > 0.005
    # Where the subject went, the merge shows what the reference saw there;
    # where it was in the reference, the subject -- once, not twice.
    moved = np.abs(merged[400:520, 500:620] - reference[400:520, 500:620]).mean()
    assert float(moved) < 0.003
    np.testing.assert_allclose(merged[405:515, 305:415].mean(), 0.15, atol=3e-3)

    ghosted, _ = merge_hdr(
        reference.copy(), [(lambda: dark, -2.0), (lambda: bright, 2.0)], HdrOptions(deghost=False)
    )
    assert float(np.abs(ghosted[400:520, 500:620] - reference[400:520, 500:620]).mean()) > 0.01


def test_frames_of_different_scenes_fail_with_a_reason(scene):
    truth, _ = scene
    reference = _capture(truth / 4.0, 0.0, None, 9)
    unrelated = np.random.default_rng(10).uniform(0, 1, reference.shape).astype(np.float32)
    with pytest.raises(MergeFailure, match="non si allinea"):
        merge_hdr(reference, [(lambda: unrelated, -2.0)])


# --------------------------------------------------------------------------- #
# The intermediate file.


def _context(rgb: np.ndarray) -> dict:
    decoded = decoded_from_array(rgb, temperature_k=5100.0, tint=3.0, baseline_exposure_ev=1.09)
    meta = PhotoMetadata(
        camera_make="SONY", camera_model="ILCE-7M3", lens_model="E 28-75mm F2.8 A063",
        iso=100, aperture=8.0, shutter=1 / 250, focal_length=35.0,
        shot_at=datetime(2026, 2, 26, 14, 37, 59), width=rgb.shape[1], height=rgb.shape[0],
        raw_tags={"Exif.Image.Model": "ILCE-7M3", "Exif.Photo.FNumber": "8/1"},
    )
    return intermediate.context_for(decoded, meta, kind="hdr")


def test_the_intermediate_reads_back_as_the_decode_of_its_reference(xdg_home, tmp_path):
    rng = np.random.default_rng(0)
    rgb = rng.uniform(-0.01, 3.0, size=(300, 500, 3)).astype(np.float32)
    path = tmp_path / f"merge{intermediate.SUFFIX}"
    intermediate.write_intermediate(path, rgb, _context(rgb), reduced_long_edge=200)

    decoded = decode_linear(path)
    np.testing.assert_array_equal(decoded.rgb, rgb.astype(np.float16).astype(np.float32))
    assert decoded.camera.as_shot_temperature_k == 5100.0 and decoded.camera.as_shot_tint == 3.0
    assert decoded.baseline_exposure_ev == 1.09
    assert decoded.lens is None  # a synthetic frame has no optics to correct
    reduced = decode_linear(path, half_size=True)
    assert reduced.rgb.shape == (120, 200, 3)
    np.testing.assert_allclose(
        reduced.rgb, cv2.resize(rgb, (200, 120), interpolation=cv2.INTER_AREA), atol=5e-3
    )

    meta = read_metadata(path)
    assert meta.camera_model == "ILCE-7M3" and meta.iso == 100 and meta.shutter == 1 / 250
    assert meta.shot_at == datetime(2026, 2, 26, 14, 37, 59)
    assert meta.raw_tags["Exif.Photo.FNumber"] == "8/1"
    assert read_optics(path).lens_model == "E 28-75mm F2.8 A063"


def test_writing_in_bands_is_writing_at_once(xdg_home, tmp_path):
    rgb = np.random.default_rng(1).uniform(0, 1, size=(333, 420, 3)).astype(np.float32)
    whole = intermediate.write_intermediate(
        tmp_path / f"a{intermediate.SUFFIX}", rgb, _context(rgb), reduced_long_edge=100
    )
    writer = intermediate.IntermediateWriter(
        tmp_path / f"b{intermediate.SUFFIX}", 420, 333, _context(rgb), reduced_long_edge=100
    )
    for top in range(0, 333, 50):
        writer.write_rows(top, rgb[top : top + 50])
    banded = writer.close()
    assert whole.read_bytes() == banded.read_bytes()
    assert not list(tmp_path.glob(".*part"))  # nothing left under a temporary name


def test_a_missing_intermediate_is_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        decode_linear(tmp_path / f"gone{intermediate.SUFFIX}")


def test_the_merge_is_dated_by_its_first_frame(scene):
    """Section 25.1: "DateTimeOriginal del primo scatto della sequenza" -- in
    the EXIF the export copies too, not only in the catalogue. A panorama's
    reference is its middle frame: the user's 07327, two seconds after 07319."""
    _, like = scene
    tags = {"Exif.Photo.DateTimeOriginal": "2026:09:29 12:54:44",
            "Exif.Photo.SubSecTimeOriginal": "512", "Exif.Image.Model": "ILCE-7M3"}
    first = datetime(2026, 9, 29, 12, 54, 42)
    context = intermediate.context_for(like, PhotoMetadata(shot_at=first, raw_tags=tags))
    written = context["metadata"]["raw_tags"]
    assert written["Exif.Photo.DateTimeOriginal"] == "2026:09:29 12:54:42"
    assert "Exif.Photo.SubSecTimeOriginal" not in written
    assert written["Exif.Image.Model"] == "ILCE-7M3"
    assert tags["Exif.Photo.DateTimeOriginal"] == "2026:09:29 12:54:44"  # the caller's, untouched


def test_a_panorama_says_its_lens_is_already_corrected(scene, tmp_path):
    """A panorama's frames were lens-corrected before stitching: whoever reads
    its optics must not look for a profile -- the user's panorama read
    "Obiettivo senza profilo lensfun" and offered to associate one."""
    truth, like = scene
    meta = PhotoMetadata(lens_model="E 28-75mm F2.8 A063", focal_length=28.0, aperture=2.8)
    for lens, expected in ((False, True), (True, False)):
        context = intermediate.context_for(like, meta, lens=lens)
        path = intermediate.write_intermediate(
            tmp_path / f"merge-{lens}.apemerge.tif", truth[:64, :96].copy(), context,
            reduced_long_edge=48,
        )
        optics = read_optics(path)
        assert optics.optics_corrected is expected
        assert optics.lens_model == "E 28-75mm F2.8 A063"

