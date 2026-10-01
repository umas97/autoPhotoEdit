# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 7 of section 13: the XMP sidecars.

"The darktable XMP must open in darktable without errors; the Adobe XMP must
be read by Lightroom/Camera Raw (manual test, documented)."

The darktable half is automated where darktable is installed (natively or as
the Flatpak ``org.darktable.Darktable``): ``darktable-cli`` develops a real ARW
with the generated sidecar, and its log must say that every module of the
history was loaded. Everywhere, the structure is checked -- well-formed XML, the
size of each module's parameters equal to darktable 5.6's struct -- and so is
the geometry, against the renderer itself. The Lightroom half is the manual
check, done by hand; what is automated here is the convention, which
was measured on the user's own Lightroom edits.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ape.export import xmp_adobe, xmp_darktable
from ape.export.crop_map import frame_corners, sensor_corners
from ape.export.sidecar import SidecarContext
from ape.pipeline.params import CropRect, EditParams, HSLBand

_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
    "darktable": "http://darktable.sf.net/",
}

#: ``sizeof(dt_iop_<module>_params_t)`` in darktable 5.6.1, by counting fields.
DT_SIZES = {
    "temperature": 20,
    "exposure": 28,
    "sigmoid": 56,
    "colorbalancergb": 132,
    "colorequal": 128,
    "shadhi": 48,
    "bilat": 20,
    "sharpen": 12,
    "tonecurve": 520,
    "crop": 24,
    "ashift": 892,
    "lens": 356,
}


def _ctx(**changes) -> SidecarContext:
    values = {
        "raw_filename": "DSC06312.ARW",
        "cam_from_xyz": np.array(
            [[0.7374, -0.2389, -0.0551], [-0.5435, 1.3162, 0.2519], [-0.1006, 0.1795, 0.6552]]
        ),
        "as_shot_multipliers": np.array([2.0664, 1.0, 2.1289]),
        "as_shot_temperature_k": 5200.0,
        "as_shot_tint": -8.0,
        "camera_maker": "SONY",
        "camera_model": "ILCE-7M3",
        "lens_model": "E 28-75mm F2.8 A063",
        "focal_length": 75.0,
        "aperture": 2.8,
        "width": 6000,
        "height": 4000,
        "orientation": 1,
        "software": "autoPhotoEdit test",
        "baseline_exposure_ev": 1.09,
    }
    values.update(changes)
    return SidecarContext(**values)


def _everything() -> EditParams:
    """Parameters that exercise every module the translators know."""
    params = EditParams()
    params.white_balance.mode = "custom"
    params.white_balance.temperature_k = 4300.0
    params.white_balance.tint = 12.0
    params.exposure.ev = 0.7
    params.tone_shaping.shadows = 0.2
    params.tone_shaping.whites = -0.1
    params.tone_curve.points = [(0.0, 0.0), (0.25, 0.2), (0.75, 0.8), (1.0, 1.0)]
    params.color.vibrance = 0.2
    params.color.saturation = -0.1
    params.color.hsl = {
        "orange": HSLBand(hue=0.1, saturation=-0.2),
        "blue": HSLBand(luminance=-0.3),
    }
    params.color.split_toning.shadow_hue = 220.0
    params.color.split_toning.shadow_saturation = 0.2
    params.local_contrast.shadows = 0.3
    params.local_contrast.highlights = -0.4
    params.local_contrast.clarity = 0.25
    params.sharpen.amount = 0.8
    params.geometry.rotation_deg = -1.5
    params.geometry.crop = CropRect(x=0.1, y=0.05, width=0.8, height=0.9)
    return params


def _description(payload: bytes) -> ET.Element:
    root = ET.fromstring(payload)
    return root.find(".//rdf:Description", _NS)


def _attr(element: ET.Element, prefix: str, name: str) -> str | None:
    return element.get(f"{{{_NS[prefix]}}}{name}")


# --------------------------------------------------------------------------- #
# Geometry: the corners the sidecars write are the corners the renderer crops.


def test_frame_corners_of_an_untouched_frame_are_the_frame():
    params = EditParams()
    assert frame_corners(params.geometry, 6000, 4000) == [(0, 0), (1, 0), (1, 1), (0, 1)]


def test_frame_corners_match_the_renderer():
    """Render a picture of its own coordinates; the output's corners must be there."""
    from ape.pipeline import geometry

    height, width = 400, 600
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    coordinates = np.dstack([(xs + 0.5) / width, (ys + 0.5) / height, np.zeros_like(xs)])
    params = _everything().geometry
    out = geometry.apply(coordinates, params)
    expected = frame_corners(params, width, height)
    h, w = out.shape[:2]
    for (u, v), (row, column) in zip(
        expected, ((0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0)), strict=True
    ):
        got = out[row, column, :2]
        # One output pixel of slack: the corner pixel's centre sits half a
        # pixel inside the corner itself.
        assert abs(got[0] - u) < 1.5 / width and abs(got[1] - v) < 1.5 / height


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        (1, ((0.1, 0.2), (0.6, 0.8))),
        (6, ((0.2, 0.4), (0.8, 0.9))),
        (8, ((0.2, 0.1), (0.8, 0.6))),
        (3, ((0.4, 0.2), (0.9, 0.8))),
    ],
)
def test_sensor_corners_follow_the_orientation(orientation, expected):
    params = EditParams()
    params.geometry.crop = CropRect(x=0.1, y=0.2, width=0.5, height=0.6)
    (left, top), (right, bottom) = sensor_corners(params.geometry, 6000, 4000, orientation)
    assert (round(left, 6), round(top, 6)) == expected[0]
    assert (round(right, 6), round(bottom, 6)) == expected[1]


# --------------------------------------------------------------------------- #
# Adobe.


def test_adobe_as_shot_stays_as_shot():
    description = _description(xmp_adobe.build(EditParams(), _ctx()))
    assert _attr(description, "crs", "WhiteBalance") == "As Shot"
    assert _attr(description, "crs", "Temperature") is None
    assert _attr(description, "crs", "RawFileName") == "DSC06312.ARW"
    assert _attr(description, "crs", "HasCrop") == "False"


@pytest.mark.parametrize(
    ("ours", "adobe"),
    # Our as-shot estimate and Lightroom's, for three of the user's photos.
    [((7027, 46.8), (7400, 44)), ((3290, 27.0), (3300, 24)), ((7651, 2.5), (8300, 0))],
)
def test_adobe_white_balance_follows_the_measured_fit(ours, adobe):
    kelvin, tint = xmp_adobe.adobe_temperature_tint(*ours)
    # Within the fit's worst case: 5.6 mired and 3.4 of tint.
    assert abs(1e6 / kelvin - 1e6 / adobe[0]) < 6.0
    assert abs(tint - adobe[1]) < 3.5


def test_adobe_carries_the_whole_edit():
    description = _description(xmp_adobe.build(_everything(), _ctx()))
    assert _attr(description, "crs", "WhiteBalance") == "Custom"
    assert _attr(description, "crs", "Exposure2012") == "+0.52"
    assert _attr(description, "crs", "Vibrance") == "+20"
    assert _attr(description, "crs", "HueAdjustmentOrange") == "+10"
    assert _attr(description, "crs", "LuminanceAdjustmentBlue") == "-30"
    assert _attr(description, "crs", "Shadows2012") == "+50"
    assert _attr(description, "crs", "CropAngle") == "-1.50"
    assert _attr(description, "crs", "HasCrop") == "True"
    points = description.findall("crs:ToneCurvePV2012/rdf:Seq/rdf:li", _NS)
    assert [p.text for p in points] == ["0, 0", "64, 51", "191, 204", "255, 255"]


# --------------------------------------------------------------------------- #
# darktable.


def _history(payload: bytes) -> list[dict[str, str]]:
    description = _description(payload)
    items = description.findall("darktable:history/rdf:Seq/rdf:li", _NS)
    prefix = f"{{{_NS['darktable']}}}"
    return [{k.removeprefix(prefix): v for k, v in item.attrib.items()} for item in items]


def test_darktable_history_matches_darktable_structs():
    payload = xmp_darktable.build(_everything(), _ctx())
    history = _history(payload)
    description = _description(payload)
    assert _attr(description, "darktable", "history_end") == str(len(history))
    assert _attr(description, "darktable", "auto_presets_applied") == "1"
    operations = [h["operation"] for h in history]
    assert set(operations) >= set(DT_SIZES) - {"lens"}
    for entry in history:
        params = bytes.fromhex(entry["params"])
        assert len(params) == DT_SIZES[entry["operation"]], entry["operation"]


def test_darktable_neutral_edit_is_small():
    operations = [h["operation"] for h in _history(xmp_darktable.build(EditParams(), _ctx()))]
    assert operations[0] == "temperature"
    assert "exposure" in operations and "sigmoid" in operations
    assert not {"colorequal", "shadhi", "bilat", "ashift", "crop"} & set(operations)


def test_darktable_white_balance_reproduces_the_camera_at_as_shot():
    """Our illuminant back to multipliers: the as-shot one must give the camera's."""
    from ape.raw.whitepoint import camera_multipliers_to_illuminant, xyz_to_cct_tint

    ctx = _ctx()
    illuminant = camera_multipliers_to_illuminant(ctx.as_shot_multipliers, ctx.cam_from_xyz)
    kelvin, tint = xyz_to_cct_tint(illuminant)
    back = xmp_darktable.multipliers_for(ctx, kelvin, tint)
    assert np.allclose(back, ctx.as_shot_multipliers, rtol=2e-3)


def test_darktable_sigmoid_fit_follows_our_curve():
    from ape.export import darktable_modules as dt
    from ape.pipeline.ops import tone

    params = EditParams()
    ev = np.linspace(-7, 5, 97)
    ours = np.power(tone.sigmoid_response(ev, params.tone), 2.4)
    contrast, skew, offset = dt.fit_sigmoid(ev, ours)
    theirs = dt._dt_sigmoid(0.18 * np.power(2.0, ev + offset), contrast, skew)
    error = np.abs(np.power(theirs, 1 / 2.4) - np.power(ours, 1 / 2.4))
    # darktable's curve never reaches black, ours does at the black point, so
    # the extremes cannot match exactly. Measured on four tone settings: mean
    # 0.008 of display code (two 8-bit steps), worst 0.029 at -7 EV.
    assert error.mean() < 0.01 and error.max() < 0.03, (error.mean(), error.max())


# --------------------------------------------------------------------------- #
# darktable itself.


def _darktable_cli() -> list[str] | None:
    native = shutil.which("darktable-cli")
    if native:
        return [native]
    flatpak = shutil.which("flatpak")
    if flatpak:
        listed = subprocess.run(
            [flatpak, "info", "org.darktable.Darktable"], capture_output=True, check=False
        )
        if listed.returncode == 0:
            return [flatpak, "run", "--command=darktable-cli", "org.darktable.Darktable"]
    return None


@pytest.mark.slow
@pytest.mark.fixtures
@pytest.mark.parametrize("case", ["neutral", "everything"])
def test_darktable_opens_the_sidecar_without_errors(raw_fixtures, tmp_path, case):
    """Test 7, automated: every module of the history is loaded by darktable."""
    from ape.export.sidecar import read_sidecar_context

    command = _darktable_cli()
    if command is None:
        pytest.skip("darktable non è installato")
    if command[0].endswith("flatpak"):
        command = [command[0], "run", f"--filesystem={tmp_path}", *command[2:]]

    raw = tmp_path / raw_fixtures[0].name
    shutil.copy2(raw_fixtures[0], raw)
    params = EditParams() if case == "neutral" else _everything()
    ctx = read_sidecar_context(raw, software="autoPhotoEdit test")
    sidecar = tmp_path / f"{raw.name}.xmp"
    sidecar.write_bytes(xmp_darktable.build(params, ctx))
    expected = {h["operation"] for h in _history(sidecar.read_bytes())}
    output = tmp_path / "out.jpg"
    run = subprocess.run(
        [
            *command,
            str(raw),
            str(sidecar),
            str(output),
            "--width",
            "512",
            "--height",
            "512",
            "--core",
            "--library",
            ":memory:",
            "--configdir",
            str(tmp_path / "dtconf"),
            "-d",
            "params",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    log = run.stdout + run.stderr
    assert run.returncode == 0, log[-3000:]
    assert output.is_file() and output.stat().st_size > 0
    loaded = {
        line.split("successfully loaded module ")[1].split()[0]
        for line in log.splitlines()
        if "successfully loaded module" in line
    }
    assert expected <= loaded, (expected - loaded, log[-3000:])
    assert "mismatch" not in log.lower(), log[-3000:]
    assert not math.isnan(output.stat().st_size)
