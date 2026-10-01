# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lens correction: which profile, and what it does to the pixels (section 6.4).

Phase 5's acceptance asks that "the common Sony lenses are recognised by
lensfun". The list below is the EXIF ``LensModel`` of the FE lenses most people
own, spelled as the camera writes it. Every test here runs against the database
bundled with the wheel, inside a temporary XDG home, so the outcome does not
depend on whether this machine's user has updated their lensfun data.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

lensfunpy = pytest.importorskip("lensfunpy")

from ape import lensdb  # noqa: E402
from ape.lensdb import LensIdentity  # noqa: E402
from ape.pipeline import lens  # noqa: E402

COMMON_SONY = (
    "FE 28-70mm F3.5-5.6 OSS",
    "FE 24-70mm F2.8 GM",
    "FE 24-70mm F4 ZA OSS",
    "FE 24-105mm F4 G OSS",
    "FE 16-35mm F4 ZA OSS",
    "FE 16-35mm F2.8 GM",
    "FE 70-200mm F2.8 GM OSS",
    "FE 70-300mm F4.5-5.6 G OSS",
    "FE 24-240mm F3.5-6.3 OSS",
    "FE 100-400mm F4.5-5.6 GM OSS",
    "FE 200-600mm F5.6-6.3 G OSS",
    "FE 12-24mm F4 G",
    "FE 50mm F1.8",
    "FE 85mm F1.8",
    "FE 85mm F1.4 GM",
    "FE 55mm F1.8 ZA",
    "FE 35mm F1.8",
    "FE 35mm F2.8 ZA",
    "FE 28mm F2",
    "FE 20mm F1.8 G",
    "FE 24mm F1.4 GM",
    "FE 90mm F2.8 Macro G OSS",
    "FE 50mm F2.8 Macro",
)

#: The user's own lens, trimmed from lensfun's ``mil-tamron.xml``, and the body
#: it was calibrated on: enough for a database update that needs no network.
_A063_XML = """<lensdatabase version="1">
    <camera>
        <maker>Sony</maker>
        <model>ILCE-7M3</model>
        <mount>Sony E</mount>
        <cropfactor>1</cropfactor>
    </camera>
    <lens>
        <maker>Tamron</maker>
        <model>Tamron E 28-75mm F/2.8 Di III VXD G2 A063Z</model>
        <mount>Sony E</mount>
        <cropfactor>1.0</cropfactor>
        <calibration>
            <distortion model="ptlens" focal="28.0" a="0.0217313" b="-0.0742292" c="0.0601221"/>
            <distortion model="ptlens" focal="75.0" a="0.00860079" b="-0.0129982" c="0.0328982"/>
            <vignetting model="pa" focal="28.0" aperture="2.8" distance="10" k1="0.0306281" k2="-0.7278004" k3="0.3541124"/>
            <vignetting model="pa" focal="28.0" aperture="2.8" distance="1000" k1="0.0306281" k2="-0.7278004" k3="0.3541124"/>
            <vignetting model="pa" focal="75.0" aperture="2.8" distance="10" k1="0.0142939" k2="-0.6701271" k3="0.3471165"/>
            <vignetting model="pa" focal="75.0" aperture="2.8" distance="1000" k1="0.0142939" k2="-0.6701271" k3="0.3471165"/>
        </calibration>
    </lens>
</lensdatabase>
"""


def _identity(model: str, focal: float, aperture: float = 2.8, **kwargs) -> LensIdentity:
    return LensIdentity("SONY", "ILCE-7M3", model, focal, aperture, **kwargs)


def _stated_focal(model: str) -> float:
    low = lensdb._focal_range(model)
    assert low is not None
    return low[0]


def _archive(tmp_path: Path, xml: str = _A063_XML) -> Path:
    path = tmp_path / "version_1.tar.bz2"
    with tarfile.open(path, "w:bz2") as tar:
        data = xml.encode()
        info = tarfile.TarInfo("mil-test.xml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize("model", COMMON_SONY)
def test_common_sony_lenses_are_recognised(xdg_home, model):
    profile = lensdb.resolve(_identity(model, _stated_focal(model)))
    assert profile is not None, f"{model} senza profilo lensfun"
    assert profile.maker == "Sony"
    assert profile.source == "auto"


def test_a_lens_the_database_lacks_is_not_matched_to_a_lookalike(xdg_home):
    """The A063 is not in the wheel's data; the older A036, also a 28-75, is.

    Correcting one with the other's profile would be a silent wrong answer.
    """
    assert lensdb.data_status()["source"] == "bundled"
    assert lensdb.resolve(_identity("E 28-75mm F2.8 A063", 28.0)) is None


def test_a_match_whose_focal_range_disagrees_is_refused(xdg_home):
    """The second line behind lensfun's matcher: the numbers must agree.

    lensfun itself is strict about focal lengths today; the guard is there for
    the day a looser match slips through, and for EXIF that contradicts itself.
    """
    from types import SimpleNamespace

    zoom = SimpleNamespace(min_focal=24.0, max_focal=70.0)
    assert lensdb._plausible(_identity("FE 24-70mm F2.8 GM", 50.0), zoom)
    assert not lensdb._plausible(_identity("FE 28-70mm F3.5-5.6 OSS", 50.0), zoom)
    # A focal length outside the lens's range means the name lied.
    assert not lensdb._plausible(_identity("FE 24-70mm F2.8 GM", 200.0), zoom)
    assert lensdb.resolve(_identity("FE 24-70mm F2.8 GM", 200.0)) is None


def test_update_adds_the_missing_lens_and_invalidates_the_caches(xdg_home, tmp_path):
    identity = _identity("E 28-75mm F2.8 A063", 28.0)
    assert lensdb.resolve(identity) is None

    status = lensdb.install_update(_archive(tmp_path))
    assert status["source"] == "updated"

    profile = lensdb.resolve(identity)
    assert profile is not None
    assert profile.maker == "Tamron" and "A063" in profile.model


def test_a_broken_update_keeps_the_working_database(xdg_home, tmp_path):
    with pytest.raises(ValueError):
        lensdb.install_update(_archive(tmp_path, xml="<lensdatabase version='1'><lens>"))
    assert lensdb.data_status()["source"] == "bundled"
    assert lensdb.resolve(_identity("FE 85mm F1.8", 85.0)) is not None


def test_manual_association_wins_over_matching(xdg_home):
    identity = _identity("Obiettivo sconosciuto 85mm", 85.0)
    assert lensdb.resolve(identity) is None
    candidates = lensdb.search_lenses("85mm 1.8 sony")
    chosen = next(c for c in candidates if c["model"] == "FE 85mm f/1.8")
    manual = _identity(
        "Obiettivo sconosciuto 85mm", 85.0, override=(chosen["maker"], chosen["model"])
    )
    profile = lensdb.resolve(manual)
    assert profile is not None and profile.source == "override"


def test_correction_brightens_the_corners_and_leaves_the_centre(xdg_home, tmp_path):
    lensdb.install_update(_archive(tmp_path))
    flat = np.full((400, 600, 3), 0.18, dtype=np.float32)
    before = flat.copy()
    corrected = lens.apply(flat, _identity("E 28-75mm F2.8 A063", 28.0))

    np.testing.assert_array_equal(flat, before)  # a pure function
    centre = corrected[200, 300, 1]
    corner = corrected[2, 2, 1]
    assert abs(centre - 0.18) < 1e-3
    # f/2.8 at 28 mm: the profile lifts the corners by about half a stop.
    assert 1.3 < corner / centre < 1.7


def test_correction_straightens_what_the_lens_bent(xdg_home, tmp_path):
    """Distort a straight line with lensfun's own reverse model, then correct it."""
    lensdb.install_update(_archive(tmp_path))
    identity = _identity("E 28-75mm F2.8 A063", 28.0)
    height, width = 600, 900
    line = np.zeros((height, width, 3), dtype=np.float32)
    line[48:52] = 1.0  # a horizontal line near the top edge, where distortion is worst

    camera, found = lensdb.find_lens(identity)
    modifier = lensfunpy.Modifier(found, 1.0, width, height)
    modifier.initialize(28.0, 2.8, 1000.0, pixel_format=np.float32, reverse=True,
                        flags=lensfunpy.ModifyFlags.DISTORTION)
    import cv2

    coords = modifier.apply_geometry_distortion()
    bent = cv2.remap(line, coords, None, cv2.INTER_LINEAR)

    def row_centres(img: np.ndarray) -> np.ndarray:
        columns = img[..., 1]
        weights = columns.sum(axis=0)
        rows = np.arange(img.shape[0], dtype=np.float64)[:, None]
        valid = weights > 0.5
        return ((columns * rows).sum(axis=0)[valid] / weights[valid])

    bent_spread = np.ptp(row_centres(bent[:200]))
    straight_spread = np.ptp(row_centres(lens.apply(bent, identity)[:200]))
    assert bent_spread > 3.0
    assert straight_spread < bent_spread / 4


def test_correction_is_resolution_independent(xdg_home, tmp_path):
    """The proxy and the export must agree (section 6.2)."""
    import cv2

    lensdb.install_update(_archive(tmp_path))
    identity = _identity("E 28-75mm F2.8 A063", 28.0)
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.random((1200, 1800, 3)).astype(np.float32), (0, 0), 12)
    large = lens.apply(base, identity)
    small = lens.apply(cv2.resize(base, (600, 400), interpolation=cv2.INTER_AREA), identity)
    reduced = cv2.resize(large, (600, 400), interpolation=cv2.INTER_AREA)
    assert float(np.abs(reduced - small)[8:-8, 8:-8].mean()) < 2e-3


def test_disabled_or_unknown_lens_is_the_identity(xdg_home):
    img = np.full((40, 60, 3), 0.3, dtype=np.float32)
    assert lens.apply(img, _identity("FE 85mm F1.8", 85.0), enabled=False) is img
    assert lens.apply(img, _identity("Chissà 50mm", 50.0)) is img
    assert lens.apply(img, None) is img
