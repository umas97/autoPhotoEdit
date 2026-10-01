# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 14 of section 13: the metadata of an exported file (section 16.3).

"The export carries the shooting EXIF, ``Software`` set and ``Orientation=1``;
Artist/Copyright reflect the preferences; GPS is present or absent according to
the toggle, and with the toggle on *no* GPS tag survives."

"No tag survives" is checked in every block exiv2 can read -- EXIF, XMP, IPTC
-- and then in the bytes of the file, so that a block exiv2 does not parse
could not hide one either.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.db.enums import ExportConflict
from ape.export.image import SOFTWARE_TAG, ExportFormat, encode_image
from ape.export.metadata import ExportMetadata, embed_metadata

pyexiv2 = pytest.importorskip("pyexiv2")

SOURCE = {
    "Exif.Image.Make": "SONY",
    "Exif.Image.Model": "ILCE-7M3",
    "Exif.Image.Orientation": "6",
    "Exif.Image.Software": "ILCE-7M3 v4.01",
    "Exif.Photo.ExposureTime": "1/125",
    "Exif.Photo.FNumber": "28/10",
    "Exif.Photo.ISOSpeedRatings": "250",
    "Exif.Photo.FocalLength": "750/10",
    "Exif.Photo.LensModel": "E 28-75mm F2.8 A063",
    "Exif.Photo.DateTimeOriginal": "2026:02:26 14:37:59",
    "Exif.Photo.BodySerialNumber": "1234567",
    "Exif.Photo.LensSerialNumber": "7654321",
    "Exif.Photo.MakerNote": "1 2 3 4",
    "Exif.GPSInfo.GPSLatitudeRef": "N",
    "Exif.GPSInfo.GPSLatitude": "43/1 4/1 0/1",
    "Exif.GPSInfo.GPSLongitudeRef": "E",
    "Exif.GPSInfo.GPSLongitude": "12/1 37/1 0/1",
}


def _read(payload: bytes) -> tuple[dict, dict, dict]:
    image = pyexiv2.ImageData(payload)
    try:
        return image.read_exif(), image.read_xmp(), image.read_iptc()
    finally:
        image.close()


def _export(fmt: ExportFormat, **changes) -> bytes:
    rng = np.random.default_rng(1)
    pixels = rng.uniform(0, 1, size=(40, 60, 3)).astype(np.float32)
    meta = ExportMetadata(
        source_exif=SOURCE,
        width=60,
        height=40,
        software=SOFTWARE_TAG,
        artist="Filippo Castellan",
        copyright="© 2026 Filippo Castellan",
        **changes,
    )
    return embed_metadata(encode_image(pixels, fmt), meta)


@pytest.mark.parametrize("fmt", list(ExportFormat))
def test_shooting_exif_software_and_orientation(fmt):
    exif, xmp, _iptc = _read(_export(fmt))
    for key in (
        "Exif.Image.Make",
        "Exif.Image.Model",
        "Exif.Photo.ExposureTime",
        "Exif.Photo.FNumber",
        "Exif.Photo.ISOSpeedRatings",
        "Exif.Photo.FocalLength",
        "Exif.Photo.LensModel",
        "Exif.Photo.DateTimeOriginal",
    ):
        assert exif[key] == SOURCE[key], key
    assert exif["Exif.Image.Software"] == SOFTWARE_TAG
    assert xmp["Xmp.xmp.CreatorTool"] == SOFTWARE_TAG
    # The pixels are upright: anything but 1 would rotate them a second time.
    assert exif["Exif.Image.Orientation"] == "1"
    assert exif["Exif.Photo.PixelXDimension"] == "60"


@pytest.mark.parametrize("fmt", list(ExportFormat))
def test_artist_and_copyright_come_from_the_preferences(fmt):
    exif, xmp, _iptc = _read(_export(fmt))
    assert exif["Exif.Image.Artist"] == "Filippo Castellan"
    assert exif["Exif.Image.Copyright"] == "© 2026 Filippo Castellan"
    assert xmp["Xmp.dc.creator"] == ["Filippo Castellan"]
    assert list(xmp["Xmp.dc.rights"].values()) == ["© 2026 Filippo Castellan"]


def test_serial_numbers_and_maker_notes_are_never_copied():
    payload = _export(ExportFormat.JPEG)
    exif, _xmp, _iptc = _read(payload)
    assert not [k for k in exif if "Serial" in k or "MakerNote" in k]
    assert b"1234567" not in payload and b"7654321" not in payload


@pytest.mark.parametrize("fmt", list(ExportFormat))
def test_gps_is_copied_by_default(fmt):
    exif, _xmp, _iptc = _read(_export(fmt))
    assert exif["Exif.GPSInfo.GPSLatitude"] == "43/1 4/1 0/1"
    assert exif["Exif.GPSInfo.GPSLongitudeRef"] == "E"


@pytest.mark.parametrize("fmt", list(ExportFormat))
def test_with_the_toggle_no_gps_tag_survives_anywhere(fmt):
    payload = _export(fmt, strip_gps=True)
    exif, xmp, iptc = _read(payload)
    assert not [k for k in exif if "GPS" in k]
    assert not [k for k in xmp if "GPS" in k or "Location" in k]
    assert not [k for k in iptc if "Location" in k or "City" in k]
    assert b"GPS" not in payload


def test_the_icc_profile_survives_the_rewrite():
    image = pyexiv2.ImageData(_export(ExportFormat.TIFF16))
    try:
        assert len(image.read_icc()) > 0
    finally:
        image.close()


def test_colour_space_tag_follows_the_output_space():
    from ape.pipeline.colorspace import OutputSpace

    srgb, _x, _i = _read(_export(ExportFormat.JPEG))
    p3, _x, _i = _read(_export(ExportFormat.JPEG, output_space=OutputSpace.DISPLAY_P3))
    assert srgb["Exif.Photo.ColorSpace"] == "1"
    assert p3["Exif.Photo.ColorSpace"] == "65535"


# --------------------------------------------------------------------------- #
# End to end: the preferences reach the file through a batch.


@pytest.mark.parametrize("strip", [False, True])
def test_a_batch_writes_the_preferences_and_honours_the_toggle(
    catalog, tmp_path, monkeypatch, strip
):
    from ape.db.session import write_setting
    from export_helpers import make_project, run_queue, start_batch, synthetic_decode

    synthetic_decode(monkeypatch)
    project = make_project(catalog, tmp_path / "card", count=1)
    with catalog() as session:
        write_setting(session, "artist", "Filippo Castellan")
        write_setting(session, "copyright", "© 2026")
        session.commit()
    out = tmp_path / "export"
    start_batch(
        catalog, project, output_dir=str(out), strip_gps=strip, on_conflict=ExportConflict.RENAME
    )
    run_queue()

    payload = (out / "DSC00001.jpg").read_bytes()
    exif, _xmp, _iptc = _read(payload)
    assert exif["Exif.Image.Artist"] == "Filippo Castellan"
    assert exif["Exif.Image.Copyright"] == "© 2026"
    assert exif["Exif.Image.Orientation"] == "1"
    assert exif["Exif.Photo.DateTimeOriginal"] == "2026:02:26 14:37:59"
    assert ("Exif.GPSInfo.GPSLatitude" in exif) is not strip
    if strip:
        assert b"GPS" not in payload


@pytest.mark.fixtures
def test_a_real_export_carries_the_real_exif(raw_fixtures, tmp_path):
    """One full-resolution export of a real ARW, reduced to 1024 px."""
    from ape.export.run import ExportWork, export_one
    from ape.export.settings import ExportSettings
    from ape.pipeline.params import neutral_params
    from ape.raw.metadata import read_metadata

    raw = raw_fixtures[0]
    settings = ExportSettings(output_dir=str(tmp_path), long_edge=1024)
    work = ExportWork(
        source=raw,
        filename=raw.name,
        name=f"{raw.stem}.jpg",
        params=neutral_params(),
        settings=settings,
        policy=ExportConflict.RENAME,
        protected=str(raw.parent),
    )
    result = export_one(work)
    exif, _xmp, _iptc = _read(result.image.path.read_bytes())
    original = read_metadata(raw).raw_tags
    for key in ("Exif.Image.Model", "Exif.Photo.ExposureTime", "Exif.Photo.DateTimeOriginal"):
        assert exif[key] == original[key]
    assert exif["Exif.Image.Orientation"] == "1"
    assert int(exif["Exif.Photo.PixelXDimension"]) in (1024, 683, 684)
