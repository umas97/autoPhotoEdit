# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The metadata written into an exported file (docs/SPEC.md section 16.3).

What goes in is decided by an allow-list, never by copying the original and
deleting the unwanted. The RAW carries some two thousand five hundred tags --
Sony's maker notes, the embedded previews, the body's serial number in three
places -- and a deny-list is one unknown tag away from leaking the one the user
asked us to remove. With an allow-list, "no GPS tag survives in any block" is
true by construction: the only GPS written is the one copied on purpose, from
the EXIF GPS directory, and only when the toggle is off. The test (section
13.14) checks the output anyway.

* **Copied**: body, lens, exposure triangle, focal length, the time of the shot
  and its time zone, and the handful of shooting-mode tags a photographer
  filters on. **Not** the serial numbers of body and lens, nor the maker notes
  that contain them.
* **Written**: ``Orientation = 1`` (the pixels are already upright; anything
  else rotates them twice in half the viewers), ``Software`` and
  ``xmp:CreatorTool``, the pixel dimensions actually exported, the colour
  space, and the user's ``Artist`` / ``Copyright`` from the global settings.

This is one of the two places allowed to import pyexiv2 (section 24). The rest
of the program hands it bytes and gets bytes back.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..pipeline.colorspace import OutputSpace

__all__ = ["COPIED_EXIF", "ExportMetadata", "build_tags", "embed_metadata"]

_log = logging.getLogger(__name__)

#: The EXIF tags copied from the original, verbatim.
COPIED_EXIF: tuple[str, ...] = (
    "Exif.Image.Make",
    "Exif.Image.Model",
    "Exif.Photo.LensMake",
    "Exif.Photo.LensModel",
    "Exif.Photo.LensSpecification",
    "Exif.Photo.ISOSpeedRatings",
    "Exif.Photo.SensitivityType",
    "Exif.Photo.RecommendedExposureIndex",
    "Exif.Photo.ExposureTime",
    "Exif.Photo.FNumber",
    "Exif.Photo.MaxApertureValue",
    "Exif.Photo.FocalLength",
    "Exif.Photo.FocalLengthIn35mmFilm",
    "Exif.Photo.ExposureBiasValue",
    "Exif.Photo.ExposureProgram",
    "Exif.Photo.ExposureMode",
    "Exif.Photo.MeteringMode",
    "Exif.Photo.Flash",
    "Exif.Photo.LightSource",
    "Exif.Photo.SceneCaptureType",
    "Exif.Photo.DateTimeOriginal",
    "Exif.Photo.DateTimeDigitized",
    "Exif.Photo.SubSecTimeOriginal",
    "Exif.Photo.SubSecTimeDigitized",
    "Exif.Photo.OffsetTimeOriginal",
    "Exif.Photo.OffsetTimeDigitized",
)

#: Never copied, even if a future edit of the list above names them: the spec
#: excludes the serial numbers by name, and a maker note is where Sony keeps
#: another copy of them.
_NEVER = (
    "SerialNumber",
    "Exif.Photo.MakerNote",
    "Exif.Image.DNGPrivateData",
    "Exif.Image.XMLPacket",
)

#: EXIF ``ColorSpace``: 1 is sRGB; everything else is "uncalibrated", and the
#: embedded ICC profile says what it really is.
_EXIF_SRGB, _EXIF_UNCALIBRATED = "1", "65535"


@dataclass(slots=True)
class ExportMetadata:
    """Everything the metadata of one exported file depends on."""

    #: The original's EXIF, as ``raw/metadata.read_metadata`` returns it.
    source_exif: Mapping[str, Any] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    output_space: OutputSpace = OutputSpace.SRGB
    software: str = ""
    artist: str | None = None
    copyright: str | None = None
    strip_gps: bool = False
    #: When the file was produced. A parameter, so that tests are reproducible.
    written_at: datetime | None = None


def _forbidden(key: str) -> bool:
    return any(part in key for part in _NEVER)


def build_tags(meta: ExportMetadata) -> tuple[dict[str, str], dict[str, Any]]:
    """The EXIF and XMP to write, as exiv2 keys. Pure: no file is opened.

    Returns:
        ``(exif, xmp)``.
    """
    exif: dict[str, str] = {}
    for key in COPIED_EXIF:
        value = meta.source_exif.get(key)
        if value not in (None, "") and not _forbidden(key):
            exif[key] = str(value)
    if not meta.strip_gps:
        for key, value in meta.source_exif.items():
            if key.startswith("Exif.GPSInfo.") and value not in (None, ""):
                exif[key] = str(value)

    exif["Exif.Image.Orientation"] = "1"
    if meta.software:
        exif["Exif.Image.Software"] = meta.software
    if meta.written_at is not None:
        exif["Exif.Image.DateTime"] = meta.written_at.strftime("%Y:%m:%d %H:%M:%S")
    if meta.width and meta.height:
        exif["Exif.Photo.PixelXDimension"] = str(meta.width)
        exif["Exif.Photo.PixelYDimension"] = str(meta.height)
    exif["Exif.Photo.ColorSpace"] = (
        _EXIF_SRGB if meta.output_space is OutputSpace.SRGB else _EXIF_UNCALIBRATED
    )
    if meta.artist:
        exif["Exif.Image.Artist"] = meta.artist
    if meta.copyright:
        exif["Exif.Image.Copyright"] = meta.copyright

    xmp: dict[str, Any] = {}
    if meta.software:
        xmp["Xmp.xmp.CreatorTool"] = meta.software
    if meta.artist:
        xmp["Xmp.dc.creator"] = [meta.artist]
    if meta.copyright:
        xmp["Xmp.dc.rights"] = {'lang="x-default"': meta.copyright}
    return exif, xmp


def embed_metadata(payload: bytes, meta: ExportMetadata) -> bytes:
    """Return ``payload`` (a complete JPEG or TIFF) with its metadata written.

    Everything happens in memory: the caller writes the result once, under a
    temporary name, and renames it (``naming.write_exclusive``). If exiv2 is not
    available the pixels are exported without metadata and the reason is
    logged -- a missing optional library degrades an export, it does not stop
    one (section 3).

    The file is rebuilt from a clean slate: whatever metadata ``payload``
    already carried is cleared first, the ICC profile excepted.
    """
    try:
        import pyexiv2
    except Exception as exc:  # pragma: no cover - depends on the system libraries
        _log.warning("exiv2 non disponibile, export senza metadati: %s", exc)
        return payload

    exif, xmp = build_tags(meta)
    image = pyexiv2.ImageData(payload)
    try:
        image.clear_exif()
        image.clear_xmp()
        image.clear_iptc()
        image.modify_exif(exif)
        if xmp:
            image.modify_xmp(xmp)
        return image.get_bytes()
    finally:
        image.close()
