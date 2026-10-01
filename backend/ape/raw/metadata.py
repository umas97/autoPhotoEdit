# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading EXIF from RAW files, and the pairing clues of edited ones.

This module and ``export/`` are the **only** places in the codebase allowed to
import ``pyexiv2`` (docs/SPEC.md section 24, enforced by
``tests/test_licence_isolation.py``). pyexiv2 is GPL-3.0 and it is what forces
the project's own licence; keeping every call behind this interface is what
keeps replacing it a few hours of work rather than a refactor.

exiv2 is also optional at runtime: if the library is missing, every field comes
back ``None`` and the program keeps working with what LibRaw alone can tell it.
A missing lens model degrades the lens correction and lowers the confidence
score; it does not stop an import.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from .intermediate import is_intermediate

__all__ = [
    "PhotoMetadata",
    "ReferenceIdentity",
    "exiv2_available",
    "exiv2_version",
    "read_camera_model",
    "read_optics",
    "read_metadata",
    "read_reference_identity",
]

_log = logging.getLogger(__name__)


@dataclass
class PhotoMetadata:
    """What the pipeline and the catalogue need to know about a shot."""

    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    iso: int | None = None
    aperture: float | None = None
    shutter: float | None = None  # seconds
    focal_length: float | None = None  # mm
    exposure_bias: float | None = None  # EV, the bracketing marker of section 25.2
    shot_at: datetime | None = None
    orientation: int | None = None
    width: int | None = None
    height: int | None = None
    has_gps: bool = False
    #: The lens is already corrected in the pixels -- a panorama, stitched from
    #: frames corrected one by one -- and no profile has anything left to do.
    optics_corrected: bool = False
    #: Everything exiv2 returned, for the diagnostics bundle and for phases that
    #: need tags this dataclass does not name (Sony bracketing counters, focus
    #: distance). Never written back anywhere.
    raw_tags: dict[str, Any] = field(default_factory=dict)

    @property
    def camera(self) -> str | None:
        parts = [p for p in (self.camera_make, self.camera_model) if p]
        if not parts:
            return None
        # Sony writes "SONY" in Make and "ILCE-7M3" in Model; joining them blindly
        # gives "SONY ILCE-7M3", which is what users expect to see.
        if len(parts) == 2 and parts[1].upper().startswith(parts[0].upper()):
            return parts[1]
        return " ".join(parts)


def exiv2_available() -> bool:
    """Whether metadata reading is possible on this machine."""
    try:
        import pyexiv2  # noqa: F401
    except Exception:  # pragma: no cover - depends on the system libraries
        return False
    return True


def exiv2_version() -> str | None:
    """The exiv2 library under pyexiv2, for the diagnostics bundle (section 19)."""
    try:
        import pyexiv2
    except Exception:  # pragma: no cover - depends on the system libraries
        return None
    return str(getattr(pyexiv2, "__exiv2_version__", None) or "") or None


def _as_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(Fraction(value.strip()))
    except (ValueError, ZeroDivisionError):
        return None


def _as_int(value: str | None) -> int | None:
    number = _as_float(value)
    return None if number is None else int(round(number))


def _as_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value.strip()[:19], pattern)
        except ValueError:
            continue
    return None


def _first(tags: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = tags.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def _intermediate_metadata(path: Path) -> PhotoMetadata:
    """The EXIF a merged photo inherits from its reference frame (section 25.1).

    Written into the intermediate when the merge ran, with the shooting time of
    the first frame of the sequence. An intermediate that cannot be read gives
    an empty result, like a RAW exiv2 cannot parse.
    """
    from .intermediate import read_context

    try:
        stored = dict(read_context(path).get("metadata") or {})
    except (OSError, ValueError) as exc:
        _log.warning("metadati della fusione illeggibili in %s: %s", path.name, exc)
        return PhotoMetadata()
    if stored.get("shot_at"):
        stored["shot_at"] = datetime.fromisoformat(stored["shot_at"])
    known = {name for name in PhotoMetadata.__dataclass_fields__}
    return PhotoMetadata(**{k: v for k, v in stored.items() if k in known})


def read_metadata(path: str | Path) -> PhotoMetadata:
    """Read the EXIF of a file. Never writes, never raises on a bad file.

    Args:
        path: the RAW (or JPEG) to read. Opened read-only.

    Returns:
        A :class:`PhotoMetadata`, possibly empty. An unreadable or untagged file
        is a degraded import, not a failed one.
    """
    source = Path(path).expanduser()
    if is_intermediate(source):
        return _intermediate_metadata(source)
    meta = PhotoMetadata()
    try:
        import pyexiv2
    except Exception as exc:  # pragma: no cover - depends on the system libraries
        _log.info("exiv2 non disponibile, metadati non letti: %s", exc)
        return meta

    handle = None
    try:
        handle = pyexiv2.Image(str(source))
        tags = handle.read_exif()
    except Exception as exc:
        _log.warning("EXIF illeggibili in %s: %s", source.name, exc)
        return meta
    finally:
        if handle is not None:
            handle.close()

    meta.raw_tags = tags
    meta.camera_make = _first(tags, "Exif.Image.Make")
    meta.camera_model = _first(tags, "Exif.Image.Model")
    meta.lens_model = _first(
        tags, "Exif.Photo.LensModel", "Exif.Sony1.LensSpec", "Exif.Image.LensInfo"
    )
    meta.iso = _as_int(
        _first(tags, "Exif.Photo.ISOSpeedRatings", "Exif.Photo.RecommendedExposureIndex")
    )
    meta.aperture = _as_float(_first(tags, "Exif.Photo.FNumber", "Exif.Photo.ApertureValue"))
    meta.shutter = _as_float(_first(tags, "Exif.Photo.ExposureTime"))
    meta.focal_length = _as_float(_first(tags, "Exif.Photo.FocalLength"))
    meta.exposure_bias = _as_float(_first(tags, "Exif.Photo.ExposureBiasValue"))
    meta.shot_at = _as_datetime(
        _first(tags, "Exif.Photo.DateTimeOriginal", "Exif.Image.DateTime")
    )
    meta.orientation = _as_int(_first(tags, "Exif.Image.Orientation"))
    meta.width = _as_int(_first(tags, "Exif.Photo.PixelXDimension", "Exif.Image.ImageWidth"))
    meta.height = _as_int(_first(tags, "Exif.Photo.PixelYDimension", "Exif.Image.ImageLength"))
    meta.has_gps = any(key.startswith("Exif.GPSInfo.") for key in tags)
    return meta


def read_optics(path: str | Path) -> PhotoMetadata:
    """Body, lens, focal length and aperture -- what the decode needs to know.

    Its own function because ``decode.py`` calls it on every file, to pick the
    exposure anchor and the lens profile, and building a full
    :class:`PhotoMetadata` with every tag in ``raw_tags`` for that is a waste of
    the import budget of section 12. The fields it does not read stay ``None``.

    Never raises: an unreadable file gives an empty result, which the caller
    treats as "unknown body, unknown lens" rather than as an error.
    """
    if is_intermediate(path):
        full = _intermediate_metadata(Path(path))
        return PhotoMetadata(
            camera_make=full.camera_make, camera_model=full.camera_model,
            lens_model=full.lens_model, aperture=full.aperture, focal_length=full.focal_length,
            optics_corrected=full.optics_corrected,
        )
    meta = PhotoMetadata()
    try:
        import pyexiv2
    except Exception:  # pragma: no cover - depends on the system libraries
        return meta

    handle = None
    try:
        handle = pyexiv2.Image(str(Path(path).expanduser()))
        tags = handle.read_exif()
    except Exception as exc:
        _log.debug("EXIF non leggibili da %s: %s", path, exc)
        return meta
    finally:
        if handle is not None:
            handle.close()
    meta.camera_make = _first(tags, "Exif.Image.Make")
    meta.camera_model = (_first(tags, "Exif.Image.Model") or "").strip() or None
    meta.lens_model = _first(tags, "Exif.Photo.LensModel")
    meta.aperture = _as_float(_first(tags, "Exif.Photo.FNumber", "Exif.Photo.ApertureValue"))
    meta.focal_length = _as_float(_first(tags, "Exif.Photo.FocalLength"))
    return meta


def read_camera_model(path: str | Path) -> str | None:
    """Just the body name, as EXIF spells it -- ``ILCE-7M3`` for an A7 III.

    Returns ``None`` when exiv2 is missing or the file carries no model, which
    the caller must treat as "unknown body" rather than as an error.
    """
    return read_optics(path).camera_model


@dataclass(slots=True)
class ReferenceIdentity:
    """What an edited file remembers about the RAW it came from (section 8.1).

    Editors rename on export -- the user's Lightroom writes
    ``evento_001.jpg`` for ``DSC06312.ARW`` -- so the base name is often
    useless for pairing. The XMP usually is not: Lightroom and Camera Raw keep
    the original file name (``xmpMM:PreservedFileName``, ``crs:RawFileName``),
    and every editor keeps the shooting time.
    """

    original_name: str | None = None
    shot_at: datetime | None = None
    subsec: str | None = None
    camera_model: str | None = None


def read_reference_identity(path: str | Path) -> ReferenceIdentity:
    """Read the pairing clues of an edited JPEG/TIFF. Never writes, never raises."""
    identity = ReferenceIdentity()
    try:
        import pyexiv2
    except Exception:  # pragma: no cover - depends on the system libraries
        return identity
    handle = None
    try:
        handle = pyexiv2.Image(str(Path(path).expanduser()))
        exif = handle.read_exif()
        try:
            xmp = handle.read_xmp()
        except Exception:
            xmp = {}
    except Exception as exc:
        _log.debug("metadati non leggibili da %s: %s", path, exc)
        return identity
    finally:
        if handle is not None:
            handle.close()
    identity.original_name = _first(
        xmp, "Xmp.xmpMM.PreservedFileName", "Xmp.crs.RawFileName", "Xmp.darktable.import_filename"
    )
    identity.shot_at = _as_datetime(
        _first(exif, "Exif.Photo.DateTimeOriginal")
        or (_first(xmp, "Xmp.exif.DateTimeOriginal", "Xmp.photoshop.DateCreated") or "").replace(
            "T", " "
        )
    )
    identity.subsec = _first(exif, "Exif.Photo.SubSecTimeOriginal")
    identity.camera_model = (_first(exif, "Exif.Image.Model") or "").strip() or None
    return identity
