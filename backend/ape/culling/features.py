# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One photo's culling analysis, from preview to stored features.

``analyse_preview`` is the whole per-photo half of culling: the part that runs
once per file, in a worker, and that the project-wide half (grouping and
selection, in ``service.py``) then reads back without ever touching the file
again. Everything a later recomputation could want is in the dict it returns,
so that switching a criterion on or off, or moving a slider, never needs the
photo a second time (section 7.3).

The Sony-specific tags come in through ``PhotoMetadata.raw_tags``: this module
reads what ``raw/metadata.py`` already extracted and never imports exiv2 itself
(section 24).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..merge.features import sharpness_map
from ..raw.embedded import EmbeddedPreview
from ..raw.metadata import PhotoMetadata
from .burst import visual_signature
from .technical import TechnicalScores, analysis_gray, technical_scores

__all__ = ["FEATURES_VERSION", "PreviewAnalysis", "analyse_preview", "camera_details"]

#: Bumped when the stored features change meaning. A photo analysed by an older
#: build is re-analysed rather than compared with numbers it cannot match.
FEATURES_VERSION = 1


def _int_tag(tags: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = tags.get(name)
        if value in (None, ""):
            continue
        try:
            return int(str(value).split()[0])
        except (TypeError, ValueError):
            continue
    return None


def _focus_point(tags: dict[str, Any], flip: int) -> tuple[float, float] | None:
    """Where the camera focused, as fractions of the upright frame.

    Sony records ``FocusLocation`` as ``width height x y`` in sensor
    coordinates, before any rotation. It is what the comparison view zooms to:
    the point the photographer chose is the point worth checking at 1:1.
    """
    raw = tags.get("Exif.Sony2.FocusLocation")
    if not raw:
        return None
    try:
        width, height, x, y = (float(v) for v in str(raw).split()[:4])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or (x == 0 and y == 0):
        return None
    u, v = x / width, y / height
    # The same rotations as ``raw.embedded.orient``, applied to a point.
    if flip == 3:
        u, v = 1.0 - u, 1.0 - v
    elif flip == 5:
        u, v = v, 1.0 - u
    elif flip == 6:
        u, v = 1.0 - v, u
    return (round(min(1.0, max(0.0, u)), 4), round(min(1.0, max(0.0, v)), 4))


#: Exposure bracketing in each of the tags that can say so, by ExifTool's
#: tables. The two Sony tags number their modes differently -- in
#: ``Sony2.ReleaseMode`` 2 is a plain continuous burst -- so each is read with
#: its own table. ``Photo.ExposureMode`` is the standard EXIF tag (2 = "Auto
#: bracket"), written by the A7 III for exposure brackets and by other makers too.
_BRACKET_CODES = (
    ("Exif.SonyMisc3c.ReleaseMode2", frozenset({2, 23})),
    ("Exif.Sony2.ReleaseMode", frozenset({5})),
    ("Exif.Photo.ExposureMode", frozenset({2})),
)


def _camera_bracket(tags: dict[str, Any]) -> bool | None:
    known = False
    for name, codes in _BRACKET_CODES:
        value = _int_tag(tags, name)
        if value is None:
            continue
        if value in codes:
            return True
        known = True
    return False if known else None


def camera_details(meta: PhotoMetadata, flip: int = 0) -> dict[str, Any]:
    """The EXIF that grouping needs and the catalogue has no column for."""
    tags = meta.raw_tags or {}
    return {
        "exposure_bias": meta.exposure_bias,
        "release_mode": _int_tag(tags, "Exif.SonyMisc3c.ReleaseMode2", "Exif.Sony2.ReleaseMode"),
        # What the camera says about exposure bracketing (``merge/detect.py``).
        "camera_bracket": _camera_bracket(tags),
        "sequence_number": _int_tag(
            tags, "Exif.SonyMisc3c.SequenceImageNumber", "Exif.Sony2.SequenceNumber"
        ),
        "sequence_length": _int_tag(tags, "Exif.SonyMisc3c.SequenceLength1"),
        # Increases by one per frame since the camera was switched on: the
        # order of two frames taken in the same second, which the timestamp
        # cannot give.
        "shot_number": _int_tag(tags, "Exif.SonyMisc3c.ShotNumberSincePowerUp"),
        "focus_point": _focus_point(tags, flip),
        # Where the focus actuator was: moves frame to frame in a focus stack
        # (section 25.2). 255 when the camera did not say.
        "focus_position": _int_tag(tags, "Exif.Sony2Fp.FocusPosition2"),
    }


@dataclass(frozen=True)
class PreviewAnalysis:
    scores: TechnicalScores
    features: dict[str, Any]


def analyse_preview(
    preview: EmbeddedPreview, meta: PhotoMetadata | None = None, *, flip: int = 0
) -> PreviewAnalysis:
    """Score one preview and gather everything grouping will need.

    Args:
        preview: the upright camera preview.
        meta: the photo's EXIF, when there is any; a file without EXIF is
            analysed all the same and simply never joins a burst.
        flip: LibRaw's orientation code, to place the focus point.
    """
    image = np.ascontiguousarray(preview.image)
    gray = analysis_gray(image)
    scores = technical_scores(image, gray)
    features: dict[str, Any] = {
        "version": FEATURES_VERSION,
        "preview": {"source": preview.source, "width": preview.width, "height": preview.height},
        **scores.features(),
        "signature": visual_signature(image),
        "camera": camera_details(meta, flip) if meta is not None else {},
        # Which part of the frame is sharp: what tells a focus stack from a
        # burst of the same framing (section 25.2). Absent from photos
        # analysed before phase 11, which detection then judges on EXIF alone.
        "sharp_map": sharpness_map(gray),
    }
    return PreviewAnalysis(scores=scores, features=features)
