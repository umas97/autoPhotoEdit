# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The options of the Export screen (section 10.6), as one validated value.

Four of them have columns of their own since phase 2 -- destination, template,
collision policy, GPS -- because section 11 names them. The rest live in
``Project.export_settings``. Either way the screen and a new batch see one
:class:`ExportSettings`, and a batch keeps a frozen copy of it: changing the
format while a batch runs changes the next batch, not the photos still queued.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from ..db.enums import ExportConflict
from ..pipeline.colorspace import OutputSpace
from .image import ExportFormat
from .naming import DEFAULT_TEMPLATE

__all__ = [
    "ExportSettings",
    "OutputSharpening",
    "SHARPENING_AMOUNT",
    "load_settings",
    "store_settings",
]


class OutputSharpening(StrEnum):
    """Sharpening for the output medium, on top of the photo's own (section 10.6)."""

    NONE = "none"
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


#: Unsharp-mask amount of each level, at a radius of about one output pixel.
#: The three steps of Lightroom's "sharpen for screen", which is the convention
#: users already know; 0.6 at 0.8 px is visibly crisper on a 2048 px web export
#: and invisible as halo at 100%.
SHARPENING_AMOUNT = {
    OutputSharpening.NONE: 0.0,
    OutputSharpening.LOW: 0.3,
    OutputSharpening.STANDARD: 0.6,
    OutputSharpening.HIGH: 1.0,
}


class ExportSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    output_dir: str | None = None
    template: str = DEFAULT_TEMPLATE
    on_conflict: ExportConflict = ExportConflict.ASK
    strip_gps: bool = False

    #: Develop and write the image. Off to export only the sidecars.
    images: bool = True
    format: ExportFormat = ExportFormat.JPEG
    quality: Annotated[int, Field(ge=1, le=100)] = 92
    output_space: OutputSpace = OutputSpace.SRGB
    #: Long edge in pixels; ``None`` keeps the native resolution.
    long_edge: Annotated[int | None, Field(ge=256, le=20000)] = None
    sharpening: OutputSharpening = OutputSharpening.NONE
    dither: bool = False

    xmp_darktable: bool = False
    xmp_adobe: bool = False
    #: Write the sidecars next to the RAW files too. Off by default, and only
    #: ever *creates* files there (section 2.4, ``safety.create_beside_source``).
    xmp_beside_raw: bool = False

    #: Export only the photos the review approved; otherwise every photo kept.
    only_approved: bool = False

    @property
    def extension(self) -> str:
        return "jpg" if self.format is ExportFormat.JPEG else "tif"

    @property
    def writes_anything(self) -> bool:
        return self.images or self.xmp_darktable or self.xmp_adobe


#: The fields that are columns of ``Project`` rather than keys of its JSON.
_COLUMNS = {
    "output_dir": "output_dir",
    "template": "export_template",
    "on_conflict": "export_on_conflict",
    "strip_gps": "export_strip_gps",
}


def load_settings(project: Any) -> ExportSettings:
    """The project's export settings, defaults filled in."""
    stored = dict(getattr(project, "export_settings", None) or {})
    for field, column in _COLUMNS.items():
        value = getattr(project, column, None)
        if value is not None:
            stored[field] = value
    return ExportSettings.model_validate(stored)


def store_settings(project: Any, settings: ExportSettings) -> None:
    """Write the settings back: four columns and one JSON document."""
    data = settings.model_dump(mode="json")
    for field, column in _COLUMNS.items():
        value = data.pop(field)
        if field == "output_dir" and value:
            value = str(Path(value).expanduser())
        setattr(project, column, value if field != "on_conflict" else ExportConflict(value))
    project.export_settings = data
