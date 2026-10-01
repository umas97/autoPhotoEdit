# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the XMP translators need to know about a shot, and their common entry point.

A sidecar describes how to develop *the RAW*, not the exported JPEG, so it is
named after the RAW, the way each program looks for it next to the file:

* darktable: ``DSC06312.ARW.xmp`` -- the full name plus ``.xmp``;
* Lightroom / Camera Raw: ``DSC06312.xmp`` -- the stem plus ``.xmp``.

The two can therefore sit side by side, in the export folder or (behind the
option of section 2.4) next to the RAW.

Translating ``EditParams`` into another program's controls is approximate by
nature -- different tone curves, different colour models, different ideas of
what "+20 shadows" means. Each translator says in its docstring what it maps,
how, and what it drops. What is *not* approximate is the geometry and the white
balance, which is why the context below carries the camera's colour matrix:
darktable's white balance is a set of channel multipliers, and the only way to
compute the multipliers of a temperature is through the sensor's own matrix.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ..pipeline.params import EditParams

__all__ = [
    "SIDECAR_KINDS",
    "SidecarContext",
    "build_sidecar",
    "context_from_decoded",
    "read_sidecar_context",
    "sidecar_name",
]

SIDECAR_KINDS = ("darktable", "adobe")


@dataclass(slots=True)
class SidecarContext:
    """The shot a sidecar is for."""

    raw_filename: str
    #: XYZ -> camera, 3x3, as LibRaw reports it (``CameraColor.cam_from_xyz``).
    cam_from_xyz: np.ndarray
    #: The camera's own white balance, green-normalised.
    as_shot_multipliers: np.ndarray
    as_shot_temperature_k: float
    as_shot_tint: float
    camera_maker: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    focal_length: float | None = None
    aperture: float | None = None
    shot_at: datetime | None = None
    #: Pixel size of the *upright* picture, as the renderer sees it.
    width: int | None = None
    height: int | None = None
    #: EXIF orientation of the RAW (1 upright, 6 and 8 portrait).
    orientation: int | None = None
    #: Program version, for the ``CreatorTool`` of the packet.
    software: str = ""
    #: The decoder's exposure anchor (``raw/decode.py``), which our renders
    #: add to ``exposure.ev`` and another program has to be told about.
    baseline_exposure_ev: float = 0.0
    #: The optics as the lens correction resolves them, override included.
    lens_identity: object | None = None


def sidecar_name(kind: str, raw_filename: str) -> str:
    """The file name each program looks for next to the RAW."""
    if kind == "darktable":
        return f"{raw_filename}.xmp"
    if kind == "adobe":
        return f"{Path(raw_filename).stem}.xmp"
    raise ValueError(f"tipo di sidecar sconosciuto: {kind}")


def context_from_decoded(decoded, raw_filename: str, **fields) -> SidecarContext:
    """Build the context from a frame the export has already decoded."""
    camera = decoded.camera
    lens = decoded.lens
    fields.setdefault("baseline_exposure_ev", float(decoded.baseline_exposure_ev))
    fields.setdefault("lens_identity", lens)
    return SidecarContext(
        raw_filename=raw_filename,
        cam_from_xyz=np.asarray(camera.cam_from_xyz, dtype=np.float64),
        as_shot_multipliers=np.asarray(camera.as_shot_multipliers, dtype=np.float64),
        as_shot_temperature_k=float(camera.as_shot_temperature_k),
        as_shot_tint=float(camera.as_shot_tint),
        camera_maker=getattr(lens, "camera_maker", None),
        camera_model=getattr(lens, "camera_model", None),
        lens_model=getattr(lens, "lens_model", None),
        focal_length=getattr(lens, "focal_length", None),
        aperture=getattr(lens, "aperture", None),
        **fields,
    )


def read_sidecar_context(
    path: str | Path, *, lens_override: tuple[str, str] | None = None, **fields
) -> SidecarContext:
    """Read the context straight from the RAW, without demosaicing it.

    For a batch that writes only sidecars: opening the file with LibRaw costs a
    few milliseconds, the decode it replaces a second and a half.

    Raises:
        FileNotFoundError, ValueError: as ``raw/decode.py``.
    """
    import rawpy

    from ..lensdb import LensIdentity
    from ..raw.decode import baseline_exposure_for, read_camera_color
    from ..raw.metadata import read_optics

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"file RAW non trovato: {source}")
    optics = read_optics(source)
    try:
        with rawpy.imread(str(source)) as raw:
            camera = read_camera_color(raw)
            width, height = int(raw.sizes.width), int(raw.sizes.height)
            # LibRaw's flip: 5 and 6 are the quarter turns, whose upright
            # picture is the sensor frame on its side.
            flip = int(raw.sizes.flip)
            if flip in (5, 6):
                width, height = height, width
            fields.setdefault("orientation", {3: 3, 5: 8, 6: 6}.get(flip, 1))
    except rawpy.LibRawError as exc:
        raise ValueError(f"file RAW illeggibile: {source.name} ({exc})") from exc
    fields.setdefault("width", width)
    fields.setdefault("height", height)
    fields.setdefault("baseline_exposure_ev", baseline_exposure_for(optics.camera_model))
    fields.setdefault(
        "lens_identity",
        LensIdentity(
            camera_maker=optics.camera_make,
            camera_model=optics.camera_model,
            lens_model=optics.lens_model,
            focal_length=optics.focal_length,
            aperture=optics.aperture,
            override=lens_override,
        ),
    )
    return SidecarContext(
        raw_filename=source.name,
        cam_from_xyz=np.asarray(camera.cam_from_xyz, dtype=np.float64),
        as_shot_multipliers=np.asarray(camera.as_shot_multipliers, dtype=np.float64),
        as_shot_temperature_k=float(camera.as_shot_temperature_k),
        as_shot_tint=float(camera.as_shot_tint),
        camera_maker=optics.camera_make,
        camera_model=optics.camera_model,
        lens_model=optics.lens_model,
        focal_length=optics.focal_length,
        aperture=optics.aperture,
        **fields,
    )


def _builders() -> dict[str, Callable[[EditParams, SidecarContext], bytes]]:
    from . import xmp_adobe, xmp_darktable

    return {"darktable": xmp_darktable.build, "adobe": xmp_adobe.build}


def build_sidecar(kind: str, params: EditParams, ctx: SidecarContext) -> bytes:
    """The complete XMP file of one kind, UTF-8."""
    return _builders()[kind](params, ctx)
