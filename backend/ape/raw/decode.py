# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""RAW decoding: LibRaw bytes in, linear scene-referred Rec.2020 float32 out.

This is the only module that opens a RAW file, and it opens it read-only
(docs/SPEC.md section 2.1). It produces the input of the pipeline and nothing else:
no tone mapping, no gamma, no auto-brightening.

**Exposure anchoring.** LibRaw normalises so that sensor saturation is 1.0, and
our tone mapping is anchored on a scene-linear middle grey of 0.1845. Where the
camera puts middle grey below saturation is therefore the one number that
decides whether a neutral development comes out at the brightness the
photographer metered for. It is applied at decode as ``baseline_exposure_ev``
and reported in the result, so it can be inspected and overridden.

ISO 12232 answers the question in theory: the metered grey sits at 10/78 =
0.128 of the saturation exposure, which is +0.53 EV from our anchor. Every
manufacturer then keeps its own margin above that, so the number is *measured*
per body, by asking the camera: each ARW carries the JPEG the camera itself
would have produced, and the sensor level it renders at middle grey is Sony's
own answer. ``tests/bench/measure_baseline_exposure.py`` does the asking; the
table below is what it found. An unknown body falls back to the standard, which
is the honest default -- conservative, and wrong by a known amount rather than
by an unknown one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..lensdb import LensIdentity
from ..pipeline.colorspace import MIDDLE_GREY, XYZ_TO_REC2020, apply_matrix
from .whitepoint import (
    camera_multipliers_to_illuminant,
    normalise_camera_matrix,
    xyz_to_cct_tint,
)

__all__ = [
    "DecodedRaw",
    "baseline_exposure_for",
    "decode_intermediate",
    "decode_linear",
    "read_camera_color",
]

#: Scene-linear position of middle grey relative to sensor saturation, ISO 12232.
ISO_MIDDLE_GREY = 10.0 / 78.0

#: Standards-derived offset between the LibRaw normalisation and our anchor.
#: The fallback for a body nobody has measured yet.
BASELINE_EXPOSURE_EV = math.log2(MIDDLE_GREY / ISO_MIDDLE_GREY)

#: ``EXIF model -> stops``, measured with ``tests/bench/measure_baseline_exposure.py``.
#:
#: ILCE-7M3 (A7 III): +1.09 EV, the median over 21 frames from ISO 100 to 1600,
#: standard deviation 0.058 EV and a full range of 0.86..1.16. That tightness
#: across different scenes and sensitivities is what says it is a property of
#: the camera rather than of the pictures. Sony keeps about half a stop more
#: headroom above middle grey than ISO 12232 assumes, so the standard number
#: renders a correctly metered Sony frame visibly dark.
MEASURED_BASELINE_EV: dict[str, float] = {
    "ILCE-7M3": 1.09,
}


def baseline_exposure_for(camera_model: str | None) -> float:
    """Stops between LibRaw's normalisation and our middle grey, for one body."""
    if camera_model:
        measured = MEASURED_BASELINE_EV.get(camera_model.strip().upper())
        if measured is not None:
            return measured
    return BASELINE_EXPOSURE_EV


# D65, the white point of the Rec.2020 working space.
_D65_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

_RAW_MAX = 65535.0


@dataclass(frozen=True)
class CameraColor:
    """Everything colour-related that is specific to this camera body."""

    #: Camera matrix as LibRaw reports it (XYZ -> camera), unnormalised.
    cam_from_xyz: np.ndarray
    #: Camera -> XYZ, renormalised so the as-shot neutral lands exactly on D65.
    xyz_from_cam: np.ndarray
    #: As-shot multipliers, normalised to green = 1.
    as_shot_multipliers: np.ndarray
    as_shot_temperature_k: float
    as_shot_tint: float


@dataclass
class DecodedRaw:
    """A decoded frame plus the context the pipeline needs to interpret it."""

    #: ``(H, W, 3)`` float32, linear, scene-referred, Rec.2020 primaries, D65.
    rgb: np.ndarray
    camera: CameraColor
    #: Applied by the renderer before the user's own exposure.
    baseline_exposure_ev: float = BASELINE_EXPOSURE_EV
    source_path: Path | None = None
    half_size: bool = False
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    #: The optics that took the frame, for the lens correction stage. ``None``
    #: for a synthetic or merged frame, which has no lens to correct.
    lens: LensIdentity | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[0], self.rgb.shape[1]


def _camera_matrix(raw: Any) -> np.ndarray:
    """The 3x3 XYZ -> camera matrix, with a sane fallback for unknown bodies."""
    matrix = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix)) < 1e-9:
        # LibRaw has no profile for this body. sRGB primaries are wrong but
        # invertible, and being visibly off beats crashing on import.
        from ..pipeline.colorspace import XYZ_TO_RGB, OutputSpace

        return np.asarray(XYZ_TO_RGB[OutputSpace.SRGB], dtype=np.float64)
    return matrix


def read_camera_color(raw: Any) -> CameraColor:
    """Derive the colour context from an open ``rawpy`` handle."""
    cam_from_xyz = _camera_matrix(raw)

    multipliers = np.asarray(raw.camera_whitebalance, dtype=np.float64)[:3]
    if np.any(multipliers <= 0) or not np.all(np.isfinite(multipliers)):
        multipliers = np.asarray(raw.daylight_whitebalance, dtype=np.float64)[:3]
    if np.any(multipliers <= 0) or not np.all(np.isfinite(multipliers)):
        multipliers = np.ones(3, dtype=np.float64)
    multipliers = multipliers / multipliers[1]

    try:
        illuminant = camera_multipliers_to_illuminant(multipliers, cam_from_xyz)
        temperature, tint = xyz_to_cct_tint(illuminant)
    except (ValueError, np.linalg.LinAlgError):
        # Unknown or nonsensical white balance: fall back to daylight rather
        # than refusing the file. The UI flags it through the confidence score.
        temperature, tint = 5500.0, 0.0

    # Normalising on D65 is what makes the as-shot neutral come out neutral: the
    # multipliers have already put the scene illuminant on the camera's neutral
    # axis, and this puts that axis on the working space's white point.
    xyz_from_cam = np.linalg.inv(normalise_camera_matrix(cam_from_xyz, _D65_XYZ))

    return CameraColor(
        cam_from_xyz=cam_from_xyz,
        xyz_from_cam=xyz_from_cam,
        as_shot_multipliers=multipliers,
        as_shot_temperature_k=temperature,
        as_shot_tint=tint,
    )


#: Rows converted at a time from the decoder's integers to the working space:
#: the float copy and the matrix product of a band stay in cache, and a 24 MP
#: decode holds one float frame instead of two.
_CONVERT_ROWS = 64


def _to_working(rgb16: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Sensor integers -> float32 linear Rec.2020, band by band.

    Row by row the arithmetic is that of the whole frame, so the result is the
    same to the bit (checked on the fixtures when this was introduced).
    """
    out = np.empty(rgb16.shape, dtype=np.float32)
    scale = np.float32(_RAW_MAX)
    for top in range(0, rgb16.shape[0], _CONVERT_ROWS):
        band = rgb16[top : top + _CONVERT_ROWS].astype(np.float32)
        band /= scale
        out[top : top + _CONVERT_ROWS] = apply_matrix(band, matrix)
    return out


def decode_linear(
    path: str | Path,
    *,
    half_size: bool = False,
    quality: bool = False,
    white_balance: np.ndarray | None = None,
    lens_override: tuple[str, str] | None = None,
) -> DecodedRaw:
    """Decode a RAW file into the linear working space.

    Args:
        path: the RAW file. Opened read-only and never modified.
        half_size: skip demosaicing and emit one pixel per sensor site. Four
            times faster and a quarter of the memory; used for proxies and for
            the group detection of section 25.
        quality: use DCB instead of AHD. Slower, slightly better on fine detail.
        white_balance: multipliers to demosaic with, defaulting to the camera's
            own. Demosaicing on a white-balanced signal is what keeps AHD from
            inventing colour fringes, so this happens here rather than later;
            the user's own white balance is a chromatic adaptation applied
            downstream, on top of this one.
        lens_override: ``(maker, model)`` of the lensfun profile the user
            associated with this lens by hand (``LensProfileOverride``), which
            only the catalogue knows about.

    Returns:
        A :class:`DecodedRaw` whose ``rgb`` is float32, linear, scene-referred,
        Rec.2020 primaries, D65 white, with sensor saturation at 1.0.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if LibRaw cannot make sense of the file.
    """
    import rawpy

    from .intermediate import is_intermediate
    from .metadata import read_optics

    source = Path(path).expanduser()
    if is_intermediate(source):
        return decode_intermediate(source, reduced=half_size, lens_override=lens_override)
    if not source.is_file():
        raise FileNotFoundError(f"file RAW non trovato: {source}")

    # Which body this is decides the exposure anchor, and body plus lens the
    # correction profile. Read before the pixels so that a file exiv2 cannot
    # parse still decodes, on the standard number and without correction.
    optics = read_optics(source)
    camera_model = optics.camera_model

    try:
        with rawpy.imread(str(source)) as raw:
            camera = read_camera_color(raw)
            mults = camera.as_shot_multipliers if white_balance is None else white_balance
            # LibRaw wants four multipliers (the second green of a Bayer quad).
            user_wb = [float(mults[0]), float(mults[1]), float(mults[2]), float(mults[1])]

            algorithm = (
                rawpy.DemosaicAlgorithm.DCB if quality else rawpy.DemosaicAlgorithm.AHD
            )
            rgb16 = raw.postprocess(
                output_bps=16,
                gamma=(1.0, 1.0),
                no_auto_bright=True,
                use_camera_wb=False,
                use_auto_wb=False,
                user_wb=user_wb,
                output_color=rawpy.ColorSpace.raw,
                demosaic_algorithm=algorithm,
                half_size=half_size,
                # Clip rather than let LibRaw invent highlight data: our own
                # recovery stage does that, with parameters the user can see.
                highlight_mode=rawpy.HighlightMode.Clip,
                median_filter_passes=0,
                four_color_rgb=False,
            )
            raw_metadata = {
                "raw_width": int(raw.sizes.raw_width),
                "raw_height": int(raw.sizes.raw_height),
                "white_level": int(raw.white_level),
                "black_level_per_channel": [int(v) for v in raw.black_level_per_channel],
                "num_colors": int(raw.num_colors),
            }
    except rawpy.LibRawError as exc:  # pragma: no cover - needs a corrupt file
        raise ValueError(f"file RAW illeggibile: {source.name} ({exc})") from exc

    matrix = XYZ_TO_REC2020 @ camera.xyz_from_cam
    rgb = _to_working(rgb16, matrix)
    del rgb16

    return DecodedRaw(
        rgb=rgb,
        camera=camera,
        baseline_exposure_ev=baseline_exposure_for(camera_model),
        source_path=source,
        half_size=half_size,
        raw_metadata=raw_metadata,
        lens=LensIdentity(
            camera_maker=optics.camera_make,
            camera_model=optics.camera_model,
            lens_model=optics.lens_model,
            focal_length=optics.focal_length,
            aperture=optics.aperture,
            override=lens_override,
        ),
    )


def decode_intermediate(
    path: str | Path, *, reduced: bool = False, lens_override: tuple[str, str] | None = None
) -> DecodedRaw:
    """The buffer of a merged photo, as the decode of its reference frame.

    Section 25.1: the merge replaces the decode stage and nothing else. The
    camera colour, exposure anchor and optics are the reference frame's, read
    back from the intermediate itself, so the renderer treats the merge exactly
    as it would treat that shot. ``reduced`` reads the copy at the proxy's long
    edge, which is what a half-size decode is for.
    """
    from .intermediate import read_frame

    rgb, context = read_frame(path, reduced=reduced)
    colour = context["camera"]
    camera = CameraColor(
        cam_from_xyz=np.asarray(colour["cam_from_xyz"], dtype=np.float64),
        xyz_from_cam=np.asarray(colour["xyz_from_cam"], dtype=np.float64),
        as_shot_multipliers=np.asarray(colour["as_shot_multipliers"], dtype=np.float64),
        as_shot_temperature_k=float(colour["as_shot_temperature_k"]),
        as_shot_tint=float(colour["as_shot_tint"]),
    )
    lens = context.get("lens")
    identity = None
    if lens is not None:
        identity = LensIdentity(
            camera_maker=lens.get("camera_maker"),
            camera_model=lens.get("camera_model"),
            lens_model=lens.get("lens_model"),
            focal_length=lens.get("focal_length"),
            aperture=lens.get("aperture"),
            distance=float(lens.get("distance", 1000.0)),
            override=lens_override,
        )
    return DecodedRaw(
        rgb=rgb,
        camera=camera,
        baseline_exposure_ev=float(context["baseline_exposure_ev"]),
        source_path=Path(path),
        half_size=reduced,
        raw_metadata=dict(context.get("raw_metadata") or {}),
        lens=identity,
    )


def decoded_from_array(
    rgb: np.ndarray,
    *,
    temperature_k: float = 5500.0,
    tint: float = 0.0,
    baseline_exposure_ev: float = 0.0,
) -> DecodedRaw:
    """Wrap an already-linear Rec.2020 array as a :class:`DecodedRaw`.

    Used by the tests, by the synthetic fixtures, and -- from phase 11 -- by the
    multi-shot merges, whose intermediate buffer replaces the decode stage and
    nothing else (docs/SPEC.md section 25.1).
    """
    identity = np.eye(3, dtype=np.float64)
    camera = CameraColor(
        cam_from_xyz=identity,
        xyz_from_cam=identity,
        as_shot_multipliers=np.ones(3, dtype=np.float64),
        as_shot_temperature_k=temperature_k,
        as_shot_tint=tint,
    )
    return DecodedRaw(
        rgb=np.ascontiguousarray(rgb, dtype=np.float32),
        camera=camera,
        baseline_exposure_ev=baseline_exposure_ev,
    )
