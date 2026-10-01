# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""One training pair, from two files on disk to everything a profile keeps.

``prepare`` does the per-pair work of section 8 that needs pixels, once:

* decode the RAW at half size, lens-corrected like every proxy;
* render it neutral at :data:`FEATURE_EDGE` -- the image the scene features,
  the embedding and the exposure anchor are measured on, exactly as they are on
  a project photo's browsing proxy (``analysis/scene.py``, ``style/auto.py``);
* read the reference, colour-managed into sRGB, and register it on the neutral
  rendering (``style/align.py``);
* resample both to :data:`COMPARE_EDGE` for the inversion (``style/learn.py``);
* keep a 512 px JPEG of the reference, which is what a portable profile
  carries instead of the files (section 20.1).

The RAW and the reference are opened read-only and never written (section 2).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..analysis.scene import SceneInputs, scene_features
from ..pipeline.filters import resize_long_edge
from ..pipeline.params import EditParams
from ..pipeline.render import RenderOptions, render
from ..raw.decode import DecodedRaw, decode_linear
from . import auto
from . import vector as sv
from .align import Alignment, align, warp_reference
from .problem import PairImages

__all__ = [
    "COMPARE_EDGE",
    "FEATURE_EDGE",
    "PreparedPair",
    "SceneDescription",
    "describe_neutral",
    "prepare",
    "read_reference",
]

#: Section 8.2.1: the comparison happens at a long edge of 512 px.
COMPARE_EDGE = 512

#: Where the scene is described. The features reduce to 512 and CLIP to 224,
#: so the 2048 of a proxy would only cost time; 1024 is also what the
#: registration works at.
FEATURE_EDGE = 1024

THUMBNAIL_EDGE = 512


@dataclass(slots=True)
class SceneDescription:
    """What a style knows about a photo before predicting anything for it."""

    features: np.ndarray
    embedding: np.ndarray | None
    context: sv.StyleContext
    auto: auto.AutoResult


@dataclass(slots=True)
class PreparedPair:
    images: PairImages
    scene: SceneDescription
    alignment: Alignment
    thumbnail_jpeg: bytes
    #: ``review/measure.py`` tails of the neutral frame: how much the user's
    #: edit of it burns and crushes beyond the scene (``review/reference.py``).
    tails: dict | None = None


def read_reference(path: str | Path) -> np.ndarray:
    """An edited JPEG or TIFF as upright sRGB, uint8 or uint16.

    The embedded ICC profile is honoured: an Adobe RGB export compared as if it
    were sRGB would teach the profile a desaturation that is not there. 16-bit
    TIFFs go through OpenCV, because Pillow silently reduces them to 8 bits;
    they are assumed sRGB, which is what every editor's default TIFF export is.

    Raises:
        ValueError: if the file cannot be read as an image.
    """
    from PIL import Image, ImageCms, ImageOps

    source = Path(path)
    if source.suffix.lower() in (".tif", ".tiff"):
        data = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if data is not None and data.dtype == np.uint16 and data.ndim == 3:
            return np.ascontiguousarray(data[..., 2::-1])
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            icc = image.info.get("icc_profile")
            image = image.convert("RGB")
            if icc:
                try:
                    profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                    srgb = ImageCms.createProfile("sRGB")
                    image = ImageCms.profileToProfile(
                        image, profile, srgb, renderingIntent=ImageCms.Intent.PERCEPTUAL
                    )
                except (ImageCms.PyCMSError, OSError):
                    pass  # a broken profile is read as sRGB, the common case anyway
            return np.asarray(image)
    except OSError as exc:
        raise ValueError(f"immagine illeggibile: {source.name} ({exc})") from exc


def _to_float(image: np.ndarray) -> np.ndarray:
    scale = 65535.0 if image.dtype == np.uint16 else 255.0
    return image.astype(np.float32) / np.float32(scale)


def _uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def describe_neutral(
    neutral: np.ndarray,
    exif: SceneInputs,
    *,
    as_shot_temperature_k: float,
    as_shot_tint: float,
    with_embedding: bool = True,
) -> SceneDescription:
    """Scene features, embedding and style context of a neutral rendering.

    The same function describes a training pair and a project photo, so the
    two can never drift apart.
    """
    from ..analysis import embed

    image = neutral if neutral.dtype == np.uint8 else _uint8(neutral)
    exif.as_shot_temperature_k = as_shot_temperature_k
    exif.as_shot_tint = as_shot_tint
    measured = auto.measure(
        image, as_shot_temperature_k=as_shot_temperature_k, as_shot_tint=as_shot_tint
    )
    return SceneDescription(
        features=scene_features(image, exif),
        embedding=embed.embed(image) if with_embedding else None,
        context=sv.StyleContext(
            as_shot_temperature_k=as_shot_temperature_k,
            as_shot_tint=as_shot_tint,
            exposure_anchor_ev=measured.exposure_anchor_ev,
        ),
        auto=measured,
    )


def _resized_copy(decoded: DecodedRaw, edge: int) -> DecodedRaw:
    from dataclasses import replace

    return replace(decoded, rgb=np.ascontiguousarray(resize_long_edge(decoded.rgb, edge)))


def _thumbnail(reference: np.ndarray) -> bytes:
    from PIL import Image

    small = reference
    if reference.dtype == np.uint16:
        small = (reference >> 8).astype(np.uint8)
    image = Image.fromarray(small)
    image.thumbnail((THUMBNAIL_EDGE, THUMBNAIL_EDGE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def prepare(
    raw_path: str | Path,
    reference_path: str | Path,
    *,
    lens_override: tuple[str, str] | None = None,
    with_embedding: bool = True,
) -> PreparedPair:
    """Everything the inversion and the profile need from one pair of files.

    Raises:
        FileNotFoundError: a file is missing.
        ValueError: a file cannot be read.
        AlignmentError: the reference is not the same shot, or too different.
    """
    from ..raw.metadata import read_metadata

    decoded = decode_linear(raw_path, half_size=True, lens_override=lens_override)
    meta = read_metadata(raw_path)
    exif = SceneInputs(
        iso=meta.iso,
        aperture=meta.aperture,
        shutter=meta.shutter,
        focal_length=meta.focal_length,
    )

    features_decoded = _resized_copy(decoded, FEATURE_EDGE)
    neutral = render(features_decoded, EditParams(), RenderOptions())
    camera = decoded.camera
    scene = describe_neutral(
        neutral,
        exif,
        as_shot_temperature_k=camera.as_shot_temperature_k,
        as_shot_tint=camera.as_shot_tint,
        with_embedding=with_embedding,
    )

    reference = read_reference(reference_path)
    alignment = align(neutral, reference)

    compare = _resized_copy(decoded, COMPARE_EDGE)
    height, width = compare.rgb.shape[:2]
    warped, mask = warp_reference(alignment, _to_float(reference), (height, width), erode_px=3)
    images = PairImages(decoded=compare, reference=warped, mask=mask, context=scene.context)
    from ..review.measure import neutral_tails

    return PreparedPair(
        images=images,
        scene=scene,
        alignment=alignment,
        thumbnail_jpeg=_thumbnail(reference),
        tails=neutral_tails(neutral),
    )
