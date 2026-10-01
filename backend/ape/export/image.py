# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Writing the developed pixels out as JPEG or TIFF.

Everything that leaves the program goes through here, and everything that leaves
the program goes through ``safety.guarded_open`` first: this is one of the two
modules that are allowed to create files at all, and the source folder is not a
place either of them may write (docs/SPEC.md section 2).

Quantisation is a plain round, with an optional ordered dither. The dither is
off by default because it changes the pixels, and a default that changes pixels
is a decision for the user rather than for the implementation; it is worth
turning on for 8-bit exports of skies and other smooth gradients, where a
quarter of a code value of patterned noise is far less visible than the banding
it replaces.
"""

from __future__ import annotations

import io
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..pipeline.colorspace import OutputSpace
from ..safety import guarded_open
from .icc import profile_for
from .tiff import write_tiff

__all__ = ["ExportFormat", "SOFTWARE_TAG", "encode_image", "quantise", "save_image"]

#: Value written to the Software / CreatorTool tags (section 16.3).
SOFTWARE_TAG = f"autoPhotoEdit {__version__}"


class ExportFormat(StrEnum):
    JPEG = "jpeg"
    TIFF8 = "tiff8"
    TIFF16 = "tiff16"

    @property
    def suffix(self) -> str:
        return ".jpg" if self is ExportFormat.JPEG else ".tif"

    @property
    def bit_depth(self) -> int:
        return 16 if self is ExportFormat.TIFF16 else 8


# 8x8 Bayer matrix, normalised to [-0.5, 0.5). Ordered rather than random so the
# output stays byte-for-byte reproducible (test 4 of section 13).
_BAYER8 = (
    np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    )
    / 64.0
    - 0.5
)


def quantise(image: np.ndarray, bit_depth: int, dither: bool = False) -> np.ndarray:
    """Float [0, 1] -> unsigned integers of the requested depth.

    Args:
        image: ``(H, W, 3)`` float array, already through the output transfer
            function. Values outside [0, 1] are clipped.
        bit_depth: 8 or 16.
        dither: add an ordered sub-LSB pattern before rounding.

    Returns:
        ``(H, W, 3)`` uint8 or uint16.
    """
    if bit_depth not in (8, 16):
        raise ValueError(f"unsupported bit depth {bit_depth}")
    peak = float((1 << bit_depth) - 1)
    scaled = np.clip(image, 0.0, 1.0).astype(np.float32) * peak
    if dither:
        height, width = scaled.shape[:2]
        tile = np.tile(_BAYER8, (height // 8 + 1, width // 8 + 1))[:height, :width]
        scaled = scaled + tile[..., None]
    rounded = np.clip(np.rint(scaled), 0.0, peak)
    return rounded.astype(np.uint8 if bit_depth == 8 else np.uint16)


def encode_image(
    image: np.ndarray,
    fmt: ExportFormat,
    *,
    output_space: OutputSpace = OutputSpace.SRGB,
    quality: int = 92,
    dither: bool = False,
    embed_profile: bool = True,
) -> bytes:
    """Encode a rendered float image into the bytes of a file.

    Args:
        image: ``(H, W, 3)`` float in [0, 1], already in ``output_space`` and
            through its transfer function -- that is, the output of
            ``pipeline.render.render``.
        fmt: JPEG, 8-bit TIFF or 16-bit TIFF.
        output_space: which ICC profile to embed.
        quality: JPEG quality, 1-100. Ignored for TIFF.
        dither: see :func:`quantise`.
        embed_profile: attach the ICC profile. Off only for tests.

    Returns:
        The complete file contents.
    """
    profile = profile_for(output_space) if embed_profile else None
    data = quantise(image, fmt.bit_depth, dither=dither)

    if fmt is not ExportFormat.JPEG:
        return write_tiff(data, icc_profile=profile, software=SOFTWARE_TAG)

    from PIL import Image

    buffer = io.BytesIO()
    save_args: dict[str, Any] = {
        "format": "JPEG",
        "quality": int(np.clip(quality, 1, 100)),
        "optimize": True,
        "progressive": True,
        # 4:4:4. Chroma subsampling on a photographic export throws away colour
        # detail to save a few percent of a file nobody is streaming.
        "subsampling": 0,
    }
    if profile:
        save_args["icc_profile"] = profile
    Image.fromarray(data, mode="RGB").save(buffer, **save_args)
    return buffer.getvalue()


def save_image(
    image: np.ndarray,
    path: str | Path,
    fmt: ExportFormat | None = None,
    *,
    source: Any = None,
    **kwargs: Any,
) -> Path:
    """Encode and write to disk, refusing any destination inside a source folder.

    Args:
        image: the rendered float image.
        path: where to write. Its suffix picks the format when ``fmt`` is None.
        fmt: explicit format.
        source: project or source directory to guard against (section 2.3).
        **kwargs: forwarded to :func:`encode_image`.

    Returns:
        The resolved path that was written.

    Raises:
        SourceWriteError: if the destination falls inside a protected folder.
    """
    target = Path(path).expanduser()
    if fmt is None:
        jpeg_suffixes = (".jpg", ".jpeg")
        fmt = (
            ExportFormat.JPEG
            if target.suffix.lower() in jpeg_suffixes
            else ExportFormat.TIFF16
        )

    payload = encode_image(image, fmt, **kwargs)
    with guarded_open(target, "wb", source=source) as handle:
        handle.write(payload)
    return target.resolve()
