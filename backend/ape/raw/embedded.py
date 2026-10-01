# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The JPEG the camera already made: what culling looks at (docs/SPEC.md section 7.2).

Culling must not decode RAW files. A Sony ARW carries a full development of
itself -- a 1616x1080 JPEG on the A7 III, written by the camera at the moment of
the shot -- and pulling it out costs a few milliseconds where a decode costs a
second. On two thousand files that is the whole difference between a culling
pass the user waits for and one they do not notice.

When the camera also wrote a JPEG next to the RAW (section 15), that file is
used instead: it is the same development at full resolution, and reading it
does not even open the RAW.

Orientation is applied here, explicitly, from what LibRaw reports for the RAW.
The embedded JPEG usually carries its own orientation tag and OpenCV would
honour it, but "usually" is not a property to build on: a preview without the
tag would come out lying on its side, and every score computed on it -- the
sharpest region, the comparison view -- would be computed on a rotated frame.

Two cached copies come out of this module, both named after the photo's content
like the proxies are: a small one for the dense grid of section 7.6 and the full
preview for the synchronised 1:1 comparison. Neither is a proxy -- no pipeline
ran -- and neither lives anywhere near the source folder.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import get_settings
from ..safety import guarded_open

__all__ = [
    "GRID_EDGE",
    "EmbeddedPreview",
    "PreviewPaths",
    "orient",
    "preview_paths_for",
    "read_embedded_preview",
    "write_preview_cache",
]

#: Long edge of the grid thumbnail. A card in the culling grid is about two
#: hundred CSS pixels wide; four hundred covers a 2x display, and anything more
#: is decoding work the browser does for pixels it throws away.
GRID_EDGE = 400

#: JPEG quality of the cached copies. They are shown, never exported, and the
#: embedded preview they come from is itself a camera JPEG: re-encoding it any
#: finer preserves the camera's artefacts, not the photograph.
_CACHE_QUALITY = 88

#: LibRaw's ``sizes.flip`` to the rotation that makes the frame upright. LibRaw
#: speaks its own dialect here, not EXIF: 3 is 180 degrees, 5 is a quarter turn
#: counter-clockwise, 6 a quarter turn clockwise.
_FLIP_TO_ROTATION = {
    3: cv2.ROTATE_180,
    5: cv2.ROTATE_90_COUNTERCLOCKWISE,
    6: cv2.ROTATE_90_CLOCKWISE,
}


@dataclass(frozen=True)
class EmbeddedPreview:
    """An upright, camera-developed view of one shot.

    ``image`` is 8-bit BGR in the camera's output space (sRGB for every Sony
    body this program supports), in OpenCV's channel order.
    """

    image: np.ndarray
    #: ``embedded`` (the ARW's own preview), ``sidecar`` (the JPEG next to it)
    #: or ``decoded`` (neither existed and the RAW was developed at half size).
    source: str
    #: LibRaw's orientation code for the RAW, already applied to ``image``;
    #: kept for whatever else has to be turned the same way, like a focus point.
    flip: int = 0

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


def orient(image: np.ndarray, flip: int) -> np.ndarray:
    """Rotate a frame decoded as stored into the way it was held."""
    rotation = _FLIP_TO_ROTATION.get(int(flip or 0))
    return image if rotation is None else cv2.rotate(image, rotation)


def _decode_jpeg(data: bytes | np.ndarray, *, reduce: int = 1) -> np.ndarray:
    buffer = np.frombuffer(data, dtype=np.uint8) if isinstance(data, bytes) else data
    # Orientation is ignored on purpose: it is applied from the RAW's own
    # metadata, which is the one source every path through here shares.
    flags = cv2.IMREAD_IGNORE_ORIENTATION | {
        1: cv2.IMREAD_COLOR,
        2: cv2.IMREAD_REDUCED_COLOR_2,
        4: cv2.IMREAD_REDUCED_COLOR_4,
    }[reduce]
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise ValueError("anteprima JPEG illeggibile")
    return image


def _sidecar_reduction(path: Path) -> int:
    """How much to shrink a full-resolution camera JPEG while decoding it.

    The DCT can be decoded at a half or a quarter of the size for a fraction of
    the cost. A 6000 px sidecar decoded at half size is 3000 px -- still above
    what the analysis uses -- and costs about as much as the 1616 px preview.
    """
    size = path.stat().st_size
    return 2 if size > 1_500_000 else 1


def read_embedded_preview(
    raw_path: str | Path, *, sidecar: str | Path | None = None
) -> EmbeddedPreview:
    """The camera's own development of a shot, upright.

    Args:
        raw_path: the ARW. Opened read-only; not decoded unless it carries no
            usable preview at all.
        sidecar: a JPEG the camera wrote next to it, preferred when readable.

    Returns:
        An :class:`EmbeddedPreview`.

    Raises:
        FileNotFoundError: the RAW is not there.
        ValueError: LibRaw cannot read the file at all -- a corrupt or
            truncated RAW, which the caller records as a failed photo.
    """
    import rawpy

    source = Path(raw_path)
    if not source.is_file():
        raise FileNotFoundError(f"{source.name} non è più nella cartella sorgente")

    try:
        raw = rawpy.imread(str(source))
    except (rawpy.LibRawError, OSError) as exc:
        raise ValueError(f"file RAW illeggibile ({type(exc).__name__})") from exc

    with raw:
        flip = int(raw.sizes.flip)

        if sidecar is not None:
            jpeg = Path(sidecar)
            try:
                with open(jpeg, "rb") as handle:
                    data = handle.read()
                image = _decode_jpeg(data, reduce=_sidecar_reduction(jpeg))
                return EmbeddedPreview(orient(image, flip), "sidecar", flip)
            except (OSError, ValueError):
                pass  # an unreadable sidecar is not a reason to fail: fall through

        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            thumb = None
        except rawpy.LibRawError as exc:
            raise ValueError(f"anteprima incorporata illeggibile ({type(exc).__name__})") from exc

        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            return EmbeddedPreview(orient(_decode_jpeg(thumb.data), flip), "embedded", flip)
        if thumb is not None and thumb.format == rawpy.ThumbFormat.BITMAP:
            bitmap = np.ascontiguousarray(thumb.data[..., ::-1])
            return EmbeddedPreview(orient(bitmap, flip), "embedded", flip)

        # No preview at all: rare on a Sony, but a file written by another tool
        # may have lost it. A half-size camera-style development is the honest
        # fallback -- slower, and said so by ``source``.
        try:
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
        except rawpy.LibRawError as exc:
            raise ValueError(f"file RAW illeggibile ({type(exc).__name__})") from exc
    # postprocess already applies the flip.
    return EmbeddedPreview(np.ascontiguousarray(rgb[..., ::-1]), "decoded", flip)


@dataclass(frozen=True)
class PreviewPaths:
    grid: Path
    full: Path

    def exist(self) -> bool:
        return self.grid.is_file() and self.full.is_file()


def preview_paths_for(identity: str) -> PreviewPaths:
    """Where the cached copies of a photo's preview live.

    Named after the content hash, like the proxies (``raw/proxy.py``): a renamed
    file keeps its preview, two identical files share one.
    """
    digest = hashlib.sha256(identity.encode()).hexdigest()
    folder = get_settings().cache_dir / "previews" / digest[:2]
    return PreviewPaths(grid=folder / f"{digest}-grid.jpg", full=folder / f"{digest}-full.jpg")


def _write_jpeg(image: np.ndarray, destination: Path) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, _CACHE_QUALITY])
    if not ok:
        raise ValueError("codifica JPEG dell'anteprima non riuscita")
    # Written next to the destination and renamed over it: a worker killed
    # halfway leaves a stray temporary file, never a truncated JPEG with the
    # right name that the grid would then show as a grey band for ever.
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with guarded_open(temporary, "wb") as handle:
        handle.write(encoded.tobytes())
    os.replace(temporary, destination)


def write_preview_cache(preview: EmbeddedPreview, identity: str) -> PreviewPaths:
    """Store the grid thumbnail and the full preview. Overwrites both."""
    from ..pipeline.filters import resize_long_edge

    paths = preview_paths_for(identity)
    _write_jpeg(preview.image, paths.full)
    small = resize_long_edge(preview.image, GRID_EDGE)
    _write_jpeg(np.ascontiguousarray(small), paths.grid)
    return paths
