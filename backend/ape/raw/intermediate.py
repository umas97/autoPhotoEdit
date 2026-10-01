# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The linear buffer of a merged photo, on disk (docs/SPEC.md section 25.1).

A merged photo has no RAW of its own. What stands in for one is this file: the
output of the merge, linear, scene-referred, Rec.2020, exactly what
``decode_linear`` returns for a single shot, plus the context the pipeline needs
to interpret it -- the camera colour of the reference frame, its exposure
anchor, its optics and its EXIF. ``decode.py`` and ``metadata.py`` read it
through the same functions they use for a RAW, so nothing downstream can tell
the difference, which is the architectural rule of section 25.1.

**Format.** A TIFF, as section 25.1 allows, with two images in it: the frame at
full resolution and a copy at the proxy's long edge, which is what the editor
opens -- a panorama of 80 MP would otherwise be read whole to show 2048 px.
Both are half-precision floats, RGB, uncompressed, one strip each:

* half precision keeps 11 significant bits over a range of 2^-14..65504: finer
  than any display step after the tone curve, and a panorama costs 480 MB
  instead of 960;
* uncompressed and one strip make the pixels a plain array at a known offset,
  so they are written band by band while a panorama is being composed and read
  back through a memory map, never all at once.

The context travels in ``ImageDescription`` as JSON, so the file is
self-describing: an intermediate is only ever a cache entry (section 20.3), and
whatever reads it needs nothing from the catalogue to use it.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SUFFIX",
    "IntermediateWriter",
    "context_for",
    "frame_map",
    "full_size",
    "is_intermediate",
    "read_context",
    "read_frame",
    "reduced_size",
    "write_intermediate",
]

#: What names an intermediate. The decoder dispatches on it.
SUFFIX = ".apemerge.tif"

_FORMAT = "autophotoedit-intermediate"
_VERSION = 1

#: Largest value half precision holds. A merge brighter than 2^15 times white
#: does not exist; the clip is there so that an outlier becomes a highlight,
#: not an infinity that poisons every average downstream.
_HALF_MAX = 65000.0

_SHORT, _ASCII, _LONG = 3, 2, 4
_TAGS_PER_IFD = 12


def is_intermediate(path: str | os.PathLike[str] | None) -> bool:
    return path is not None and str(path).endswith(SUFFIX)


def reduced_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Size of the reduced copy: the proxy's long edge, never an enlargement."""
    scale = min(1.0, long_edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def context_for(decoded: Any, metadata: Any, *, lens: bool = True, **extra: Any) -> dict:
    """What a reader needs to treat the merge as the reference frame's decode.

    Args:
        decoded: the reference frame's :class:`~ape.raw.decode.DecodedRaw`.
        metadata: its :class:`~ape.raw.metadata.PhotoMetadata`.
        lens: keep the optics, so the lens stage corrects the merge like the
            single shot it is aligned to. A panorama passes False: its frames
            were corrected before stitching, and its geometry is no lens's.
        extra: merge-specific notes (kind, members) kept for diagnostics.

    The shooting time is the first frame's (section 25.1), and the EXIF tags
    the export copies say so too: they are the reference's, and a panorama's
    reference is its middle frame.
    """
    camera = decoded.camera
    stored = asdict(metadata)
    # Said to whoever reads the optics (the analysis, the lens list): a profile
    # found for this lens would describe a correction already made.
    stored["optics_corrected"] = not lens
    shot_at = getattr(metadata, "shot_at", None)
    tags = dict(stored.get("raw_tags") or {})
    if shot_at is not None and tags:
        stamp = shot_at.strftime("%Y:%m:%d %H:%M:%S")
        if tags.get("Exif.Photo.DateTimeOriginal", stamp) != stamp:
            tags["Exif.Photo.DateTimeOriginal"] = tags["Exif.Photo.DateTimeDigitized"] = stamp
            # The fraction of a second is the reference's, not the first frame's.
            for key in ("Exif.Photo.SubSecTimeOriginal", "Exif.Photo.SubSecTimeDigitized"):
                tags.pop(key, None)
            stored["raw_tags"] = tags
    return _jsonable(
        {
            "camera": {
                "cam_from_xyz": camera.cam_from_xyz,
                "xyz_from_cam": camera.xyz_from_cam,
                "as_shot_multipliers": camera.as_shot_multipliers,
                "as_shot_temperature_k": camera.as_shot_temperature_k,
                "as_shot_tint": camera.as_shot_tint,
            },
            "baseline_exposure_ev": decoded.baseline_exposure_ev,
            "raw_metadata": decoded.raw_metadata,
            "lens": asdict(decoded.lens) if lens and decoded.lens is not None else None,
            "metadata": stored,
            "merge": extra,
        }
    )


def _ifd(
    width: int, height: int, data_offset: int, extra_offset: int, description: tuple[int, int],
    next_ifd: int, reduced: bool,
) -> tuple[bytes, bytes]:
    """One IFD and the values that do not fit in it (bits and sample format)."""
    extra = struct.pack("<3H3H", 16, 16, 16, 3, 3, 3)
    entries = [
        (254, _LONG, 1, struct.pack("<I", 1 if reduced else 0)),
        (256, _LONG, 1, struct.pack("<I", width)),
        (257, _LONG, 1, struct.pack("<I", height)),
        (258, _SHORT, 3, struct.pack("<I", extra_offset)),
        (259, _SHORT, 1, struct.pack("<HH", 1, 0)),
        (262, _SHORT, 1, struct.pack("<HH", 2, 0)),
        (270, _ASCII, description[1], struct.pack("<I", description[0])),
        (273, _LONG, 1, struct.pack("<I", data_offset)),
        (277, _SHORT, 1, struct.pack("<HH", 3, 0)),
        (278, _LONG, 1, struct.pack("<I", height)),
        (279, _LONG, 1, struct.pack("<I", width * height * 6)),
        (339, _SHORT, 3, struct.pack("<I", extra_offset + 6)),
    ]
    assert len(entries) == _TAGS_PER_IFD
    body = struct.pack("<H", len(entries))
    for tag, kind, count, payload in entries:
        body += struct.pack("<HHI", tag, kind, count) + payload
    body += struct.pack("<I", next_ifd)
    return body, extra


class IntermediateWriter:
    """Writes an intermediate band by band, then its reduced copy.

    The file is built under a temporary name and renamed into place by
    :meth:`close`, so a crash in the middle of a panorama leaves no truncated
    intermediate that a reader would take for a finished one.
    """

    def __init__(
        self, path: str | os.PathLike[str], width: int, height: int, context: dict,
        *, reduced_long_edge: int,
    ) -> None:
        from ..safety import assert_outside_source

        self.path = assert_outside_source(path)
        self.width, self.height = int(width), int(height)
        if self.width * self.height * 6 >= 2**32:
            raise ValueError("fusione troppo grande per un intermedio TIFF (oltre 4 GB)")
        self.reduced = reduced_size(self.width, self.height, reduced_long_edge)
        description = json.dumps(
            {"format": _FORMAT, "version": _VERSION, **context}, ensure_ascii=True
        ).encode("ascii") + b"\0"

        ifd_size = 2 + 12 * _TAGS_PER_IFD + 4
        first, second = 8, 8 + ifd_size
        extras = 8 + 2 * ifd_size
        text = extras + 24
        data = text + len(description)
        data += (-data) % 16
        self._full_offset = data
        self._reduced_offset = data + self.width * self.height * 6
        total = self._reduced_offset + self.reduced[0] * self.reduced[1] * 6

        head, extra_one = _ifd(
            self.width, self.height, self._full_offset, extras, (text, len(description)),
            second, reduced=False,
        )
        tail, extra_two = _ifd(
            *self.reduced, self._reduced_offset, extras + 12, (text, len(description)),
            0, reduced=True,
        )
        self._part = self.path.with_name(f".{self.path.name}.{os.getpid()}.part")
        self._part.parent.mkdir(parents=True, exist_ok=True)
        with open(self._part, "wb") as handle:
            handle.write(b"II*\0" + struct.pack("<I", first))
            handle.write(head + tail + extra_one + extra_two)
            handle.write(description)
            handle.truncate(total)
        self._map = np.memmap(
            self._part, dtype=np.float16, mode="r+", offset=self._full_offset,
            shape=(self.height, self.width, 3),
        )

    def write_rows(self, top: int, rows: np.ndarray) -> None:
        """Store rows ``top .. top + len(rows)`` of the full frame."""
        band = np.clip(rows, -_HALF_MAX, _HALF_MAX)
        self._map[top : top + band.shape[0]] = band.astype(np.float16)

    def close(self) -> Path:
        """Compute the reduced copy, flush, and move the file into place."""
        self._map.flush()
        reduced = _reduce(self._map, self.reduced)
        small = np.memmap(
            self._part, dtype=np.float16, mode="r+", offset=self._reduced_offset,
            shape=(self.reduced[1], self.reduced[0], 3),
        )
        small[:] = np.clip(reduced, -_HALF_MAX, _HALF_MAX).astype(np.float16)
        small.flush()
        del small
        del self._map
        os.replace(self._part, self.path)
        return self.path

    def abort(self) -> None:
        self._map = None  # type: ignore[assignment]
        self._part.unlink(missing_ok=True)


#: Rows converted to single precision at a time when reducing: 12 MB of a
#: 16 000 px panorama, so the reduction never holds more than one band of it.
_REDUCE_ROWS = 128


def _reduce(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Area-average ``frame`` to ``size``, in bands.

    Area averaging is separable: first every band is narrowed to the target
    width, which leaves a column of manageable size, then that is shortened to
    the target height. The result equals a one-step ``INTER_AREA`` up to
    rounding.
    """
    import cv2

    width, height = size
    if (frame.shape[1], frame.shape[0]) == size:
        return np.asarray(frame, dtype=np.float32)
    narrow = np.empty((frame.shape[0], width, 3), dtype=np.float32)
    for top in range(0, frame.shape[0], _REDUCE_ROWS):
        band = np.asarray(frame[top : top + _REDUCE_ROWS], dtype=np.float32)
        narrow[top : top + band.shape[0]] = cv2.resize(
            band, (width, band.shape[0]), interpolation=cv2.INTER_AREA
        )
    return cv2.resize(narrow, (width, height), interpolation=cv2.INTER_AREA)


def write_intermediate(
    path: str | os.PathLike[str], rgb: np.ndarray, context: dict, *, reduced_long_edge: int
) -> Path:
    """Write a whole frame at once (HDR and focus stacks, which hold one anyway)."""
    height, width = rgb.shape[:2]
    writer = IntermediateWriter(path, width, height, context, reduced_long_edge=reduced_long_edge)
    try:
        for top in range(0, height, _REDUCE_ROWS):
            writer.write_rows(top, rgb[top : top + _REDUCE_ROWS])
        return writer.close()
    except BaseException:
        writer.abort()
        raise


@dataclass(frozen=True)
class _Layout:
    context: dict
    images: tuple[tuple[int, int, int], ...]  # (width, height, offset) per IFD


def _read_layout(path: Path) -> _Layout:
    with open(path, "rb") as handle:
        head = handle.read(8)
        if head[:4] != b"II*\0":
            raise ValueError(f"non è un intermedio di autoPhotoEdit: {path.name}")
        offset = struct.unpack("<I", head[4:])[0]
        images = []
        description = None
        while offset:
            handle.seek(offset)
            count = struct.unpack("<H", handle.read(2))[0]
            tags: dict[int, tuple[int, int, bytes]] = {}
            for _ in range(count):
                tag, kind, number = struct.unpack("<HHI", handle.read(8))
                tags[tag] = (kind, number, handle.read(4))
            offset = struct.unpack("<I", handle.read(4))[0]
            width = struct.unpack("<I", tags[256][2])[0]
            height = struct.unpack("<I", tags[257][2])[0]
            data = struct.unpack("<I", tags[273][2])[0]
            images.append((width, height, data))
            if description is None and 270 in tags:
                _kind, length, where = tags[270]
                position = handle.tell()
                handle.seek(struct.unpack("<I", where)[0])
                description = handle.read(length).rstrip(b"\0")
                handle.seek(position)
    try:
        context = json.loads(description or b"")
    except ValueError as exc:
        raise ValueError(f"intermedio illeggibile: {path.name}") from exc
    if context.get("format") != _FORMAT or context.get("version") != _VERSION:
        raise ValueError(f"intermedio di un formato sconosciuto: {path.name}")
    return _Layout(context=context, images=tuple(images))


def read_context(path: str | os.PathLike[str]) -> dict:
    """The JSON context of an intermediate, without touching its pixels."""
    return _read_layout(Path(path)).context


def read_frame(path: str | os.PathLike[str], *, reduced: bool = False) -> tuple[np.ndarray, dict]:
    """The frame (or its reduced copy) as float32, and the context.

    Raises:
        FileNotFoundError: the intermediate is gone -- the cache quota or the
            user removed it, and the merge has to be run again.
        ValueError: the file is not an intermediate this build can read.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"intermedio della fusione non trovato: {source.name}")
    layout = _read_layout(source)
    width, height, offset = layout.images[1 if reduced and len(layout.images) > 1 else 0]
    pixels = np.memmap(source, dtype=np.float16, mode="r", offset=offset, shape=(height, width, 3))
    frame = np.empty((height, width, 3), dtype=np.float32)
    for top in range(0, height, _REDUCE_ROWS):
        frame[top : top + _REDUCE_ROWS] = pixels[top : top + _REDUCE_ROWS]
    del pixels
    return frame, layout.context


def full_size(path: str | os.PathLike[str]) -> tuple[int, int]:
    """``(width, height)`` of the full frame."""
    width, height, _ = _read_layout(Path(path)).images[0]
    return width, height


def frame_map(path: str | os.PathLike[str]) -> np.ndarray:
    """The full frame as a read-only half-precision memory map."""
    layout = _read_layout(Path(path))
    width, height, offset = layout.images[0]
    return np.memmap(path, dtype=np.float16, mode="r", offset=offset, shape=(height, width, 3))
