# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A small baseline TIFF writer, for 8- and 16-bit RGB.

Pillow has no 16-bit RGB image mode at all, so it cannot write the format a
serious photographic export needs. The alternatives were adding a TIFF library
to the dependency list of docs/SPEC.md section 4 or writing the ~100 lines of
baseline TIFF that this actually takes; the second keeps the dependency list
honest and, more usefully, keeps the tag layout under our control -- section
16.3 has opinions about Orientation and Software that are easier to satisfy from
here than through somebody else's abstraction.

What this writes is deliberately plain: little-endian, single IFD, strip-based,
Adobe Deflate, contiguous planes. Every reader understands it. EXIF and XMP are
added afterwards by ``raw/metadata.py`` through exiv2 (phase 8), which is the
only component allowed to know about exiv2 at all (section 24).
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

__all__ = ["write_tiff"]

_BYTE, _ASCII, _SHORT, _LONG, _RATIONAL = 1, 2, 3, 4, 5
_UNDEFINED = 7

# Tag numbers, in the order the IFD must list them (ascending, as required).
_IMAGE_WIDTH = 256
_IMAGE_LENGTH = 257
_BITS_PER_SAMPLE = 258
_COMPRESSION = 259
_PHOTOMETRIC = 262
_IMAGE_DESCRIPTION = 270
_STRIP_OFFSETS = 273
_ORIENTATION = 274
_SAMPLES_PER_PIXEL = 277
_ROWS_PER_STRIP = 278
_STRIP_BYTE_COUNTS = 279
_X_RESOLUTION = 282
_Y_RESOLUTION = 283
_PLANAR_CONFIG = 284
_RESOLUTION_UNIT = 296
_SOFTWARE = 305
_SAMPLE_FORMAT = 339
_ICC_PROFILE = 34675

_COMPRESSION_DEFLATE = 8
_PHOTOMETRIC_RGB = 2

#: One strip per ~256 KB of pixel data: small enough that a reader never has to
#: hold the whole image to decode a band, large enough that deflate still works.
_STRIP_TARGET_BYTES = 256 * 1024


def _entry(tag: int, field_type: int, count: int, payload: bytes) -> tuple[int, int, int, bytes]:
    return (tag, field_type, count, payload)


def _short(*values: int) -> bytes:
    return b"".join(struct.pack("<H", v) for v in values)


def _long(*values: int) -> bytes:
    return b"".join(struct.pack("<I", v) for v in values)


def _rational(numerator: int, denominator: int) -> bytes:
    return struct.pack("<II", numerator, denominator)


def _ascii(text: str) -> bytes:
    return text.encode("ascii", errors="replace") + b"\0"


def write_tiff(
    image: np.ndarray,
    *,
    icc_profile: bytes | None = None,
    software: str | None = None,
    description: str | None = None,
    dpi: int = 300,
) -> bytes:
    """Serialise an ``(H, W, 3)`` uint8 or uint16 RGB array as a TIFF file.

    Args:
        image: the pixels. C-contiguous is not required but is faster.
        icc_profile: profile bytes to embed, or ``None`` for an untagged file.
        software: value of the Software tag (section 16.3).
        description: value of ImageDescription.
        dpi: resolution recorded in the file. Metadata only; nothing is resampled.

    Returns:
        The complete file, ready to be written through ``safety.guarded_open``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB array, got {image.shape}")
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"expected uint8 or uint16, got {image.dtype}")

    height, width = int(image.shape[0]), int(image.shape[1])
    bits = 8 if image.dtype == np.uint8 else 16
    bytes_per_row = width * 3 * (bits // 8)
    rows_per_strip = max(1, min(height, _STRIP_TARGET_BYTES // max(bytes_per_row, 1)))
    strip_count = (height + rows_per_strip - 1) // rows_per_strip

    # Little-endian on disk regardless of the host, so the bytes are the same
    # everywhere -- which is part of the determinism guarantee.
    pixels = np.ascontiguousarray(image, dtype=f"<u{bits // 8}")
    strips = [
        zlib.compress(pixels[start : start + rows_per_strip].tobytes(), 6)
        for start in range(0, height, rows_per_strip)
    ]

    entries: list[tuple[int, int, int, bytes]] = [
        _entry(_IMAGE_WIDTH, _LONG, 1, _long(width)),
        _entry(_IMAGE_LENGTH, _LONG, 1, _long(height)),
        _entry(_BITS_PER_SAMPLE, _SHORT, 3, _short(bits, bits, bits)),
        _entry(_COMPRESSION, _SHORT, 1, _short(_COMPRESSION_DEFLATE)),
        _entry(_PHOTOMETRIC, _SHORT, 1, _short(_PHOTOMETRIC_RGB)),
    ]
    if description:
        entries.append(_entry(_IMAGE_DESCRIPTION, _ASCII, 0, _ascii(description)))
    # Placeholder; the real offsets are only known once the layout is fixed.
    entries.append(_entry(_STRIP_OFFSETS, _LONG, strip_count, _long(*([0] * strip_count))))
    entries += [
        # The renderer has already baked the rotation into the pixels, so the
        # file must claim "no rotation" or half the viewers will apply it twice.
        _entry(_ORIENTATION, _SHORT, 1, _short(1)),
        _entry(_SAMPLES_PER_PIXEL, _SHORT, 1, _short(3)),
        _entry(_ROWS_PER_STRIP, _LONG, 1, _long(rows_per_strip)),
        _entry(_STRIP_BYTE_COUNTS, _LONG, strip_count, _long(*(len(s) for s in strips))),
        _entry(_X_RESOLUTION, _RATIONAL, 1, _rational(dpi, 1)),
        _entry(_Y_RESOLUTION, _RATIONAL, 1, _rational(dpi, 1)),
        _entry(_PLANAR_CONFIG, _SHORT, 1, _short(1)),
        _entry(_RESOLUTION_UNIT, _SHORT, 1, _short(2)),  # inches
    ]
    if software:
        entries.append(_entry(_SOFTWARE, _ASCII, 0, _ascii(software)))
    entries.append(_entry(_SAMPLE_FORMAT, _SHORT, 3, _short(1, 1, 1)))  # unsigned integer
    if icc_profile:
        entries.append(_entry(_ICC_PROFILE, _UNDEFINED, len(icc_profile), icc_profile))

    entries.sort(key=lambda item: item[0])

    ifd_size = 2 + 12 * len(entries) + 4
    external_base = 8 + ifd_size

    # First pass: place every value that does not fit inline, so we know where
    # the pixel data starts and can fill in the strip offsets.
    external = bytearray()
    placement: dict[int, int] = {}
    for tag, _field_type, _count, payload in entries:
        if len(payload) <= 4:
            continue
        placement[tag] = external_base + len(external)
        external += payload
        if len(external) % 2:
            external += b"\0"

    data_base = external_base + len(external)
    offsets = []
    position = data_base
    for strip in strips:
        offsets.append(position)
        position += len(strip)

    ifd = bytearray(struct.pack("<H", len(entries)))
    for tag, field_type, count, payload in entries:
        if tag == _STRIP_OFFSETS:
            payload = _long(*offsets)
        real_count = count or len(payload)
        if len(payload) <= 4:
            ifd += struct.pack("<HHI", tag, field_type, real_count)
            ifd += payload.ljust(4, b"\0")
        else:
            ifd += struct.pack("<HHII", tag, field_type, real_count, placement[tag])
    ifd += struct.pack("<I", 0)  # no further IFDs

    header = struct.pack("<2sHI", b"II", 42, 8)
    return header + bytes(ifd) + bytes(external) + b"".join(strips)
