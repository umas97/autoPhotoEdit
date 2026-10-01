# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal ICC v2 matrix/shaper profile writer.

An exported file whose colour space is not declared is a file whose colours are
a guess, so every export carries an embedded profile. Rather than ship binary
profile blobs -- which would have to be licensed, checksummed and kept in sync
with the primaries the pipeline actually uses -- the profiles are generated from
those same primaries, here.

ICC v2 rather than v4 for one reason: v2 matrix/shaper profiles are understood
by everything, down to browsers and phone galleries, and the only v4 feature
this would want is the parametric curve type, which a 1024-entry sampled curve
replaces exactly.

The profile connection space of a v2 display profile is XYZ under D50, so the
colorants are Bradford-adapted from the D65 the working space uses, and the
adaptation itself is recorded in the ``chad`` tag as the specification requires.
"""

from __future__ import annotations

import struct

import numpy as np

from ..pipeline.colorspace import RGB_TO_XYZ, OutputSpace, decode_transfer

__all__ = ["build_profile", "profile_for"]

#: ICC's D50, the profile connection space white point.
_D50 = np.array([0.9642, 1.0, 0.8249], dtype=np.float64)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

#: Bradford cone response, the transform ICC mandates for the ``chad`` tag.
_BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)

_CURVE_POINTS = 1024

_DESCRIPTIONS = {
    OutputSpace.SRGB: "autoPhotoEdit sRGB",
    OutputSpace.DISPLAY_P3: "autoPhotoEdit Display P3",
    OutputSpace.ADOBE_RGB: "autoPhotoEdit Adobe RGB (1998) compatible",
    OutputSpace.REC2020: "autoPhotoEdit ITU-R BT.2020",
}


def _s15f16(value: float) -> bytes:
    return struct.pack(">i", int(round(value * 65536.0)))


def _pad4(data: bytes) -> bytes:
    return data + b"\0" * (-len(data) % 4)


def _bradford_d65_to_d50() -> np.ndarray:
    source = _BRADFORD @ _D65
    destination = _BRADFORD @ _D50
    return np.linalg.inv(_BRADFORD) @ np.diag(destination / source) @ _BRADFORD


def _xyz_tag(xyz: np.ndarray) -> bytes:
    return b"XYZ " + b"\0" * 4 + b"".join(_s15f16(float(v)) for v in xyz)


def _curve_tag(space: OutputSpace) -> bytes:
    """Sampled tone response curve, as uint16.

    ICC defines a TRC as *device value -> linear*, i.e. the EOTF, not the
    encoding function. Getting this backwards produces a profile that parses
    cleanly and renders everything far too bright, which is exactly the kind of
    bug that survives a smoke test, so there is a round-trip assertion for it in
    ``tests/test_icc_profiles.py``.
    """
    samples = np.linspace(0.0, 1.0, _CURVE_POINTS, dtype=np.float32)
    linear = decode_transfer(samples, space)
    values = np.clip(np.rint(linear * 65535.0), 0, 65535).astype(">u2")
    return b"curv" + b"\0" * 4 + struct.pack(">I", _CURVE_POINTS) + values.tobytes()


def _text_description_tag(text: str) -> bytes:
    """ICC v2 ``textDescriptionType``. The Unicode and ScriptCode halves stay empty."""
    ascii_bytes = text.encode("ascii", errors="replace") + b"\0"
    return (
        b"desc"
        + b"\0" * 4
        + struct.pack(">I", len(ascii_bytes))
        + ascii_bytes
        + struct.pack(">I", 0)  # Unicode language code
        + struct.pack(">I", 0)  # Unicode character count
        + struct.pack(">H", 0)  # ScriptCode code
        + struct.pack(">B", 0)  # ScriptCode count
        + b"\0" * 67  # ScriptCode description, fixed length
    )


def _text_tag(text: str) -> bytes:
    return b"text" + b"\0" * 4 + text.encode("ascii", errors="replace") + b"\0"


def _chad_tag(matrix: np.ndarray) -> bytes:
    return b"sf32" + b"\0" * 4 + b"".join(_s15f16(float(v)) for v in matrix.reshape(-1))


#: Fixed creation timestamp. A real clock here would make every exported file
#: differ from the last one byte for byte, which would break the determinism
#: guarantee of docs/SPEC.md section 13.4 for no benefit: nothing reads this field.
_CREATED = (2024, 1, 1, 0, 0, 0)


def _header(size: int) -> bytes:
    header = bytearray(128)
    struct.pack_into(">I", header, 0, size)
    header[4:8] = b"\0" * 4  # preferred CMM: none
    struct.pack_into(">I", header, 8, 0x02100000)  # ICC 2.1
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[20:24] = b"XYZ "
    struct.pack_into(">HHHHHH", header, 24, *_CREATED)
    header[36:40] = b"acsp"
    header[40:44] = b"\0" * 4  # platform: none
    struct.pack_into(">I", header, 44, 0)  # flags: not embedded, use anywhere
    header[64:68] = struct.pack(">I", 0)  # rendering intent: perceptual
    header[68:80] = b"".join(_s15f16(float(v)) for v in _D50)
    return bytes(header)


def build_profile(space: OutputSpace, description: str | None = None) -> bytes:
    """Serialise an ICC v2 matrix/shaper profile for one of our output spaces."""
    if space not in RGB_TO_XYZ:
        raise ValueError(f"no primaries known for {space!r}")

    adaptation = _bradford_d65_to_d50()
    colorants = adaptation @ RGB_TO_XYZ[space]
    curve = _curve_tag(space)

    tags: list[tuple[bytes, bytes]] = [
        (b"desc", _text_description_tag(description or _DESCRIPTIONS[space])),
        (b"wtpt", _xyz_tag(_D50)),
        (b"rXYZ", _xyz_tag(colorants[:, 0])),
        (b"gXYZ", _xyz_tag(colorants[:, 1])),
        (b"bXYZ", _xyz_tag(colorants[:, 2])),
        (b"rTRC", curve),
        (b"gTRC", curve),
        (b"bTRC", curve),
        (b"chad", _chad_tag(adaptation)),
        (b"cprt", _text_tag("Generated by autoPhotoEdit. No rights reserved on the profile.")),
    ]

    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    entries = bytearray()
    body = bytearray()
    # The three TRC tags are byte-identical; pointing them at one block is both
    # legal and what every real-world profile does.
    placed: dict[bytes, tuple[int, int]] = {}
    for signature, data in tags:
        key = bytes(data)
        if key not in placed:
            placed[key] = (offset + len(body), len(data))
            body += _pad4(data)
        position, length = placed[key]
        entries += signature + struct.pack(">II", position, length)

    size = offset + len(body)
    return _header(size) + struct.pack(">I", len(tags)) + bytes(entries) + bytes(body)


_CACHE: dict[OutputSpace, bytes] = {}


def profile_for(space: OutputSpace) -> bytes:
    """Cached profile bytes for an output space."""
    if space not in _CACHE:
        _CACHE[space] = build_profile(space)
    return _CACHE[space]
