# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The generated ICC profiles say what the pipeline actually does.

A profile that parses is not a profile that is correct. The check that matters
is the round trip: pushing our own sRGB output through our own sRGB profile and
into littleCMS's sRGB must be the identity, because the two describe the same
space. Getting the tone response curve backwards -- writing the encoding
function where ICC expects the decoding one -- produces a profile that opens
cleanly in every tool and renders everything far too bright, and this is the
test that catches it.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from ape.export.icc import build_profile
from ape.pipeline.colorspace import OutputSpace

pytest.importorskip("PIL.ImageCms")


def _open(space: OutputSpace):
    from PIL import ImageCms

    return ImageCms.getOpenProfile(io.BytesIO(build_profile(space)))


def test_every_profile_parses_and_is_named():
    from PIL import ImageCms

    for space in OutputSpace:
        description = ImageCms.getProfileDescription(_open(space)).strip()
        assert description.startswith("autoPhotoEdit")


def test_our_srgb_profile_is_the_identity_against_a_reference_srgb():
    from PIL import Image, ImageCms

    reference = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransform(
        _open(OutputSpace.SRGB), reference, "RGB", "RGB", renderingIntent=1
    )
    for colour in ((0, 0, 0), (18, 18, 18), (60, 150, 90), (200, 30, 120), (255, 255, 255)):
        image = Image.new("RGB", (1, 1), colour)
        result = ImageCms.applyTransform(image, transform).getpixel((0, 0))
        assert all(abs(a - b) <= 1 for a, b in zip(result, colour, strict=True)), (
            f"{colour} -> {result}"
        )


def test_wider_spaces_are_actually_wider():
    """A mid green must come out more saturated when tagged with a wider gamut."""
    from PIL import Image, ImageCms

    reference = ImageCms.createProfile("sRGB")
    image = Image.new("RGB", (1, 1), (60, 150, 90))
    reds = {}
    for space in (OutputSpace.SRGB, OutputSpace.DISPLAY_P3, OutputSpace.REC2020):
        transform = ImageCms.buildTransform(
            _open(space), reference, "RGB", "RGB", renderingIntent=1
        )
        reds[space] = ImageCms.applyTransform(image, transform).getpixel((0, 0))[0]
    assert reds[OutputSpace.SRGB] > reds[OutputSpace.DISPLAY_P3] >= reds[OutputSpace.REC2020]


def test_profiles_are_embedded_in_what_we_write():
    from PIL import Image

    from ape.export.image import ExportFormat, encode_image

    image = np.linspace(0.0, 1.0, 64 * 48 * 3, dtype=np.float32).reshape(48, 64, 3)
    for fmt in ExportFormat:
        payload = encode_image(image, fmt, output_space=OutputSpace.DISPLAY_P3)
        opened = Image.open(io.BytesIO(payload))
        assert opened.info.get("icc_profile"), f"{fmt} senza profilo ICC"
        assert opened.size == (64, 48)


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_our_tiffs_read_back_unchanged(bit_depth):
    """The TIFF writer must produce files a third-party reader agrees with.

    OpenCV rather than Pillow, because Pillow has no 16-bit RGB mode at all --
    which is why we write the TIFFs ourselves -- and silently hands back an
    8-bit downconversion instead of refusing.
    """
    import cv2

    from ape.export.image import ExportFormat, encode_image, quantise

    fmt = ExportFormat.TIFF16 if bit_depth == 16 else ExportFormat.TIFF8
    rng = np.random.default_rng(11)
    image = rng.random((37, 53, 3), dtype=np.float32)

    payload = encode_image(image, fmt)
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None, "OpenCV non ha riconosciuto il TIFF generato"
    assert decoded.dtype == (np.uint16 if bit_depth == 16 else np.uint8)
    # OpenCV hands back BGR.
    assert np.array_equal(decoded[..., ::-1], quantise(image, bit_depth))


def test_odd_sized_images_do_not_break_the_strip_layout():
    """Sizes that do not divide the strip height are where a TIFF writer breaks."""
    import cv2

    from ape.export.image import ExportFormat, encode_image, quantise

    rng = np.random.default_rng(12)
    for height, width in ((1, 1), (1, 997), (997, 1), (1013, 13)):
        image = rng.random((height, width, 3), dtype=np.float32)
        payload = encode_image(image, ExportFormat.TIFF16)
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
        assert decoded is not None, f"{height}x{width} illeggibile"
        assert np.array_equal(decoded[..., ::-1], quantise(image, 16)), f"{height}x{width}"
