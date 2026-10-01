# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``EditParams`` as a Lightroom / Camera Raw sidecar (``crs:`` namespace).

Lightroom reads ``DSC06312.xmp`` next to ``DSC06312.ARW`` at import (or with
"Read Metadata from File"). What it gets is a *starting point in Lightroom's own
controls* that develops close to our render -- not the same pixels, because the
two programs have different tone curves and colour models, and Camera Raw's are
not published.

**Translated exactly** (geometry and white balance are physics, not taste):

* straighten and crop -- ``crs:CropAngle`` and the corners of the crop in sensor
  orientation (``crop_map.py``; convention measured on the user's own edits);
* lens profile on or off;
* white balance. Our temperature and Adobe's are two estimates of the same
  illuminant from the same sensor data: on the user's 64 photos left "As
  Shot", Adobe's as-shot temperature is ``1.040 x ours - 16.1`` in mired (RMS
  error 1.9 mired, i.e. about 25 K at 3500 K), and Adobe's tint is
  ``0.906 x ours - 0.0029 x mired + 0.5`` (RMS 1.2) -- the same sign since
  parameters version 2 (``raw/whitepoint.py``). An "as shot" edit is written
  as ``As Shot``, so Camera Raw uses its own estimate and nothing is lost to
  the fit;
* exposure, with the offset measured on the same 65 pairs: our EV sits 0.18
  stops above the Exposure2012 the user set for the same look (standard
  deviation 0.48, which is the look-for-look ambiguity, not a bias).

**Translated by name**, at the scale of the two controls' ranges (our ±1 is
Adobe's ±100): shadows/highlights (local and global summed), whites/blacks,
vibrance, saturation, clarity, the eight HSL bands (same bands, same hue
direction), split toning, the parametric and point tone curves, sharpening and
noise reduction. Our sigmoid's contrast becomes ``Contrast2012`` on a log
scale around our default of 1.2. These are the controls whose meaning differs
between the programs; the user's own pairs confirm there is no one-to-one
mapping to learn (their correlations with our values stay under 0.5).

**Dropped**: the sigmoid's black and white points, toe, shoulder and hue
preservation, which Camera Raw has no control for, and the masks (phase 9).
"""

from __future__ import annotations

import math

from ..pipeline.params import HSL_BANDS, EditParams
from .crop_map import sensor_corners
from .sidecar import SidecarContext
from .xmp_packet import Packet, format_number

__all__ = ["build", "adobe_temperature_tint", "adobe_exposure"]

_NS = {
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "tiff": "http://ns.adobe.com/tiff/1.0/",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
}

#: Camera Raw version the packet claims, and the 2012 process (PV5): the one
#: every Lightroom since 4 and Camera Raw since 7 reads with these field names.
_CRS_VERSION = "15.4"
_PROCESS_VERSION = "11.0"

#: Fitted on the user's 64 "As Shot" photos (module docstring).
_MIRED_SLOPE, _MIRED_OFFSET = 1.040, -16.1
_TINT_SLOPE, _TINT_PER_MIRED, _TINT_OFFSET = 0.906, -0.0029, 0.5
_EXPOSURE_OFFSET = -0.18

#: Our default sigmoid contrast, which is Adobe's 0.
_NEUTRAL_CONTRAST = 1.2

_BAND_NAMES = {band: band.capitalize() for band in HSL_BANDS}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def adobe_temperature_tint(temperature_k: float, tint: float) -> tuple[int, int]:
    """Our (K, tint) as Camera Raw's, through the fit of the module docstring."""
    ours_mired = 1e6 / max(temperature_k, 1.0)
    adobe_mired = _MIRED_SLOPE * ours_mired + _MIRED_OFFSET
    kelvin = _clamp(1e6 / max(adobe_mired, 1.0), 2000.0, 50000.0)
    adobe_tint = _TINT_SLOPE * tint + _TINT_PER_MIRED * ours_mired + _TINT_OFFSET
    return int(round(kelvin)), int(round(_clamp(adobe_tint, -150.0, 150.0)))


def adobe_exposure(ev: float) -> float:
    return _clamp(ev + _EXPOSURE_OFFSET, -5.0, 5.0)


def _hundred(value: float) -> str:
    return format_number(_clamp(value * 100.0, -100.0, 100.0), sign=True)


def _curve_points(params: EditParams) -> list[str]:
    points = params.tone_curve.points or [(0.0, 0.0), (1.0, 1.0)]
    return [f"{round(x * 255)}, {round(y * 255)}" for x, y in points]


def _tone(packet: Packet, params: EditParams) -> None:
    tone, shaping, local = params.tone, params.tone_shaping, params.local_contrast
    contrast = 100.0 * math.log2(tone.contrast / _NEUTRAL_CONTRAST)
    packet.update(
        {
            "crs:Exposure2012": format_number(adobe_exposure(params.exposure.ev), 2, sign=True),
            "crs:Contrast2012": format_number(_clamp(contrast, -100, 100), sign=True),
            "crs:Highlights2012": _hundred(shaping.highlights + local.highlights),
            "crs:Shadows2012": _hundred(shaping.shadows + local.shadows),
            "crs:Whites2012": _hundred(shaping.whites),
            "crs:Blacks2012": _hundred(shaping.blacks),
            "crs:Clarity2012": _hundred(local.clarity),
            "crs:Texture": "0",
            "crs:Dehaze": "0",
            "crs:ParametricShadows": _hundred(params.tone_curve.shadows),
            "crs:ParametricDarks": _hundred(params.tone_curve.darks),
            "crs:ParametricLights": _hundred(params.tone_curve.lights),
            "crs:ParametricHighlights": _hundred(params.tone_curve.highlights),
            "crs:ParametricShadowSplit": "25",
            "crs:ParametricMidtoneSplit": "50",
            "crs:ParametricHighlightSplit": "75",
            "crs:ToneCurveName2012": "Custom" if params.tone_curve.points else "Linear",
        }
    )
    packet.seq("crs:ToneCurvePV2012", _curve_points(params))


def _colour(packet: Packet, params: EditParams) -> None:
    colour = params.color
    packet.set("crs:Vibrance", _hundred(colour.vibrance))
    packet.set("crs:Saturation", _hundred(colour.saturation))
    for band, name in _BAND_NAMES.items():
        values = colour.band(band)
        packet.set(f"crs:HueAdjustment{name}", _hundred(values.hue))
        packet.set(f"crs:SaturationAdjustment{name}", _hundred(values.saturation))
        packet.set(f"crs:LuminanceAdjustment{name}", _hundred(values.luminance))
    split = colour.split_toning
    packet.update(
        {
            "crs:SplitToningShadowHue": format_number(split.shadow_hue % 360.0),
            "crs:SplitToningShadowSaturation": format_number(split.shadow_saturation * 100),
            "crs:SplitToningHighlightHue": format_number(split.highlight_hue % 360.0),
            "crs:SplitToningHighlightSaturation": format_number(
                split.highlight_saturation * 100
            ),
            "crs:SplitToningBalance": _hundred(split.balance),
        }
    )


def _detail(packet: Packet, params: EditParams, ctx: SidecarContext) -> None:
    sharpen, noise = params.sharpen, params.noise
    long_edge = max(ctx.width or 6000, ctx.height or 4000)
    # Our amount runs to 3, Camera Raw's to 150; our radius is a fraction of
    # the long edge, theirs is pixels of the full-resolution frame.
    packet.update(
        {
            "crs:Sharpness": format_number(_clamp(sharpen.amount * 50.0, 0, 150)),
            "crs:SharpenRadius": format_number(
                _clamp(sharpen.radius * long_edge, 0.5, 3.0), 1, sign=True
            ),
            "crs:SharpenDetail": "25",
            "crs:SharpenEdgeMasking": format_number(_clamp(sharpen.threshold * 500, 0, 100)),
            "crs:LuminanceSmoothing": format_number(noise.luminance * 100),
            "crs:ColorNoiseReduction": format_number(noise.chrominance * 100),
        }
    )


def _geometry(packet: Packet, params: EditParams, ctx: SidecarContext) -> None:
    geometry = params.geometry
    packet.set("crs:LensProfileEnable", "1" if geometry.lens_correction else "0")
    packet.set("crs:AutoLateralCA", "1" if geometry.lens_correction else "0")
    if geometry.crop is None and abs(geometry.rotation_deg) < 1e-3:
        packet.set("crs:HasCrop", "False")
        return
    width, height = ctx.width or 6000, ctx.height or 4000
    (left, top), (right, bottom) = sensor_corners(geometry, width, height, ctx.orientation)
    packet.update(
        {
            "crs:HasCrop": "True",
            "crs:CropTop": format_number(top, 6),
            "crs:CropLeft": format_number(left, 6),
            "crs:CropBottom": format_number(bottom, 6),
            "crs:CropRight": format_number(right, 6),
            "crs:CropAngle": format_number(geometry.rotation_deg, 2),
            "crs:CropConstrainToWarp": "0",
        }
    )


def build(params: EditParams, ctx: SidecarContext) -> bytes:
    """The complete ``.xmp`` for the RAW of ``ctx``, as UTF-8 bytes."""
    packet = Packet(namespaces=dict(_NS))
    if ctx.software:
        packet.set("xmp:CreatorTool", ctx.software)
    if ctx.camera_maker:
        packet.set("tiff:Make", ctx.camera_maker)
    if ctx.camera_model:
        packet.set("tiff:Model", ctx.camera_model)
    packet.update(
        {
            "crs:Version": _CRS_VERSION,
            "crs:ProcessVersion": _PROCESS_VERSION,
            "crs:HasSettings": "True",
            "crs:RawFileName": ctx.raw_filename,
            "crs:CameraProfile": "Adobe Standard",
        }
    )
    wb = params.white_balance
    if wb.mode == "as_shot":
        packet.set("crs:WhiteBalance", "As Shot")
    else:
        kelvin, tint = adobe_temperature_tint(wb.temperature_k, wb.tint)
        packet.set("crs:WhiteBalance", "Custom")
        packet.set("crs:Temperature", str(kelvin))
        packet.set("crs:Tint", format_number(tint, sign=True))
    _tone(packet, params)
    _colour(packet, params)
    _detail(packet, params, ctx)
    _geometry(packet, params, ctx)
    return packet.to_bytes(ctx.software or "autoPhotoEdit")
