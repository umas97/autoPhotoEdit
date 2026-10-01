# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``EditParams`` as a darktable sidecar (``DSC06312.ARW.xmp``), for darktable 5.

darktable keeps an edit as a *history*: a list of module instances with their
parameters as C structs (``darktable_modules.py``). This file writes one entry
per module our parameters need, in darktable's scene-referred modules, and marks
the auto-presets as applied so darktable adds nothing of its own on top. Every
module this history does not name is at its default -- which for the modules a
raw needs (rawprepare, demosaic, highlights, colorin, colorout, orientation) is
"on", exactly as when darktable opens a file for the first time.

**The mapping, and how close it is.**

* *White balance*: the channel multipliers of our target illuminant, computed
  through the sensor's own matrix -- the same physics as our chromatic
  adaptation, applied where darktable applies it (``temperature``, before the
  demosaic). "As shot" is written as darktable's "as shot".
* *Exposure and tone*: our EV plus the decoder's exposure anchor, then
  ``sigmoid`` with contrast and skew *fitted* to our curve (``fit_sigmoid``:
  a least-squares match of darktable's own formula to ``tone.sigmoid_response``
  over the stops a photograph occupies), the fit's exposure offset folded into
  the exposure module. Our tone shaping and tone curve become one
  ``tonecurve`` in RGB mode, translated to darktable's linear pipe.
* *Colour*: vibrance and saturation in ``colorbalancergb``, the eight HSL bands
  in ``colorequal`` (the same eight hues, in the same order), split toning in
  colour balance's shadow and highlight wheels. Scales are by name, ±1 to the
  module's full range; the modules' colour models differ from ours.
* *Local*: our local shadows/highlights in ``shadhi`` at the same radius,
  clarity in ``bilat`` (local contrast), sharpening in ``sharpen``.
* *Geometry*: the lensfun profile by name (``lens``), the rotation in
  ``ashift`` cropped to the original format -- which is our straightened frame
  -- and our crop, unchanged, in ``crop``.

**Dropped**: denoise (darktable's needs a noise profile of the body and ISO),
our highlight reconstruction (darktable's own module stays at its default),
the masks (phase 9).

A darktable render of this sidecar is ΔE2000 2-4 from ours on average over
the user's real looks (``tests/bench/bench_sidecar.py``).
"""

from __future__ import annotations

import math

import numpy as np

from ..pipeline.colorspace import DISPLAY_GAMMA
from ..pipeline.lut import LUT_SIZE, compose, is_identity
from ..pipeline.ops import tone, tone_curve
from ..pipeline.params import HSL_BANDS, EditParams
from ..raw.whitepoint import illuminant_xyz
from . import darktable_modules as dt
from .sidecar import SidecarContext
from .xmp_packet import Packet

__all__ = ["build", "history", "multipliers_for"]

_NS = {
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
    "darktable": "http://darktable.sf.net/",
}

#: darktable's own "no blending" parameters (blend version 13), the ones it
#: injects itself when it upgrades an old history (``src/common/exif.cc``).
_NO_BLEND = "gz11eJxjYGBgkGAAgRNODGiAEV0AJ2iwh+CRyscOAAdeGQQ="
_BLEND_VERSION = 13

#: ``DT_IOP_ORDER_V50``, the pipe order of a raw in darktable 5.
_IOP_ORDER_V50 = 4

#: Radius of our local shadows/highlights base layer, as a fraction of the
#: long edge (``ops/local_contrast.py``).
_BASE_RADIUS = 0.04

#: Our full HSL hue slider, in degrees (``ops/color.py``).
_HUE_TRAVEL = 30.0

#: Stops of the tone response sampled for the sigmoid fit: from deep shadow
#: to the brightest highlight a scene-referred frame usually holds.
_FIT_EV = np.linspace(-7.0, 5.0, 97)


def multipliers_for(ctx: SidecarContext, temperature_k: float, tint: float) -> np.ndarray:
    """Camera-space multipliers that neutralise our illuminant, green = 1."""
    xyz = illuminant_xyz(temperature_k, tint)
    camera = np.asarray(ctx.cam_from_xyz, dtype=np.float64) @ xyz
    multipliers = 1.0 / np.maximum(camera, 1e-9)
    return multipliers / multipliers[1]


def _white_balance(params: EditParams, ctx: SidecarContext) -> dt.DtModule:
    wb = params.white_balance
    if wb.mode == "as_shot":
        return dt.temperature(ctx.as_shot_multipliers, as_shot=True)
    return dt.temperature(multipliers_for(ctx, wb.temperature_k, wb.tint), as_shot=False)


def _tone(params: EditParams, ctx: SidecarContext) -> list[dt.DtModule]:
    encoded = tone.sigmoid_response(_FIT_EV, params.tone)
    contrast, skew, offset = dt.fit_sigmoid(_FIT_EV, np.power(encoded, DISPLAY_GAMMA))
    ev = params.exposure.ev + ctx.baseline_exposure_ev + offset
    hue = 100.0 * params.tone.chroma_preservation
    return [dt.exposure(ev), dt.sigmoid(contrast, skew, hue)]


def _curve_nodes(params: EditParams) -> list[tuple[float, float]] | None:
    """Our shaping and curve, encoded -> encoded, as darktable's linear -> linear nodes."""
    lut = compose(
        tone.build_shaping_lut(params.tone_shaping), tone_curve.build_lut(params.tone_curve)
    )
    if is_identity(lut, tolerance=1e-3):
        return None
    # Twenty nodes, spread evenly in the encoded domain: that is where the
    # curve's shape lives, and in linear terms it puts most of them in the
    # shadows, where a monotone Hermite needs them.
    x = np.linspace(0.0, 1.0, 20)
    y = np.interp(x, np.linspace(0.0, 1.0, LUT_SIZE), lut)
    y = np.maximum.accumulate(np.clip(y, 0.0, 1.0))
    return [(float(a**DISPLAY_GAMMA), float(b**DISPLAY_GAMMA)) for a, b in zip(x, y, strict=True)]


def _colour(params: EditParams) -> list[dt.DtModule]:
    colour = params.color
    modules: list[dt.DtModule] = []
    split = colour.split_toning
    if (
        abs(colour.vibrance) > 1e-3
        or abs(colour.saturation) > 1e-3
        or split.shadow_saturation > 1e-3
        or split.highlight_saturation > 1e-3
    ):
        # Colour balance's wheels run to 1.0 of chroma, which is garish; our
        # split-toning strength of 1 is a strong but photographic tint.
        modules.append(
            dt.colorbalancergb(
                vibrance=float(np.clip(colour.vibrance, -1.0, 1.0)),
                chroma=float(np.clip(colour.saturation, -1.0, 1.0)),
                shadows_hue=split.shadow_hue % 360.0,
                shadows_chroma=0.3 * split.shadow_saturation,
                highlights_hue=split.highlight_hue % 360.0,
                highlights_chroma=0.3 * split.highlight_saturation,
            )
        )
    bands = [colour.band(name) for name in HSL_BANDS]
    if any(abs(b.hue) + abs(b.saturation) + abs(b.luminance) > 1e-3 for b in bands):
        modules.append(
            dt.colorequal(
                hue_deg=[b.hue * _HUE_TRAVEL for b in bands],
                saturation=[float(np.clip(1.0 + b.saturation, 0.0, 2.0)) for b in bands],
                brightness=[float(np.clip(1.0 + b.luminance, 0.0, 2.0)) for b in bands],
            )
        )
    return modules


def _local(params: EditParams, ctx: SidecarContext) -> list[dt.DtModule]:
    local, sharpen = params.local_contrast, params.sharpen
    long_edge = float(max(ctx.width or 6000, ctx.height or 4000))
    modules: list[dt.DtModule] = []
    if abs(local.shadows) > 1e-3 or abs(local.highlights) > 1e-3:
        modules.append(
            dt.shadhi(100.0 * local.shadows, 100.0 * local.highlights, _BASE_RADIUS * long_edge)
        )
    if abs(local.clarity) > 1e-3:
        modules.append(dt.bilat(float(np.clip(local.clarity, -1.0, 4.0))))
    if sharpen.amount > 0.0:
        modules.append(
            dt.sharpen(
                float(np.clip(sharpen.radius * long_edge, 0.1, 99.0)),
                float(np.clip(sharpen.amount / 1.5, 0.0, 2.0)),
                0.5,
            )
        )
    return modules


def _straightened_box(
    rotation_deg: float, ctx: SidecarContext
) -> tuple[float, float, float, float]:
    """Our straightened frame as ``ashift``'s crop box.

    ``ashift`` runs before darktable's orientation, on the sensor frame, and
    stores its crop as fractions of the rotated frame's bounding box. Our
    straightened frame is the largest centred rectangle of the original aspect
    inside the rotated picture (``pipeline/geometry.py``); a quarter turn of
    the camera leaves that relation unchanged, so the sensor's own width and
    height are what count.
    """
    from ..pipeline.geometry import inscribed_scale

    width, height = float(ctx.width or 6000), float(ctx.height or 4000)
    if ctx.orientation in (6, 8):
        width, height = height, width
    theta = math.radians(abs(rotation_deg))
    c, s = math.cos(theta), math.sin(theta)
    box_w, box_h = width * c + height * s, width * s + height * c
    k = inscribed_scale(width, height, rotation_deg)
    left = (1.0 - width * k / box_w) / 2.0
    top = (1.0 - height * k / box_h) / 2.0
    return left, 1.0 - left, top, 1.0 - top


def _geometry(params: EditParams, ctx: SidecarContext) -> list[dt.DtModule]:
    from ..pipeline import lens as lens_ops

    geometry = params.geometry
    modules: list[dt.DtModule] = []
    focal = float(ctx.focal_length or 50.0)
    if geometry.lens_correction and ctx.lens_identity is not None:
        profile = lens_ops.profile_for(ctx.lens_identity)
        if profile is not None:
            modules.append(
                dt.lens(
                    ctx.camera_model or "",
                    profile.model,
                    focal,
                    float(ctx.aperture or 0.0),
                    1.0,
                )
            )
    if abs(geometry.rotation_deg) >= 1e-3:
        box = _straightened_box(geometry.rotation_deg, ctx)
        modules.append(dt.ashift(geometry.rotation_deg, focal, 1.0, box))
    if geometry.crop is not None:
        c = geometry.crop
        modules.append(dt.crop(c.x, c.y, c.x + c.width, c.y + c.height))
    return modules


def history(params: EditParams, ctx: SidecarContext) -> list[dt.DtModule]:
    """The modules of the history, in pipe order."""
    entries = [_white_balance(params, ctx)]
    entries += _geometry(params, ctx)
    entries += _tone(params, ctx)
    nodes = _curve_nodes(params)
    if nodes is not None:
        entries.append(dt.tonecurve(nodes))
    entries += _colour(params)
    entries += _local(params, ctx)
    return entries


def build(params: EditParams, ctx: SidecarContext) -> bytes:
    """The complete ``.ARW.xmp`` for the RAW of ``ctx``, as UTF-8 bytes."""
    entries = history(params, ctx)
    packet = Packet(namespaces=dict(_NS))
    if ctx.software:
        packet.set("xmp:CreatorTool", ctx.software)
    packet.update(
        {
            "xmpMM:DerivedFrom": ctx.raw_filename,
            "darktable:xmp_version": "5",
            "darktable:raw_params": "0",
            "darktable:auto_presets_applied": "1",
            "darktable:iop_order_version": str(_IOP_ORDER_V50),
            "darktable:history_end": str(len(entries)),
        }
    )
    packet.seq_of_structs(
        "darktable:history",
        [
            {
                "darktable:num": index,
                "darktable:operation": entry.operation,
                "darktable:enabled": "1" if entry.enabled else "0",
                "darktable:modversion": entry.version,
                "darktable:params": entry.params.hex(),
                "darktable:multi_name": "",
                "darktable:multi_name_hand_edited": "0",
                "darktable:multi_priority": "0",
                "darktable:blendop_version": _BLEND_VERSION,
                "darktable:blendop_params": _NO_BLEND,
            }
            for index, entry in enumerate(entries)
        ],
    )
    return packet.to_bytes(ctx.software or "autoPhotoEdit")
