# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The stages of a render, in the order of docs/SPEC.md section 6.2, and what each depends on.

``render.py`` runs them -- in one shot, band by band, or resuming from a cached
stage; this module only says what they are. Every stage is ``(img, ctx) -> img``
and must not write ``img``: it may be the decoded frame or a cached output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..raw.decode import DecodedRaw
from . import geometry, lens, local, retouch
from .colorspace import OutputSpace, display_decode, encode_transfer, working_to_output
from .filters import resize_long_edge
from .lut import apply_lut, compose
from .ops import (
    color,
    exposure,
    highlight_recovery,
    local_contrast,
    noise,
    sharpen,
    tone,
    tone_curve,
    white_balance,
)
from .params import EditParams

__all__ = ["OUTPUT_SHARPEN_SIGMA", "STAGES", "RenderOptions", "Stage"]


@dataclass(frozen=True)
class RenderOptions:
    """Everything about the *output* that is not part of the look.

    Kept out of ``EditParams`` on purpose: a learned style has to stay usable
    whatever size and colour space the user exports to.
    """

    output_space: OutputSpace = OutputSpace.SRGB
    #: Resize before sharpening. ``None`` keeps the native resolution.
    long_edge: int | None = None
    #: Fold in the decoder's standards-derived exposure anchor.
    apply_baseline_exposure: bool = True
    #: Stop after the display-referred stages, skipping the output conversion.
    #: Used by the style optimiser, which compares in the working space.
    stop_before_output: bool = False
    #: Unsharp-mask amount for the output medium (the Export screen's
    #: "nitidezza di output"), at ``OUTPUT_SHARPEN_SIGMA`` output pixels, on top
    #: of the photo's own ``sharpen``. 0 leaves the stage exactly as it was.
    output_sharpening: float = 0.0
    #: Apply ``EditParams.retouch``. Off only for the editor's "mostra
    #: rimozioni" switch, which suspends them without touching the parameters.
    retouch: bool = True


#: Radius of the output sharpening, in output pixels: the scale at which a
#: resampled image loses its crispness, whatever its size.
OUTPUT_SHARPEN_SIGMA = 0.8
#: Edges below this are left alone, as for the photo's own sharpening.
OUTPUT_SHARPEN_THRESHOLD = 0.01


@dataclass
class _Context:
    decoded: DecodedRaw
    params: EditParams
    options: RenderOptions
    #: The masks' selections by index, from the ``masks`` stage on.
    selections: dict[int, np.ndarray] | None = None
    #: The rows of the band being rendered (``bands.run_banded``), or the frame.
    rows: slice | None = None


Stage = Callable[[np.ndarray, _Context], np.ndarray]


def _stage_lens(img: np.ndarray, ctx: _Context) -> np.ndarray:
    """Vignetting, distortion and TCA, on the frame as the sensor recorded it."""
    return lens.apply(img, ctx.decoded.lens, ctx.params.geometry.lens_correction)


def _lens_key(ctx: _Context) -> Any:
    """What the lens stage depends on: the switch, the optics, and the profile.

    The resolved profile is part of the key so that associating a profile by
    hand, or updating the lensfun data, invalidates the cached stages without
    anyone having to remember to.
    """
    if not ctx.params.geometry.lens_correction or ctx.decoded.lens is None:
        return None
    profile = lens.profile_for(ctx.decoded.lens)
    return [repr(ctx.decoded.lens), repr(profile)]


def _stage_retouch(img: np.ndarray, ctx: _Context) -> np.ndarray:
    """The removals (``pipeline/retouch.py``), on the lens-corrected frame."""
    if not ctx.params.retouch or not ctx.options.retouch:
        return img
    return retouch.apply(img, ctx.params.retouch)


def _retouch_key(ctx: _Context) -> Any:
    if not ctx.params.retouch or not ctx.options.retouch:
        return None
    return retouch.stage_key(ctx.params.retouch)


def _stage_noise(img: np.ndarray, ctx: _Context) -> np.ndarray:
    return noise.apply(img, ctx.params.noise)


def _stage_white_balance(img: np.ndarray, ctx: _Context) -> np.ndarray:
    camera = ctx.decoded.camera
    return white_balance.apply(
        img,
        ctx.params.white_balance,
        as_shot_temperature_k=camera.as_shot_temperature_k,
        as_shot_tint=camera.as_shot_tint,
    )


def _stage_highlights(img: np.ndarray, ctx: _Context) -> np.ndarray:
    return highlight_recovery.apply(img, ctx.params.highlight_recovery)


def _stage_exposure(img: np.ndarray, ctx: _Context) -> np.ndarray:
    extra = ctx.decoded.baseline_exposure_ev if ctx.options.apply_baseline_exposure else 0.0
    return exposure.apply(img, ctx.params.exposure, extra_ev=extra)


def _stage_masks(img: np.ndarray, ctx: _Context) -> np.ndarray:
    """Evaluate the selections; the pixels pass through untouched.

    Each mask's adjustments are applied by the stage they belong to
    (``pipeline/local.py``), with the selections worked out here, on the frame
    the definitions are written in.
    """
    if ctx.params.masks:
        from ..masks_store import load

        ctx.selections = local.compute_selections(img, ctx.params.masks, ctx.params.tone, load)
    return img


def _stage_tone(img: np.ndarray, ctx: _Context) -> np.ndarray:
    """The one crossing from scene-referred to display-referred.

    The masks' exposure comes first: still linear light, as section 6.2 puts it.
    """
    if ctx.params.masks:
        img = local.apply_exposure(img, ctx.params.masks, ctx.selections or {}, ctx.rows)
    return tone.apply_sigmoid(img, ctx.params.tone)


def _local(img: np.ndarray, ctx: _Context, section: str, operation: Callable) -> np.ndarray:
    if not ctx.params.masks:
        return img
    return local.blend(
        img, ctx.params.masks, ctx.selections or {}, ctx.rows, section, operation
    )


def _shaping_only(img: np.ndarray, params: Any) -> np.ndarray:
    return apply_lut(img, tone.build_shaping_lut(params))


def _stage_tone_shaping(img: np.ndarray, ctx: _Context) -> np.ndarray:
    """Shadows/highlights/whites/blacks and the tone curve, folded into one table.

    Two consecutive per-channel lookups are one lookup on the composition of
    their tables -- same pixels, half the passes over a full-resolution frame.
    """
    lut = compose(
        tone.build_shaping_lut(ctx.params.tone_shaping),
        tone_curve.build_lut(ctx.params.tone_curve),
    )
    return _local(apply_lut(img, lut), ctx, "tone_shaping", _shaping_only)


def _stage_color(img: np.ndarray, ctx: _Context) -> np.ndarray:
    return _local(color.apply(img, ctx.params.color), ctx, "color", color.apply)


def _stage_local_contrast(img: np.ndarray, ctx: _Context) -> np.ndarray:
    """Never banded (a guided filter reads the whole frame), so ``ctx.rows`` is unset."""
    out = local_contrast.apply(img, ctx.params.local_contrast)
    if not ctx.params.masks:
        return out
    out = local.blend_local_contrast(
        out, ctx.params.masks, ctx.selections or {}, owned=out is not img
    )
    # The last stage that reads the selections: the sharpening, whose peak is
    # the render's, runs without them. ``StageRenderer`` keeps its own copy.
    ctx.selections = None
    return out


def _stage_geometry(img: np.ndarray, ctx: _Context) -> np.ndarray:
    """Straightening and crop. The lens was corrected at the start of the chain."""
    return geometry.apply(img, ctx.params.geometry)


def _stage_resize(img: np.ndarray, ctx: _Context) -> np.ndarray:
    if ctx.options.long_edge is None:
        return img
    return resize_long_edge(img, ctx.options.long_edge)


def _stage_sharpen(img: np.ndarray, ctx: _Context) -> np.ndarray:
    img = sharpen.apply(img, ctx.params.sharpen)
    if ctx.options.output_sharpening > 0.0:
        img = sharpen.unsharp(
            img, OUTPUT_SHARPEN_SIGMA, ctx.options.output_sharpening, OUTPUT_SHARPEN_THRESHOLD
        )
    return img


def _stage_output(img: np.ndarray, ctx: _Context) -> np.ndarray:
    if ctx.options.stop_before_output:
        return img
    linear = display_decode(img)
    converted = working_to_output(linear, ctx.options.output_space)
    return encode_transfer(converted, ctx.options.output_space)


def _fields(*names: str, masks: str | None = None) -> Callable[[_Context], Any]:
    """The key of a stage: its parameters, and its masks' share of ``masks``."""

    def extract(ctx: _Context) -> Any:
        key = [getattr(ctx.params, name).model_dump() for name in names]
        if masks is not None and ctx.params.masks:
            key.append(local.adjustment_key(ctx.params.masks, masks))
        return key

    return extract


#: ``(name, function, what its result depends on)``, in the order of section 6.2,
#: with the removals of docs/SPEC_rimozione.md 4.1 between the lens and the noise::
#:
#:     decode -> lens -> retouch -> noise -> white balance -> highlights
#:     -> exposure -> masks -> tone -> tone shaping + curve -> colour
#:     -> local contrast -> geometry -> resize -> sharpen -> output
#:
#: Each key is folded into the digest of every stage after it
#: (``render.StageRenderer``): a removal that changes invalidates the rest.
STAGES: tuple[tuple[str, Stage, Callable[[_Context], Any]], ...] = (
    ("lens", _stage_lens, _lens_key),
    ("retouch", _stage_retouch, _retouch_key),
    ("noise", _stage_noise, _fields("noise")),
    ("white_balance", _stage_white_balance, _fields("white_balance")),
    ("highlight_recovery", _stage_highlights, _fields("highlight_recovery")),
    ("exposure", _stage_exposure, _fields("exposure")),
    # The parametric ranges read the tone mapping, so it is part of the key.
    (
        "masks",
        _stage_masks,
        lambda ctx: [local.selection_key(ctx.params.masks), ctx.params.tone.model_dump()]
        if ctx.params.masks
        else [],
    ),
    ("tone", _stage_tone, _fields("tone", masks="exposure")),
    (
        "tone_shaping",
        _stage_tone_shaping,
        _fields("tone_shaping", "tone_curve", masks="tone_shaping"),
    ),
    ("color", _stage_color, _fields("color", masks="color")),
    ("local_contrast", _stage_local_contrast, _fields("local_contrast", masks="local_contrast")),
    ("geometry", _stage_geometry, _fields("geometry")),
    ("resize", _stage_resize, lambda ctx: ctx.options.long_edge),
    (
        "sharpen",
        _stage_sharpen,
        lambda ctx: [ctx.params.sharpen.model_dump(), ctx.options.output_sharpening],
    ),
    (
        "output",
        _stage_output,
        lambda ctx: [ctx.options.output_space, ctx.options.stop_before_output],
    ),
)
