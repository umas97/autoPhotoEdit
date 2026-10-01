# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Panoramas (section 25.5): the costliest merge, and the one that can fail.

The frames are decoded with the reference's white balance, corrected for their
lens -- distortion bends the overlaps apart and vignetting draws a band at
every seam, and the merged geometry is no lens's any more, so the correction
happens here, before stitching, and the intermediate says to skip it
(``intermediate.context_for(lens=False)``) -- then planned on reduced frames
(``pano_estimate.py``) and composed in bands (``pano_compose.py``).

**Tetto dichiarato** (25.5.4): past :data:`MAX_FRAMES` frames or
:data:`MAX_OUTPUT_MP` megapixels of output the merge is refused before it
starts, with the size and the memory it would take and what to change -- the
output resolution (``options.output_scale``) or the group. "Meglio un rifiuto
spiegato che un OOM killer."

**Crop** (25.5.5): the largest rectangle with no empty pixel is *reported*,
never applied. The analysis of the merged photo proposes it like any crop,
because its frame has empty borders (``analysis/crop.py``).

At full resolution the corrected frames wait in half-precision memory maps
(``scratch.py``): six 24 MP frames are 1.7 GB in single precision.

**Sweeps** (a panorama shot as a burst): only the frames the stitcher needs
are decoded, and the ceiling counts those (``pano_select.py``). The report
names them.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline import lens
from .align import ALIGN_EDGE
from .errors import MergeFailure
from .pano_compose import compose, layout, ram_estimate_mb
from .pano_estimate import Plan, estimate
from .pano_geometry import PROJECTIONS
from .pano_photometry import Photometry
from .pano_select import selected_members
from .scratch import Scratch
from .warp import shrink

__all__ = ["MAX_FRAMES", "MAX_OUTPUT_MP", "run", "run_full"]

MAX_FRAMES = 12
MAX_OUTPUT_MP = 200.0

#: Longest side of the preview's composition; it is rendered at 1024 px.
_PREVIEW_EDGE = 2048

#: Pyramid depth of the blending: 2**5 = 32 px transitions at full
#: resolution, 16 px on the preview's smaller frames.
_LEVELS_FULL, _LEVELS_PREVIEW = 5, 4


def _options(recipe: Any) -> tuple[str | None, float]:
    options = recipe.options or {}
    projection = options.get("projection")
    if projection not in PROJECTIONS:
        projection = None
    try:
        scale = float(options.get("output_scale", 1.0))
    except (TypeError, ValueError):
        scale = 1.0
    return projection, min(1.0, max(0.1, scale))


def _frames(recipe: Any) -> tuple[Any, dict]:
    """The recipe reduced to the frames the stitcher uses, and what to say about it."""
    used = selected_members(recipe.members)
    if len(used) > MAX_FRAMES:
        raise MergeFailure(
            f"{len(used)} scatti sono oltre il tetto di {MAX_FRAMES} per una "
            "panoramica: dividila in due gruppi"
        )
    note = {"members": len(recipe.members), "used": [m.filename for m in used]}
    return replace(recipe, members=used), note


def _load(recipe: Any, *, preview: bool, decode: Callable[..., Any], scratch: Scratch | None,
          step: Callable[[float], None]):
    """Decode and lens-correct every frame; keep them (in RAM or on disk) and reduced."""
    members = list(recipe.members)
    reference = decode(recipe.reference, preview=preview)
    balance = reference.camera.as_shot_multipliers
    sources, smalls = [], []
    for index, member in enumerate(members):
        decoded = reference if member.reference else decode(
            member, preview=preview, white_balance=balance
        )
        if decoded.rgb.shape != reference.rgb.shape:
            raise MergeFailure(f"{member.filename} ha dimensioni diverse dal riferimento")
        corrected = lens.apply(decoded.rgb, decoded.lens)
        small, _ = shrink(corrected, ALIGN_EDGE)
        smalls.append(np.ascontiguousarray(small, dtype=np.float32))
        if scratch is None:
            sources.append(corrected)
        else:
            stored = scratch.frame(corrected.shape)
            for top in range(0, corrected.shape[0], 256):
                stored[top : top + 256] = corrected[top : top + 256]
            sources.append(stored)
        if not member.reference:
            del decoded
        del corrected
        step(0.3 * (index + 1) / len(members))
    return reference, sources, smalls


def _report(plan: Plan, out, frame_size, output_scale: float, levels: int) -> dict:
    width, height = out.size
    crop = None
    if plan.crop is not None:
        x, y, w, h = plan.crop
        roi_w, roi_h = plan.roi[2], plan.roi[3]
        crop = {"x": round(x / roi_w, 4), "y": round(y / roi_h, 4),
                "width": round(w / roi_w, 4), "height": round(h / roi_h, 4)}
    return {
        "projection": plan.projection,
        "vertical": plan.vertical,
        "span_deg": list(plan.span_deg),
        "frames": len(plan.cameras),
        "gains_ev": [round(math.log2(g), 2) for g in plan.gains],
        "vignetting_ev": Photometry(plan.gains, plan.vignetting).corner_ev,
        "pairs": plan.pairs,
        "output": {"width": width, "height": height, "megapixels": round(width * height / 1e6, 1),
                   "scale": output_scale},
        "ram_estimate_mb": round(ram_estimate_mb(out.size, frame_size, levels)),
        "crop": crop,
    }


def _refuse_if_too_large(out, frame_size, output_scale: float) -> None:
    megapixels = out.size[0] * out.size[1] / 1e6
    if megapixels <= MAX_OUTPUT_MP:
        return
    ram = ram_estimate_mb(out.size, frame_size, _LEVELS_FULL)
    suggested = math.floor(output_scale * math.sqrt(MAX_OUTPUT_MP / megapixels) * 20) / 20
    raise MergeFailure(
        f"la panoramica verrebbe di {megapixels:.0f} MP ({out.size[0]}×{out.size[1]}, "
        f"circa {ram / 1000:.1f} GB di memoria), oltre il tetto di {MAX_OUTPUT_MP:.0f} MP: "
        f"riduci la risoluzione al {suggested * 100:.0f}% da «Modifica gruppo» "
        "o dividi la panoramica"
    )


def _full_estimate(recipe: Any, plan: Plan, preview_size: tuple[int, int]) -> dict:
    """What the full merge will produce, said at preview time -- refusal included.

    The full frame size is the reference's EXIF; without it, nothing is said.
    """
    from ..raw.metadata import read_metadata

    try:
        meta = read_metadata(recipe.reference.path)
    except (OSError, ValueError):
        return {}
    if not meta.width or not meta.height:
        return {}
    long_edge = max(meta.width, meta.height)
    factor = long_edge / max(preview_size)
    full_size = (round(preview_size[0] * factor), round(preview_size[1] * factor))
    _, output_scale = _options(recipe)
    out = layout(plan, full_size, output_scale)
    entry: dict[str, Any] = {
        "full": {"width": out.size[0], "height": out.size[1],
                 "megapixels": round(out.size[0] * out.size[1] / 1e6, 1),
                 "ram_estimate_mb": round(ram_estimate_mb(out.size, full_size, _LEVELS_FULL))},
    }
    try:
        _refuse_if_too_large(out, full_size, output_scale)
    except MergeFailure as exc:
        entry["too_large"] = str(exc)
    return entry


def run(recipe: Any, *, preview: bool, progress: Callable[[float], None] | None,
        decode: Callable[..., Any]) -> Any:
    """The panorama in memory: the preview, from half-size decodes.

    Returns a :class:`~.engine.MergeOutcome` with the lens left out of the
    decode: the frames were corrected before stitching.
    """
    from .engine import MergeOutcome

    def step(f: float) -> None:
        if progress is not None:
            progress(f)

    recipe, note = _frames(recipe)
    projection, _ = _options(recipe)
    reference, sources, smalls = _load(recipe, preview=preview, decode=decode, scratch=None,
                                       step=step)
    names = [m.filename for m in recipe.members]
    index = next(i for i, m in enumerate(recipe.members) if m.reference)
    plan = estimate(smalls, names, index, projection=projection)
    step(0.45)
    frame_size = (reference.rgb.shape[1], reference.rgb.shape[0])
    full = layout(plan, frame_size, 1.0)
    output_scale = min(1.0, _PREVIEW_EDGE / max(full.size))
    out = layout(plan, frame_size, output_scale)
    canvas = np.zeros((out.size[1], out.size[0], 3), dtype=np.float32)

    def sink(top: int, rows: np.ndarray) -> None:
        canvas[top : top + rows.shape[0]] = rows

    compose(sources, plan, out, sink, levels=_LEVELS_PREVIEW,
            progress=lambda f: step(0.45 + 0.5 * f))
    report = _report(plan, out, frame_size, output_scale, _LEVELS_PREVIEW) | note
    report.update(_full_estimate(recipe, plan, frame_size))
    reference.rgb = canvas
    reference.lens = None
    step(1.0)
    return MergeOutcome(decoded=reference, report=report, keep_lens=False)


def run_full(
    recipe: Any, destination: Path, context: Callable[[Any, int, int], dict], *,
    reduced_long_edge: int, progress: Callable[[float], None] | None,
    decode: Callable[..., Any],
) -> tuple[Path, dict, tuple[int, int]]:
    """The panorama at full resolution, written band by band into ``destination``."""
    from ..raw.intermediate import IntermediateWriter

    def step(f: float) -> None:
        if progress is not None:
            progress(f)

    recipe, note = _frames(recipe)
    projection, output_scale = _options(recipe)
    names = [m.filename for m in recipe.members]
    index = next(i for i, m in enumerate(recipe.members) if m.reference)
    with Scratch() as scratch:
        reference, sources, smalls = _load(recipe, preview=False, decode=decode,
                                           scratch=scratch, step=step)
        frame_size = (reference.rgb.shape[1], reference.rgb.shape[0])
        plan = estimate(smalls, names, index, projection=projection)
        del smalls
        out = layout(plan, frame_size, output_scale)
        _refuse_if_too_large(out, frame_size, output_scale)
        step(0.35)
        writer = IntermediateWriter(
            destination, out.size[0], out.size[1], context(reference, *out.size),
            reduced_long_edge=reduced_long_edge,
        )
        reference.rgb = None  # the memory map holds it
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(writer.abort)
            compose(sources, plan, out, writer.write_rows, levels=_LEVELS_FULL,
                    progress=lambda f: step(0.35 + 0.6 * f))
            path = writer.close()
            cleanup.pop_all()
    step(1.0)
    return path, _report(plan, out, frame_size, output_scale, _LEVELS_FULL) | note, out.size
