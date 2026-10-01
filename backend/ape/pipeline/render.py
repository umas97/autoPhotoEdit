# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Orchestration: ``EditParams`` + a decoded RAW -> the final array.

The stage order is the one fixed in docs/SPEC.md section 6.2 and is not negotiable:
everything physical happens in linear scene-referred light, everything
perceptual happens after the tone mapping, and sharpening happens last, on the
pixels that will actually be written.

Two entry points:

``render``
    stateless, one shot. What the export path and the CLI use.

``StageRenderer``
    keeps the output of a few chosen stages, keyed by the parameters that
    produced it, and resumes from the last one still valid. Moving a saturation
    slider therefore costs a colour pass and the two cheap stages after it -- no
    decode, no denoise, no tone mapping. This is what makes the under-150 ms
    preview of section 10 reachable.

The two share the same stage list, so they cannot drift apart.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..raw.decode import DecodedRaw
from . import bands as _bands
from .bands import PER_PIXEL, run_banded
from .params import EditParams
from .stages import STAGES, RenderOptions, _Context

__all__ = [
    "CHECKPOINTS",
    "STAGES",
    "RenderOptions",
    "StageRenderer",
    "render",
    "render_stages",
    "resume",
]


def render(
    decoded: DecodedRaw,
    params: EditParams,
    options: RenderOptions | None = None,
    *,
    consume: bool = False,
) -> np.ndarray:
    """Develop a decoded RAW into the output space.

    Args:
        decoded: the linear scene-referred frame and its camera context.
        params: the complete description of the development.
        options: output space, size and baseline handling.
        consume: the caller has no further use for ``decoded.rgb``; it is
            released as soon as the first stage has produced a frame of its
            own, which at full resolution is 290 MB less for the whole render.
            ``decoded`` keeps its camera and lens context.

    Returns:
        ``(H, W, 3)`` float32 in [0, 1], in ``options.output_space`` and already
        through its transfer function -- ready for quantisation.

    Raises:
        ValueError: a mask refers to a raster that is missing or damaged, or to
            a subject not segmented yet (``masks_store.RasterError``).
    """
    ctx = _Context(decoded=decoded, params=params, options=options or RenderOptions())
    img = decoded.rgb
    index = 0
    while index < len(STAGES):
        if consume and img is not decoded.rgb:
            decoded.rgb = np.empty((0, 0, 3), dtype=np.float32)
            consume = False
        name, function, _key = STAGES[index]
        if name not in PER_PIXEL or img.shape[0] * img.shape[1] < _bands.BAND_MIN_PIXELS:
            img = function(img, ctx)
            index += 1
            continue
        end = index
        while end < len(STAGES) and STAGES[end][0] in PER_PIXEL:
            end += 1
        img = run_banded(img, ctx, [stage[1] for stage in STAGES[index:end]])
        index = end
    return img


def render_stages(
    decoded: DecodedRaw,
    params: EditParams,
    options: RenderOptions | None = None,
) -> dict[str, np.ndarray]:
    """Same as :func:`render` but keeping every intermediate. For tests and debugging."""
    ctx = _Context(decoded=decoded, params=params, options=options or RenderOptions())
    img = decoded.rgb
    out: dict[str, np.ndarray] = {"decode": img}
    for name, function, _key in STAGES:
        img = function(img, ctx)
        out[name] = img
    return out


def resume(
    img: np.ndarray,
    decoded: DecodedRaw,
    params: EditParams,
    options: RenderOptions | None = None,
    *,
    after: str | None,
    until: str | None = None,
) -> np.ndarray:
    """Run the stages that follow ``after`` on ``img``, up to ``until`` included.

    ``after=None`` starts from the decode, i.e. from ``decoded.rgb``'s role.

    ``img`` must be what the chain produces at ``after`` for these parameters;
    this is the caller's promise, and what makes it cheap. The style optimiser
    of section 8.2 keeps the output of the denoise, which a style never
    changes, and resumes from it four hundred times -- on a sample of pixels
    for the per-pixel stages, on the whole frame for the spatial ones.
    """
    ctx = _Context(decoded=decoded, params=params, options=options or RenderOptions())
    names = [name for name, _function, _key in STAGES]
    start = names.index(after) + 1 if after is not None else 0
    stop = names.index(until) + 1 if until is not None else len(STAGES)
    for _name, function, _key in STAGES[start:stop]:
        img = function(img, ctx)
    return img


def _digest(value: Any) -> str:
    import json

    payload = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


_STAGE_INDEX = {name: index for index, (name, _function, _key) in enumerate(STAGES)}

#: Stages whose output is worth keeping between renders.
#:
#: Keeping *every* stage would cost thirteen full buffers per open photo -- some
#: 300 MB on a 2048 px proxy -- against the 400 MB section 26 gives the whole
#: server. These four are the ones a slider actually stops at: after the
#: denoise, which is expensive and rarely touched; at the end of the
#: scene-referred half; after the tone shaping, which is where every colour
#: control begins; and before the resize, which is where sharpening begins.
#: Anything between two checkpoints is cheap enough to redo.
CHECKPOINTS = frozenset({"noise", "exposure", "tone_shaping", "local_contrast"})


@dataclass
class StageRenderer:
    """A renderer that resumes from the last stage whose inputs are unchanged.

    One instance per photo and size being previewed. Section 10 asks for a
    slider to redraw in under 150 ms and forbids re-decoding the RAW for a
    parameter that lives after the tone mapping; this is how that is kept.

    The cumulative digest folds in every upstream stage, so changing a parameter
    early in the chain invalidates everything downstream without any
    bookkeeping, and changing a late one leaves the expensive prefix alone.
    """

    decoded: DecodedRaw
    _cache: dict[str, tuple[str, np.ndarray]] = field(default_factory=dict)
    #: The masks' selections and the digest of the ``masks`` stage that made
    #: them: resuming past that stage needs them, and they are a fraction of a
    #: frame (float16, one channel).
    _selections: tuple[str, dict[int, np.ndarray]] | None = None

    def render(self, params: EditParams, options: RenderOptions | None = None) -> np.ndarray:
        return self._run(params, options, len(STAGES) - 1)

    def through(
        self, params: EditParams, stage: str, options: RenderOptions | None = None
    ) -> np.ndarray:
        """The output of ``stage`` for these parameters, resuming as :meth:`render` does.

        What the masks editor evaluates a selection on is
        ``through(params, "exposure")``: a checkpoint, so right after a preview
        it costs nothing. The result may be a cached array: never write it.
        """
        return self._run(params, options, _STAGE_INDEX[stage])

    def _run(self, params: EditParams, options: RenderOptions | None, stop: int) -> np.ndarray:
        ctx = _Context(decoded=self.decoded, params=params, options=options or RenderOptions())

        cumulative = ""
        digests: list[str] = []
        for _name, _function, key in STAGES[: stop + 1]:
            cumulative = _digest([cumulative, key(ctx)])
            digests.append(cumulative)

        # The furthest checkpoint that is still valid. Everything before it is
        # by construction valid too, since each digest contains its predecessor.
        start = 0
        img = self.decoded.rgb
        for index, (name, _function, _key) in enumerate(STAGES[: stop + 1]):
            cached = self._cache.get(name)
            if cached is not None and cached[0] == digests[index]:
                start, img = index + 1, cached[1]

        masks_at = _STAGE_INDEX["masks"]
        # Valid by the same argument as the checkpoints: a checkpoint past the
        # masks stage carries that stage's digest inside its own.
        if (
            start > masks_at
            and self._selections is not None
            and self._selections[0] == digests[masks_at]
        ):
            ctx.selections = self._selections[1]

        for index in range(start, stop + 1):
            name, function, _key = STAGES[index]
            img = function(img, ctx)
            if name in CHECKPOINTS:
                self._cache[name] = (digests[index], img)
            if index == masks_at:
                self._selections = (digests[index], ctx.selections or {})
        return img

    def invalidate(self) -> None:
        self._cache.clear()
        self._selections = None

    def cached_bytes(self) -> int:
        """How much memory this renderer is holding. For the diagnostics of section 19."""
        held = sum(array.nbytes for _digest_value, array in self._cache.values())
        if self._selections is not None:
            held += sum(array.nbytes for array in self._selections[1].values())
        return held
