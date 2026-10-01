# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which patch an eraser needs, whether it exists, and making it (docs/SPEC_rimozione.md 4.3).

**The key of a fill** is the SHA-256 of everything it is a function of: the
area, its expansion and fade, the engine and its version, the seed -- and the
fingerprint of what the eraser saw: the photo (its content, or the merge's
recipe), the proxy's size, the lens correction and its profile, and every
visible removal before it in the list. Nothing of the development is in it:
no slider after the lens can make a fill stale.

The patch is saved under its key (``retouch_store``), which makes the store a
memo: the export finds the fill the editor made, a gesture undone finds its
old fill, two workers computing the same fill write the same file.

**How the reference travels** (``EraseItem.fill``), so that a patch arriving
is not a gesture (section 6 of the prompt, "Storico"): nothing rewrites a
version when a worker finishes. The server resolves ``fill`` whenever it uses
parameters -- a preview, a save, an export: the patch of the current key if it
exists; otherwise the one the item already names (an old one, shown until the
new one arrives, as the prompt asks); otherwise the one the same item named in
the photo's current version; otherwise none. A saved version carries what was
known when it was saved, and is never touched again.

:func:`compute_fills` is the one function that makes fills, used by the
``retouch_fill`` job and by the export alike: same inputs, same bits.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .. import retouch_store
from ..config import get_settings
from ..pipeline import lens as lens_ops
from ..pipeline.ops.erase import erase_into
from ..pipeline.ops.heal import heal_into
from ..pipeline.params import EditParams
from ..pipeline.retouch import area_for
from ..pipeline.retouch_params import EraseItem, HealItem

__all__ = [
    "FillUnavailable",
    "compute_fills",
    "engine_id",
    "item_keys",
    "lens_key",
    "photo_token",
    "resolve",
    "upstream_base",
]

#: Bumped when the meaning of a key changes (what goes into it).
KEY_VERSION = 1


class FillUnavailable(RuntimeError):
    """The engine an eraser asks for cannot run here (the model is missing)."""


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def photo_token(photo: Any) -> str:
    """What the photo's pixels are: the RAW's content, or the merge's recipe.

    A merged photo's intermediate is named by the digest of its recipe and
    regenerates to the same bits (section 25.1): the derived photo's own
    fingerprint, as the prompt asks.
    """
    if photo.path and photo.hash:
        return f"raw:{photo.hash}"
    if photo.intermediate_path:
        return f"merge:{Path(photo.intermediate_path).stem}"
    return f"photo:{photo.id}"


def lens_key(params: EditParams, decoded: Any) -> Any:
    """What the lens stage did to the frame: as ``stages._lens_key``."""
    if not params.geometry.lens_correction or decoded.lens is None:
        return None
    return [repr(decoded.lens), repr(lens_ops.profile_for(decoded.lens))]


def upstream_base(token: str, lens: Any) -> str:
    """The fingerprint of the frame the first removal sees."""
    return _digest([KEY_VERSION, token, get_settings().proxy_long_edge, lens])


def engine_id(engine: str) -> str:
    """The engine and the version of what it computes: part of every key."""
    if engine == "classic":
        from .classic import CLASSIC_VERSION

        return f"classic:{CLASSIC_VERSION}"
    from .ml import model_id

    return f"ml:{model_id()}"


def _shown(item: Any) -> bool:
    return item.visible and item.opacity > 0.0


def item_keys(params: EditParams, base: str) -> dict[str, str]:
    """The key of every eraser's fill, by item id, in the order of the list."""
    keys: dict[str, str] = {}
    running = base
    for item in params.retouch:
        if isinstance(item, EraseItem):
            key = _digest(
                {
                    "upstream": running,
                    "engine": engine_id(item.engine),
                    "area": item.area,
                    "expand": item.expand,
                    "feather": item.feather,
                    "seed": item.seed,
                }
            )
            keys[item.id] = key
            if _shown(item):
                running = _digest([running, "erase", key, item.opacity])
        elif _shown(item):
            running = _digest(
                [running, "heal", item.cx, item.cy, item.radius, item.sx, item.sy,
                 item.feather, item.opacity]
            )
    return keys


def resolve(
    params: EditParams, keys: dict[str, str], previous: EditParams | None = None
) -> EditParams:
    """``params`` with each eraser's ``fill`` pointing at the best patch there is.

    The current key's patch; else the one the item names, if it is still
    there (an old fill, shown until the new one arrives); else the one the
    same item had in ``previous``; else none.
    """
    if not keys:
        return params
    before: dict[str, str] = {}
    if previous is not None:
        before = {
            item.id: item.fill
            for item in previous.retouch
            if isinstance(item, EraseItem) and item.fill is not None
        }
    items = []
    changed = False
    for item in params.retouch:
        if isinstance(item, EraseItem):
            fill = None
            for candidate in (keys.get(item.id), item.fill, before.get(item.id)):
                if candidate is not None and retouch_store.exists(candidate):
                    fill = candidate
                    break
            if fill != item.fill:
                item = item.model_copy(update={"fill": fill})
                changed = True
        items.append(item)
    return params.model_copy(update={"retouch": items}) if changed else params


def missing(keys: dict[str, str]) -> dict[str, str]:
    """The fills of ``keys`` that are not in the store yet."""
    return {item: key for item, key in keys.items() if not retouch_store.exists(key)}


Engine = Callable[..., retouch_store.Patch]


def _engine(name: str) -> Engine:
    if name == "classic":
        from .classic import fill_classic

        return fill_classic
    from .ml import fill_ml

    return fill_ml


def compute_fills(
    decoded: Any,
    params: EditParams,
    base: str,
    *,
    progress: Callable[[float], None] | None = None,
    proceed: Callable[[], bool] | None = None,
) -> dict[str, str]:
    """Make every eraser's missing fill, on the proxy, in the order of the list.

    Args:
        decoded: the photo decoded at proxy size (``raw.proxy.editing_proxy``),
            whose lens context matches ``base``.
        params: the photo's parameters.
        base: :func:`upstream_base` for this photo and these parameters.
        progress: 0..1 over the fills to make.
        proceed: asked before each fill; ``False`` stops (a newer gesture made
            this work stale). The fills made so far stay: they are valid for
            their keys.

    Returns:
        The keys, by item id: all of them exist on return unless stopped.

    Raises:
        FillUnavailable: an engine cannot run (the model is missing).

    The chain is the render's own (``pipeline/retouch.py``): the frame after
    the lens, each visible removal applied in turn -- the spots healed at proxy
    size, the areas composed from their patches *as read back from the store*,
    so the frame each fill sees is the one the editor renders.
    """
    keys = item_keys(params, base)
    todo = missing(keys)
    frame = lens_ops.apply(decoded.rgb, decoded.lens, params.geometry.lens_correction)
    frame = np.array(frame, dtype=np.float32, copy=True)
    done = 0
    for item in params.retouch:
        if isinstance(item, HealItem):
            if _shown(item):
                heal_into(frame, item)
            continue
        assert isinstance(item, EraseItem)
        key = keys[item.id]
        area = area_for(item.area, item.expand, item.feather)
        if key in todo.values() and not retouch_store.exists(key):
            if proceed is not None and not proceed():
                return keys
            engine = _engine(item.engine)

            def step(fraction: float, _done: int = done) -> None:
                if progress is not None and todo:
                    progress((_done + fraction) / len(todo))

            patch = engine(frame, area, item.seed, step)
            retouch_store.save(key, patch)
            done += 1
        if _shown(item):
            erase_into(frame, item, retouch_store.load(key), area)
    return keys
