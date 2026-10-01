# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
""""Copia punti sulle foto selezionate" (docs/SPEC_rimozione.md 6, R6).

Made for the dust on a sensor: a speck sits on the same photosites in every
shot of a session, whichever way the camera was held. So the spots travel in
**sensor** coordinates. The frame a removal is written in is upright: LibRaw
turned the sensor's picture by its orientation code (``sizes.flip``, the
dialect of ``raw/embedded.py``: 3 half a turn, 5 a quarter counter-clockwise,
6 a quarter clockwise), and that turn is undone for the photo copied from and
done again for each photo copied to. Radii are fractions of the long edge,
which is the same side of the sensor either way.

The server converts to the sensor and queues one ``retouch_copy`` job per
destination; the job turns the spots into that photo's frame, finds each
spot's source again *there* (the dust is shared, the surface under it is
not), and appends them to the photo's removals as a new version -- an
explicit action of the user, the one way a removal leaves its photo. Erasers
travel only when asked ("anche le gomme"), their areas turned the same way;
their fills are computed for the new photo by the usual job.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import masks_store

__all__ = ["apply_copy", "flip_of", "from_sensor", "to_sensor", "turns"]

_log = logging.getLogger(__name__)

#: LibRaw's orientation code -> quarter turns counter-clockwise from the
#: sensor to the upright frame (``numpy.rot90``'s ``k``).
_TURNS = {0: 0, 3: 2, 5: 1, 6: -1}


def turns(flip: int) -> int:
    return _TURNS.get(int(flip or 0), 0)


def flip_of(path: str | Path) -> int:
    """LibRaw's orientation code of a RAW, read without decoding it. 0 if unreadable."""
    import rawpy

    try:
        with rawpy.imread(str(path)) as raw:
            return int(raw.sizes.flip)
    except Exception:  # an unreadable RAW is reported where it is decoded
        return 0


def to_sensor(x: float, y: float, flip: int) -> tuple[float, float]:
    """A point of the upright frame, on the sensor (both normalised)."""
    k = turns(flip) % 4
    if k == 0:
        return x, y
    if k == 2:
        return 1.0 - x, 1.0 - y
    if k == 1:  # the frame is the sensor turned counter-clockwise
        return 1.0 - y, x
    return y, 1.0 - x


def from_sensor(u: float, v: float, flip: int) -> tuple[float, float]:
    """A point of the sensor, in the upright frame (both normalised)."""
    k = turns(flip) % 4
    if k == 0:
        return u, v
    if k == 2:
        return 1.0 - u, 1.0 - v
    if k == 1:
        return v, 1.0 - u
    return 1.0 - v, u


def _turn_raster(name: str, quarter_turns: int) -> str:
    raster = masks_store.load(name)
    turned = np.ascontiguousarray(np.rot90(raster, quarter_turns))
    ok, encoded = cv2.imencode(".png", turned)
    if not ok:
        raise ValueError("codifica PNG fallita")
    return masks_store.save(encoded.tobytes())


def sensor_items(items: list[Any], flip: int, include_erase: bool) -> list[dict[str, Any]]:
    """The removals of the photo copied from, on the sensor. Spots always, erasers if asked."""
    out: list[dict[str, Any]] = []
    for item in items:
        if item.kind == "heal":
            u, v = to_sensor(item.cx, item.cy, flip)
            out.append({"kind": "heal", "u": u, "v": v, "radius": item.radius,
                        "feather": item.feather, "opacity": item.opacity})
        elif include_erase:
            out.append({
                "kind": "erase",
                # The area as the sensor saw it: turned back from the frame.
                "area": _turn_raster(item.area, -turns(flip)),
                "expand": item.expand, "feather": item.feather, "engine": item.engine,
                "seed": item.seed, "opacity": item.opacity,
            })
    return out


def _new_id(taken: set[str]) -> str:
    while True:
        candidate = f"c{uuid.uuid4().hex[:10]}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate


def apply_copy(maker: Any, payload: dict[str, Any], progress: Callable[[float], None]) -> None:
    """Append the copied removals to one photo. Payload: photo, sensor items."""
    from sqlalchemy import select

    from ..api.routes_photos import add_version
    from ..db.models import EditVersion, EditVersionSource, Photo
    from ..db.session import session_scope
    from ..jobs.handlers_analysis import lens_override_for
    from ..pipeline import lens as lens_ops
    from ..pipeline.ops.heal import find_source, heal_into
    from ..pipeline.params import EditParams, neutral_params
    from ..pipeline.retouch_params import EraseItem, HealItem
    from ..raw.proxy import editing_proxy
    from .service import enqueue_fills, resolve_for_editor

    photo_id = int(payload["photo_id"])
    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        if photo is None or not photo.path:
            return
        path = Path(photo.path)
        override = lens_override_for(session, photo.lens)
    flip = flip_of(path)
    decoded = editing_proxy(path, lens_override=override)
    progress(0.3)

    with session_scope(maker) as session:
        photo = session.get(Photo, photo_id)
        version = session.scalars(
            select(EditVersion).where(
                EditVersion.photo_id == photo_id, EditVersion.is_current.is_(True)
            )
        ).first()
        params = EditParams.from_dict(version.params) if version else neutral_params()
        # The frame the spots are looked at on: this photo's own removals first.
        resolved = resolve_for_editor(session, photo, params, decoded)
        from ..pipeline.retouch import apply as apply_retouch

        frame = lens_ops.apply(decoded.rgb, decoded.lens, params.geometry.lens_correction)
        frame = np.array(apply_retouch(frame, resolved.retouch), dtype=np.float32, copy=True)
        taken = {item.id for item in params.retouch}
        added: list[Any] = []
        for entry in payload["items"]:
            if entry["kind"] == "heal":
                cx, cy = from_sensor(entry["u"], entry["v"], flip)
                sx, sy = find_source(frame, cx, cy, entry["radius"])
                item = HealItem(id=_new_id(taken), cx=cx, cy=cy, radius=entry["radius"],
                                sx=sx, sy=sy, feather=entry["feather"], opacity=entry["opacity"])
                heal_into(frame, item)
            else:
                item = EraseItem(
                    id=_new_id(taken), area=_turn_raster(entry["area"], turns(flip)),
                    expand=entry["expand"], feather=entry["feather"], engine=entry["engine"],
                    seed=entry["seed"], opacity=entry["opacity"],
                )
            added.append(item)
        if not added:
            return
        updated = params.model_copy(update={"retouch": [*params.retouch, *added]})
        add_version(session, photo, updated, EditVersionSource.USER_EDITED)
        if any(isinstance(item, EraseItem) for item in added):
            enqueue_fills(session, photo)
    _log.info("foto %d: %d rimozioni copiate", photo_id, len(added))
    progress(1.0)
