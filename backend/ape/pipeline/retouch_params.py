# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``EditParams.retouch``: the removals of a photo (docs/SPEC_rimozione.md section 3).

A list apart from ``masks``: a removal carries no adjustment, only a place and
how to fill it. The items apply in the order of the list, each on the result
of the ones before -- as in Lightroom.

Coordinates are those of ``mask_defs.py``: the **frame**, lens-corrected and
upright, before straightening and crop, normalised along each side (``x``
across, ``y`` down). Radii are fractions of the frame's long edge.

``heal``
    the spot healing brush: a disc copied from a source disc nearby, its
    colour matched to the edge of the destination. Cheap, so it is computed
    at every render at whatever resolution; the source is chosen once, when
    the spot is made, and saved here -- the render never looks for one.

``erase``
    the magic eraser: an area, painted or taken from a mask, filled with what
    surrounds it. The filling is expensive and not repeatable across
    resolutions, so a worker computes it once on the proxy and saves it
    (``retouch_store``); ``fill`` names that patch. The render only composes.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ENGINES", "EraseItem", "HealItem", "RetouchItem"]

#: As ``mask_defs.RASTER_NAME``: an immutable, content-named file.
_SHA256 = r"^[0-9a-f]{64}$"

ENGINES = ("classic", "ml")

Unit01 = Annotated[float, Field(ge=0.0, le=1.0)]


class _Item(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    #: Stable across edits: what the interface selects, and what a patch
    #: being computed is matched back to.
    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,40}$")]
    visible: bool = True
    opacity: Unit01 = 1.0


class HealItem(_Item):
    """A spot: destination ``(cx, cy)``, radius, and where it is copied from."""

    kind: Literal["heal"] = "heal"
    cx: Unit01
    cy: Unit01
    #: 0.001 is a speck of dust at 1:1 on a 24 MP frame (6 px); 0.15 is a
    #: patch of cheek. Beyond that the eraser is the tool.
    radius: Annotated[float, Field(ge=0.001, le=0.15)] = 0.01
    sx: Unit01
    sy: Unit01
    #: False once the user dragged the source: "Scegli un'altra sorgente"
    #: then starts from there instead of forgetting it.
    source_auto: bool = True
    #: Fraction of the radius over which the copy fades into the photo.
    feather: Unit01 = 0.5


class EraseItem(_Item):
    """An area filled from its surroundings."""

    kind: Literal["erase"] = "erase"
    #: The painted area: an 8-bit raster of ``masks_store``, at the frame's aspect.
    area: Annotated[str, Field(pattern=_SHA256)]
    #: How far the area grows before it is filled, as a fraction of the long
    #: edge: the halo around an object belongs to the object.
    expand: Annotated[float, Field(ge=0.0, le=0.05)] = 0.002
    #: Width of the fade into the photo, as a fraction of the area's radius
    #: (the radius of a disc of the same surface).
    feather: Unit01 = 0.25
    engine: Literal["classic", "ml"] = "classic"
    #: "Altra variante" adds one: the same seed gives the same fill.
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] = 0
    #: The patch the worker computed (``retouch_store``), or ``None`` until
    #: it has. The server fills it in (``retouch/fills.py``); an old one
    #: keeps showing until the new one arrives.
    fill: Annotated[str, Field(pattern=_SHA256)] | None = None


RetouchItem = Annotated[HealItem | EraseItem, Field(discriminator="kind")]
