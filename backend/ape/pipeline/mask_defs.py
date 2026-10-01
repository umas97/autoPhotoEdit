# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``MaskParams.definition`` may contain, per kind (docs/SPEC.md section 6.3).

The definition travels as a plain dict -- the ``EditParams`` schema was fixed
before masks existed -- and is validated here, so a hand-edited file with a
typo is refused when it loads instead of rendering something else.

Coordinates live in the **frame**: the lens-corrected, upright picture before
straightening and crop, normalised to ``[0, 1]`` along each side (``x`` across,
``y`` down). That is where the masks are evaluated, and it is what keeps a mask
on the sky it was drawn on when the rotation or the crop change afterwards.
Lengths that must stay round when the frame is not square -- the radii of a
radial mask -- are fractions of the frame's **long edge**.

Every kind can be refined by ``combine``: a list of further selections, applied
in order, each ``add`` (OR), ``intersect`` (AND) or ``subtract`` (AND NOT) --
the combinations section 6.3 asks for. A radial mask on a face intersected
with a luminance range is the classic use.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "KINDS",
    "SEGMENT_SUBJECTS",
    "CombineStep",
    "HueRange",
    "LinearDef",
    "ParametricDef",
    "RadialDef",
    "RasterDef",
    "SegmentDef",
    "ValueRange",
    "parse",
    "rasters",
]

KINDS = ("parametric", "linear", "radial", "brush", "segment")
SEGMENT_SUBJECTS = ("sky", "person", "skin")

#: A raster is named by the SHA-256 of its PNG: immutable, deduplicated, and a
#: name no path trick can turn into somewhere else.
RASTER_NAME = r"^[0-9a-f]{64}$"

#: Positions may sit outside the frame: a gradient whose full-strength end is
#: above the top edge is how a sky is graduated.
Position = Annotated[float, Field(ge=-2.0, le=3.0)]


class _Def(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValueRange(_Def):
    """``low..high`` at full strength, fading to nothing over ``feather`` either side."""

    low: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    high: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    feather: Annotated[float, Field(ge=0.0, le=0.5)] = 0.1

    @model_validator(mode="after")
    def _ordered(self) -> ValueRange:
        if self.low > self.high:
            raise ValueError("range with low above high")
        return self


class HueRange(_Def):
    """Hues within ``width / 2`` of ``center`` degrees, fading over ``feather`` degrees.

    Degrees on the wheel of the HSL bands (``params.HSL_BAND_HUES``): 0 red,
    120 green, 240 blue.
    """

    center: Annotated[float, Field(ge=0.0, le=360.0)] = 0.0
    width: Annotated[float, Field(ge=0.0, le=360.0)] = 60.0
    feather: Annotated[float, Field(ge=0.0, le=90.0)] = 20.0


class _Shape(_Def):
    combine: list[CombineStep] = Field(default_factory=list)


class LinearDef(_Shape):
    """Full strength before ``(x0, y0)``, nothing past ``(x1, y1)``, a ramp between."""

    x0: Position = 0.5
    y0: Position = 0.0
    x1: Position = 0.5
    y1: Position = 0.5

    @model_validator(mode="after")
    def _not_a_point(self) -> LinearDef:
        if abs(self.x1 - self.x0) < 1e-6 and abs(self.y1 - self.y0) < 1e-6:
            raise ValueError("linear gradient with its two ends in the same place")
        return self


class RadialDef(_Shape):
    """An ellipse, full strength inside ``1 - feather`` of its radius, nothing outside."""

    cx: Position = 0.5
    cy: Position = 0.5
    rx: Annotated[float, Field(gt=0.0, le=3.0)] = 0.25
    ry: Annotated[float, Field(gt=0.0, le=3.0)] = 0.25
    #: Counter-clockwise, as ``geometry.rotation_deg``.
    angle: Annotated[float, Field(ge=-180.0, le=180.0)] = 0.0
    feather: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


class RasterDef(_Shape):
    """A painted mask: an 8-bit PNG in the masks folder, at the frame's aspect."""

    raster: Annotated[str, Field(pattern=RASTER_NAME)]


class ParametricDef(_Shape):
    """A selection by the picture's own values; the ranges given are multiplied."""

    luminance: ValueRange | None = None
    saturation: ValueRange | None = None
    hue: HueRange | None = None


class SegmentDef(_Shape):
    """A subject found by a segmentation model, on request. ``raster`` is its
    result, empty until the model has run on this photo."""

    subject: Literal["sky", "person", "skin"]
    raster: Annotated[str, Field(pattern=RASTER_NAME)] | None = None


class CombineStep(_Def):
    op: Literal["add", "intersect", "subtract"] = "intersect"
    kind: Literal["parametric", "linear", "radial", "brush", "segment"]
    invert: bool = False
    definition: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _flat(self) -> CombineStep:
        shape = _MODELS[self.kind].model_validate(self.definition)
        if shape.combine:
            raise ValueError("a combined selection cannot combine further")
        return self


_Shape.model_rebuild()

_MODELS: dict[str, type[_Shape]] = {
    "parametric": ParametricDef,
    "linear": LinearDef,
    "radial": RadialDef,
    "brush": RasterDef,
    "segment": SegmentDef,
}
for _model in _MODELS.values():
    _model.model_rebuild()
CombineStep.model_rebuild()


def parse(kind: str, definition: dict[str, Any]) -> _Shape:
    """The typed definition of one mask. Raises ``ValueError`` when it is not one."""
    return _MODELS[kind].model_validate(definition)


def rasters(kind: str, definition: dict[str, Any]) -> list[str]:
    """Every raster a mask refers to, its combined selections' included."""
    shape = parse(kind, definition)
    names = [getattr(shape, "raster", None)]
    for step in shape.combine:
        names.append(getattr(parse(step.kind, step.definition), "raster", None))
    return [name for name in names if name]
