# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``EditParams``: the single contract between every subsystem (docs/SPEC.md section 5).

Style learning produces one of these, the UI shows and edits one, the renderer
consumes one, the XMP exporters translate one. No module produces pixels without
going through a serialisable ``EditParams`` first -- that is what makes a result
reproducible, versionable and editable by hand.

Two conventions run through the whole schema:

* every radius is a **fraction of the image's long edge**, never a pixel count,
  so the same parameters mean the same thing at 1024 px and at 24 MP;
* every parameter has an explicit physical range, because the style optimiser
  (section 8.2) searches inside those bounds.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .retouch_params import RetouchItem

#: Bumped whenever the serialised layout changes. Old JSON keeps loading through
#: the migration registry below -- existing projects are never broken.
#: 2: the tint changed sign, to Adobe's (``raw/whitepoint.py``).
#: 3: the removals, ``retouch`` (``retouch_params.py``).
PARAMS_VERSION = 3

#: Hue centres of the eight HSL bands, in degrees, matching the industry
#: convention users already know from Lightroom and darktable.
HSL_BANDS: tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "green",
    "aqua",
    "blue",
    "purple",
    "magenta",
)
HSL_BAND_HUES: tuple[float, ...] = (0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0)


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


Unit = Annotated[float, Field(ge=-1.0, le=1.0)]
"""A symmetric -1..+1 user control. 0 is always "do nothing"."""


class WhiteBalanceParams(_Params):
    """Target illuminant. Applied as a chromatic adaptation in the working space.

    ``mode='as_shot'`` ignores the numbers and keeps the camera's own setting, so
    a freshly imported photo renders exactly as the camera intended.
    """

    mode: Literal["as_shot", "custom"] = "as_shot"
    temperature_k: Annotated[float, Field(ge=1500.0, le=25000.0)] = 5500.0
    # Raising it makes the picture more magenta, lowering it greener, as in
    # Lightroom (``raw/whitepoint.py``).
    tint: Annotated[float, Field(ge=-150.0, le=150.0)] = 0.0


class ExposureParams(_Params):
    """Linear gain, in stops, applied scene-referred."""

    ev: Annotated[float, Field(ge=-6.0, le=6.0)] = 0.0


class HighlightRecoveryParams(_Params):
    """Reconstruction of channels the sensor clipped, before tone mapping."""

    strength: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    # Fraction of the white level above which a channel counts as blown.
    threshold: Annotated[float, Field(ge=0.5, le=1.0)] = 0.96


class NoiseParams(_Params):
    """Light denoise, scene-referred, before the tone mapping amplifies the noise."""

    luminance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    chrominance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    # Long-edge-relative radius of the chroma filter.
    radius: Annotated[float, Field(ge=0.0005, le=0.02)] = 0.004


class ToneParams(_Params):
    """The sigmoid tone mapping: the one bridge from scene- to display-referred.

    ``contrast`` is a multiplier on the slope through the pivot, where 1.0 is the
    slope a plain linear-to-display transfer would have. The default of 1.2 adds
    the amount of S-curve a normal photographic rendering carries.
    """

    black_point_ev: Annotated[float, Field(ge=-16.0, le=-1.0)] = -8.0
    white_point_ev: Annotated[float, Field(ge=1.0, le=12.0)] = 4.5
    contrast: Annotated[float, Field(ge=0.3, le=3.0)] = 1.2
    # Shifts what counts as middle grey, in stops.
    pivot: Annotated[float, Field(ge=-3.0, le=3.0)] = 0.0
    toe: Annotated[float, Field(ge=0.2, le=6.0)] = 1.2
    shoulder: Annotated[float, Field(ge=0.2, le=6.0)] = 1.5
    # AgX-style inset/outset. 0 is a plain per-channel sigmoid (hues drift as one
    # channel saturates first); 1 compresses chroma hardest before the curve and
    # restores it after, keeping hues stable through the highlights.
    chroma_preservation: Annotated[float, Field(ge=0.0, le=1.0)] = 0.4

    @model_validator(mode="after")
    def _check_range(self) -> ToneParams:
        if self.white_point_ev <= self.black_point_ev:
            raise ValueError("white_point_ev must sit above black_point_ev")
        return self


class ToneShapingParams(_Params):
    """Shadows / highlights / whites / blacks, applied after the sigmoid."""

    shadows: Unit = 0.0
    highlights: Unit = 0.0
    whites: Unit = 0.0
    blacks: Unit = 0.0


class ToneCurveParams(_Params):
    """Parametric four-region curve plus a free user spline, both display-referred."""

    highlights: Unit = 0.0
    lights: Unit = 0.0
    darks: Unit = 0.0
    shadows: Unit = 0.0
    #: Control points in [0, 1]x[0, 1]. Empty means identity. Interpolated with a
    #: monotonic spline, so the curve can never fold back on itself.
    points: list[tuple[float, float]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_points(self) -> ToneCurveParams:
        if not self.points:
            return self
        if len(self.points) < 2:
            raise ValueError("a curve needs at least two points, or none at all")
        xs = [p[0] for p in self.points]
        if any(not 0.0 <= v <= 1.0 for pt in self.points for v in pt):
            raise ValueError("curve points must lie inside [0, 1]")
        if any(b <= a for a, b in zip(xs, xs[1:], strict=False)):
            raise ValueError("curve points must be sorted by strictly increasing x")
        return self


class HSLBand(_Params):
    """Per-band hue rotation, saturation and luminance offsets."""

    hue: Unit = 0.0
    saturation: Unit = 0.0
    luminance: Unit = 0.0


class SplitToningParams(_Params):
    shadow_hue: Annotated[float, Field(ge=0.0, lt=360.0)] = 0.0
    shadow_saturation: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    highlight_hue: Annotated[float, Field(ge=0.0, lt=360.0)] = 0.0
    highlight_saturation: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    balance: Unit = 0.0


class ColorParams(_Params):
    """Saturation, vibrance, the eight HSL bands and split toning."""

    saturation: Unit = 0.0
    #: Vibrance raises muted colours more than already-saturated ones and holds
    #: back skin-tone hues, which is what makes it safe to push.
    vibrance: Unit = 0.0
    hsl: dict[str, HSLBand] = Field(default_factory=dict)
    split_toning: SplitToningParams = Field(default_factory=SplitToningParams)

    @model_validator(mode="after")
    def _check_bands(self) -> ColorParams:
        unknown = set(self.hsl) - set(HSL_BANDS)
        if unknown:
            raise ValueError(f"unknown HSL bands: {sorted(unknown)}")
        return self

    def band(self, name: str) -> HSLBand:
        return self.hsl.get(name) or HSLBand()


class LocalContrastParams(_Params):
    """Clarity and local shadows/highlights, through guided filters, display-referred.

    ``shadows`` and ``highlights`` act on the *base layer* -- the large-scale
    brightness of a region -- and leave the detail inside it alone, which is
    what Lightroom's and Camera Raw's sliders of the same name do. The global
    curve of ``ToneShapingParams`` cannot: darkening a bright sky with a curve
    also darkens every bright detail in the shadows. Added in phase 6, because
    without them no edit made in those programs could be learned faithfully;
    at 0 the stage is skipped and the render is unchanged.
    """

    clarity: Unit = 0.0
    radius: Annotated[float, Field(ge=0.002, le=0.10)] = 0.02
    shadows: Unit = 0.0
    highlights: Unit = 0.0


class SharpenParams(_Params):
    """Output-referred unsharp mask, applied after the final resize."""

    amount: Annotated[float, Field(ge=0.0, le=3.0)] = 0.0
    radius: Annotated[float, Field(ge=0.0002, le=0.01)] = 0.0008
    # Edges weaker than this (in display-encoded units) are left alone, so noise
    # and smooth gradients are not sharpened into artefacts.
    threshold: Annotated[float, Field(ge=0.0, le=0.2)] = 0.01


class CropRect(_Params):
    """Crop in normalised coordinates, relative to the straightened frame."""

    x: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    y: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    width: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    height: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0

    @model_validator(mode="after")
    def _check_bounds(self) -> CropRect:
        if self.x + self.width > 1.0 + 1e-6 or self.y + self.height > 1.0 + 1e-6:
            raise ValueError("crop rectangle falls outside the frame")
        return self


class GeometryParams(_Params):
    """Lens correction, straightening and crop. Applied from phase 5 on."""

    lens_correction: bool = True
    rotation_deg: Annotated[float, Field(ge=-45.0, le=45.0)] = 0.0
    crop: CropRect | None = None


class MaskParams(_Params):
    """One mask and the adjustments it carries (section 6.3).

    The selection is ``kind`` + ``definition`` (``mask_defs.py`` says what a
    definition may contain), then ``invert``, then ``opacity``. Each adjustment
    is applied at its own stage of section 6.2 -- the exposure in linear light
    just before the tone mapping, the rest after it -- blended with the global
    result by the selection (``pipeline/local.py``).
    """

    kind: Literal["parametric", "linear", "radial", "brush", "segment"]
    #: What the interface calls it ("Cielo", "Radiale 2"). Not used by the render.
    name: Annotated[str, Field(max_length=60)] = ""
    invert: bool = False
    opacity: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    #: Geometry or selection ranges, shape depending on ``kind``.
    definition: dict[str, Any] = Field(default_factory=dict)
    #: Only the adjustments a mask may carry; geometry and tone mapping are global.
    exposure: ExposureParams = Field(default_factory=ExposureParams)
    tone_shaping: ToneShapingParams = Field(default_factory=ToneShapingParams)
    color: ColorParams = Field(default_factory=ColorParams)
    local_contrast: LocalContrastParams = Field(default_factory=LocalContrastParams)

    @model_validator(mode="after")
    def _definition_fits_kind(self) -> MaskParams:
        from .mask_defs import parse

        parse(self.kind, self.definition)
        return self


class EditParams(_Params):
    """The complete, serialisable description of one photo's development."""

    params_version: int = PARAMS_VERSION
    white_balance: WhiteBalanceParams = Field(default_factory=WhiteBalanceParams)
    exposure: ExposureParams = Field(default_factory=ExposureParams)
    highlight_recovery: HighlightRecoveryParams = Field(default_factory=HighlightRecoveryParams)
    noise: NoiseParams = Field(default_factory=NoiseParams)
    tone: ToneParams = Field(default_factory=ToneParams)
    tone_shaping: ToneShapingParams = Field(default_factory=ToneShapingParams)
    tone_curve: ToneCurveParams = Field(default_factory=ToneCurveParams)
    color: ColorParams = Field(default_factory=ColorParams)
    local_contrast: LocalContrastParams = Field(default_factory=LocalContrastParams)
    sharpen: SharpenParams = Field(default_factory=SharpenParams)
    geometry: GeometryParams = Field(default_factory=GeometryParams)
    masks: list[MaskParams] = Field(default_factory=list)
    #: The removals, in the order they apply. Each photo's own: no style,
    #: scene or preset reads or writes them.
    retouch: Annotated[list[RetouchItem], Field(max_length=500)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_retouch_ids(self) -> EditParams:
        ids = [item.id for item in self.retouch]
        if len(set(ids)) != len(ids):
            raise ValueError("two removals with the same id")
        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise. Keys stay in declaration order, so diffs stay readable."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str | bytes) -> EditParams:
        """Load, migrating older layouts forward (docs/SPEC.md section 11)."""
        import json

        return cls.from_dict(json.loads(data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EditParams:
        return cls.model_validate(migrate(dict(data)))


#: ``version -> function`` that rewrites a payload of that version into the next.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def register_migration(
    from_version: int,
) -> Callable[[Callable[[dict[str, Any]], dict[str, Any]]], Callable[..., Any]]:
    """Register the rewrite from ``from_version`` to ``from_version + 1``."""

    def decorator(func: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[..., Any]:
        MIGRATIONS[from_version] = func
        return func

    return decorator


def migrate(payload: dict[str, Any]) -> dict[str, Any]:
    """Bring a serialised payload up to :data:`PARAMS_VERSION`.

    Raises:
        ValueError: if the payload is newer than this build, or if a step in the
            chain is missing -- both are bugs we want loud, not silently wrong
            pixels.
    """
    version = int(payload.get("params_version", PARAMS_VERSION))
    if version > PARAMS_VERSION:
        raise ValueError(
            f"i parametri sono in formato {version}, questa versione del programma "
            f"arriva al {PARAMS_VERSION}: aggiorna autoPhotoEdit"
        )
    while version < PARAMS_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise ValueError(f"manca la migrazione dei parametri dalla versione {version}")
        payload = step(payload)
        version += 1
        payload["params_version"] = version
    return payload


@register_migration(1)
def _tint_to_adobe_sign(payload: dict[str, Any]) -> dict[str, Any]:
    """Version 1 had the tint the other way round: +50 turned the picture green.

    Negating it describes the same illuminant in the new convention, so the
    render is identical to the bit (``tests/test_tint_sign.py``). The as-shot
    mode ignores the number, but it is flipped anyway: switching to custom
    must start from the same light.
    """
    balance = payload.get("white_balance")
    if isinstance(balance, dict) and "tint" in balance:
        # ``0.0 - x`` rather than ``-x``: a negative zero would serialise as
        # "-0.0" and change the style signature of an untouched photo.
        flipped = 0.0 - float(balance["tint"])
        payload = {**payload, "white_balance": {**balance, "tint": flipped}}
    return payload


@register_migration(2)
def _add_retouch(payload: dict[str, Any]) -> dict[str, Any]:
    """Version 2 had no removals: an empty list, which renders exactly as before."""
    return {**payload, "retouch": list(payload.get("retouch") or [])}


def neutral_params() -> EditParams:
    """Parameters that develop a RAW as neutrally as the pipeline can.

    Not the same as "all zeros": a scene-referred RAW needs *some* tone mapping
    to be viewable at all. This is the baseline the style optimiser regularises
    towards (section 8.2) and the starting point of the built-in Neutral profile
    (section 22).
    """
    return EditParams()
