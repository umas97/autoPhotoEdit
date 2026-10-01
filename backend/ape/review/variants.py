# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The two or three alternatives offered for a photo in the review queue (section 9.2).

A photo is in the queue because the program is unsure *of something*, and the
alternatives answer that something rather than being random nudges:

* unsure of the **scene** (far from the samples, neighbours that disagree,
  exposure extrapolated): the neighbours' own edit, when the regression that
  won some parameters moved away from it; a stop brighter and darker, since
  exposure is where the user's edits vary most (measured in phase 6); and the
  *Neutro automatico* of section 22;
* unsure of the **light**: the camera's white balance and the automatic one;
* a **burnt** or **crushed** edit: the same edit half a stop the other way;
* unsure of the **geometry**: the frame without the automatic rotation.

Everything else of the photo -- crop, masks, noise, sharpening -- stays as it
is. Alternatives that would look the same as the current edit, or as each
other, are dropped. Pure: parameters in, parameters out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pipeline.params import EditParams
from ..style import vector as sv

__all__ = ["MAX_VARIANTS", "Variant", "VariantInputs", "variants"]

MAX_VARIANTS = 3

#: Half a stop: the step a photographer asks for when an exposure is "nearly".
EXPOSURE_STEP_EV = 0.5

#: Two vectors closer than this in every entry (``sv.UNITS``, "the smallest
#: change a photographer would call different") are the same variant.
_SAME = 0.5

_EXPOSURE = sv.index("exposure_offset")
_WB = sv.index("wb_mired_shift")
_TINT = sv.index("wb_tint_shift")


@dataclass(frozen=True, slots=True)
class Variant:
    #: ``neighbours`` | ``brighter`` | ``darker`` | ``neutral`` | ``camera_wb``
    #: | ``auto_wb`` | ``no_rotation``: the interface names it.
    key: str
    params: EditParams


@dataclass(slots=True)
class VariantInputs:
    current: EditParams
    context: sv.StyleContext
    #: Reason codes of ``review/confidence.py``, heaviest first.
    reasons: list[str]
    #: The k-NN's own vector, for a learned profile.
    knn_vector: np.ndarray | None = None
    #: The *Neutro automatico* vector for this photo.
    neutral_vector: np.ndarray | None = None
    #: The automatic white balance, as a shift from as-shot.
    auto_wb: tuple[float, float] | None = None


def _shifted(vector: np.ndarray, index: int, delta: float) -> np.ndarray:
    out = vector.copy()
    out[index] += delta
    return sv.clip(out)


def _candidates(inputs: VariantInputs, vector: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
    """``(key, style vector)`` in order of preference; ``None`` = geometry only."""
    brighter = ("brighter", _shifted(vector, _EXPOSURE, EXPOSURE_STEP_EV))
    darker = ("darker", _shifted(vector, _EXPOSURE, -EXPOSURE_STEP_EV))
    scene: list[tuple[str, np.ndarray | None]] = []
    if inputs.knn_vector is not None:
        scene.append(("neighbours", sv.clip(np.asarray(inputs.knn_vector, dtype=np.float64))))
    scene += [brighter, darker]
    if inputs.neutral_vector is not None:
        scene.append(("neutral", sv.clip(np.asarray(inputs.neutral_vector, dtype=np.float64))))

    light: list[tuple[str, np.ndarray | None]] = []
    camera = vector.copy()
    camera[_WB] = camera[_TINT] = 0.0
    light.append(("camera_wb", camera))
    if inputs.auto_wb is not None:
        automatic = vector.copy()
        automatic[_WB], automatic[_TINT] = inputs.auto_wb
        light.append(("auto_wb", sv.clip(automatic)))

    by_reason: dict[str, list[tuple[str, np.ndarray | None]]] = {
        "far": scene,
        "ambiguous": scene,
        "extrapolation": [brighter, darker, *scene],
        "white_balance": light,
        "burnt": [darker],
        "crushed": [brighter],
        "geometry": [("no_rotation", None)] if inputs.current.geometry.rotation_deg else [],
    }
    ordered: list[tuple[str, np.ndarray | None]] = []
    for reason in inputs.reasons:
        ordered += by_reason.get(reason, [])
    # Whatever the reasons, the queue always offers something to choose from.
    ordered += scene
    return ordered


def _distinct(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.max(np.abs(a - b) / sv.UNITS) >= _SAME)


def variants(inputs: VariantInputs) -> list[Variant]:
    """Up to :data:`MAX_VARIANTS` alternatives to the photo's current edit."""
    current_vector = sv.from_params(inputs.current, inputs.context)
    chosen: list[tuple[str, np.ndarray | None]] = []
    for key, vector in _candidates(inputs, current_vector):
        if any(key == k for k, _v in chosen):
            continue
        if vector is not None:
            if not _distinct(vector, current_vector):
                continue
            if any(v is not None and not _distinct(vector, v) for _k, v in chosen):
                continue
        chosen.append((key, vector))
        if len(chosen) == MAX_VARIANTS:
            break

    out = []
    for key, vector in chosen:
        if vector is None:
            params = inputs.current.model_copy(deep=True)
            params.geometry.rotation_deg = 0.0
        else:
            params = sv.to_params(vector, inputs.context, base=inputs.current)
        out.append(Variant(key, params))
    return out
