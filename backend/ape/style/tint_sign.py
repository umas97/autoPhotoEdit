# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The tint's change of sign, for everything a style stores as plain JSON.

Parameters version 2 turned the tint round to Adobe's convention
(``raw/whitepoint.py``). ``EditParams`` migrate themselves, scene features and
models carry a version and flip when they load; what is left are the JSON
fields with no version of their own: style vectors, as-shot contexts, stored
predictions, the rules of a user copy of a built-in. The catalogue migration of
schema 7 (``db/tint_migration.py``) and the import of a version 1
``.apestyle`` (``style/portable.py``) both go through here.

Every negation is ``0.0 - x``: a negative zero would serialise as "-0.0" and
make an untouched value look changed.
"""

from __future__ import annotations

from typing import Any

from . import vector as sv

__all__ = [
    "flip_context",
    "flip_prediction",
    "flip_rules",
    "flip_vector",
    "negate",
]

_TINT = sv.index("wb_tint_shift")


def negate(value: Any) -> Any:
    return value if value is None else 0.0 - float(value)


def flip_vector(values: list | None) -> list | None:
    """A style vector with its tint shift negated."""
    if not values or len(values) <= _TINT:
        return values
    out = list(values)
    out[_TINT] = negate(out[_TINT])
    return out


def flip_context(context: dict | None) -> dict | None:
    """A context (``as_shot_tint`` and friends) with the camera's tint negated."""
    if not context or "as_shot_tint" not in context:
        return context
    return {**context, "as_shot_tint": negate(context["as_shot_tint"])}


def flip_prediction(prediction: dict | None) -> dict | None:
    """``Photo.prediction`` in the new sign. The signature is the caller's job:
    it digests the parameters, which only the caller has."""
    if not prediction:
        return prediction
    out = dict(prediction)
    for key in ("vector", "knn_vector", "applied_vector"):
        if key in out:
            out[key] = flip_vector(out[key])
    if "context" in out:
        out["context"] = flip_context(out["context"])
    return out


def flip_rules(rules: dict | None) -> dict | None:
    """Built-in rules whose base vector sets a tint shift -- only a user's copy can."""
    base = (rules or {}).get("base")
    if not isinstance(base, dict) or "wb_tint_shift" not in base:
        return rules
    return {**rules, "base": {**base, "wb_tint_shift": negate(base["wb_tint_shift"])}}
