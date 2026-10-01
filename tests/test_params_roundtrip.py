# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 2 of docs/SPEC.md section 13: parameter round-trip and versioning.

"``EditParams -> JSON -> EditParams`` identical; the versioned JSON of a
previous version loads correctly."

The second half: the real migration of version 1 (the tint's sign) is tested
in ``test_tint_sign.py``; here a synthetic version 0 exercises the machinery
itself -- a chain of more than one step, and a missing step.
"""

from __future__ import annotations

import json

import pytest

from ape.pipeline.params import (
    MIGRATIONS,
    PARAMS_VERSION,
    EditParams,
    HSLBand,
    neutral_params,
    register_migration,
)


def _busy_params() -> EditParams:
    """Parameters with every section away from its default."""
    return EditParams.model_validate(
        {
            "white_balance": {"mode": "custom", "temperature_k": 4200.0, "tint": -12.5},
            "exposure": {"ev": -0.75},
            "highlight_recovery": {"strength": 0.42, "threshold": 0.9},
            "noise": {"luminance": 0.3, "chrominance": 0.6, "radius": 0.006},
            "tone": {
                "black_point_ev": -9.5,
                "white_point_ev": 5.25,
                "contrast": 1.45,
                "pivot": 0.3,
                "toe": 2.0,
                "shoulder": 0.8,
                "chroma_preservation": 0.65,
            },
            "tone_shaping": {"shadows": 0.4, "highlights": -0.3, "whites": 0.1, "blacks": -0.2},
            "tone_curve": {
                "highlights": 0.2,
                "lights": -0.1,
                "darks": 0.15,
                "shadows": -0.05,
                "points": [(0.0, 0.02), (0.35, 0.3), (0.7, 0.78), (1.0, 1.0)],
            },
            "color": {
                "saturation": 0.12,
                "vibrance": -0.2,
                "hsl": {
                    "orange": HSLBand(hue=-0.2, saturation=0.1, luminance=0.05).model_dump(),
                    "blue": HSLBand(saturation=0.3).model_dump(),
                },
                "split_toning": {
                    "shadow_hue": 210.0,
                    "shadow_saturation": 0.25,
                    "highlight_hue": 45.0,
                    "highlight_saturation": 0.15,
                    "balance": -0.1,
                },
            },
            "local_contrast": {"clarity": 0.35, "radius": 0.03},
            "sharpen": {"amount": 0.8, "radius": 0.001, "threshold": 0.02},
            "geometry": {"lens_correction": False, "rotation_deg": 1.25},
            "masks": [
                {
                    "kind": "radial",
                    "invert": True,
                    "opacity": 0.8,
                    "definition": {"cx": 0.5, "cy": 0.4, "rx": 0.3, "ry": 0.2, "feather": 0.5},
                    "exposure": {"ev": 0.4},
                }
            ],
            "retouch": [
                {"kind": "heal", "id": "h1", "cx": 0.3, "cy": 0.2, "radius": 0.012,
                 "sx": 0.33, "sy": 0.21, "source_auto": False, "feather": 0.4},
                {"kind": "erase", "id": "e1", "area": "ab" * 32, "expand": 0.004,
                 "engine": "classic", "seed": 3, "fill": "cd" * 32, "opacity": 0.9},
            ],
        }
    )


@pytest.mark.parametrize("factory", [neutral_params, _busy_params], ids=["neutral", "busy"])
def test_json_roundtrip_is_identical(factory):
    original = factory()
    restored = EditParams.from_json(original.to_json())
    assert restored == original
    assert restored.to_json() == original.to_json()


def test_roundtrip_is_stable_across_repeated_passes():
    """Serialising a deserialised object must not drift. Floats are the risk."""
    current = _busy_params()
    for _ in range(5):
        text = current.to_json()
        current = EditParams.from_json(text)
        assert current.to_json() == text


def test_unknown_fields_are_rejected():
    """A typo in a hand-edited file is an error, not a silently ignored key."""
    payload = json.loads(neutral_params().to_json())
    payload["exposure"]["evv"] = 1.0
    with pytest.raises(ValueError):
        EditParams.from_dict(payload)


def test_out_of_range_values_are_rejected():
    payload = json.loads(neutral_params().to_json())
    payload["tone"]["white_point_ev"] = -5.0  # below the black point
    with pytest.raises(ValueError):
        EditParams.from_dict(payload)


def test_a_payload_from_the_future_is_refused():
    payload = json.loads(neutral_params().to_json())
    payload["params_version"] = PARAMS_VERSION + 1
    with pytest.raises(ValueError, match="aggiorna autoPhotoEdit"):
        EditParams.from_dict(payload)


def test_a_payload_from_an_older_version_is_migrated():
    """Drive the migration registry with a real, if synthetic, format change.

    The scenario: version 0 spelled the exposure field ``exposure_ev`` and put it
    at the top level. The migration has to move it and rename it, which is the
    shape every real migration will take.
    """
    version_zero = {
        "params_version": 0,
        "exposure_ev": -1.5,
        "tone": {"contrast": 1.4},
    }

    @register_migration(0)
    def _zero_to_one(payload):
        moved = dict(payload)
        moved["exposure"] = {"ev": moved.pop("exposure_ev", 0.0)}
        return moved

    try:
        migrated = EditParams.from_dict(version_zero)
        assert migrated.params_version == PARAMS_VERSION
        assert migrated.exposure.ev == pytest.approx(-1.5)
        assert migrated.tone.contrast == pytest.approx(1.4)
        # Untouched sections come back at their defaults, not missing.
        assert migrated.color == neutral_params().color
    finally:
        MIGRATIONS.pop(0, None)


def test_a_missing_migration_is_an_error_not_a_guess():
    payload = {"params_version": 0, "exposure": {"ev": 0.0}}
    assert 0 not in MIGRATIONS
    with pytest.raises(ValueError, match="manca la migrazione"):
        EditParams.from_dict(payload)


def test_version_two_loads_with_no_removals():
    """Version 3 added ``retouch``: a version 2 document loads with an empty list."""
    payload = json.loads(neutral_params().to_json())
    payload["params_version"] = 2
    del payload["retouch"]
    migrated = EditParams.from_dict(payload)
    assert migrated.params_version == PARAMS_VERSION
    assert migrated.retouch == []
    assert migrated == neutral_params()
