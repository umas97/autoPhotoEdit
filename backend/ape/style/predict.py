# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene -> ``EditParams`` for one project photo (section 8.3).

What a photo contributes to a prediction is already in the catalogue after
its analysis -- scene features, embedding, the camera's white balance -- except
the exposure anchor and the automatic white balance, which are measured here on
the browsing proxy the first time and then kept in ``Photo.analysis["auto"]``.

Used by the ``predict`` job for the whole project, and synchronously by the
interface for the comparison of section 22 ("questo profilo *vs* neutro"),
which predicts one photo with one profile and applies no coherence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analysis.embed import decode_embedding
from ..analysis.scene import decode_features
from ..pipeline.params import EditParams
from . import auto
from . import vector as sv
from .sample import SceneDescription

__all__ = ["AUTO_VERSION", "describe_photo", "params_for"]

#: Bumped when ``auto.measure`` changes meaning: stored values are remeasured.
#: 3: the mixed-light spread and the tonal tails of phase 7 (``review/measure.py``).
AUTO_VERSION = 3


def _measure(proxy: Path, as_shot: dict) -> dict:
    from PIL import Image

    from ..review import measure as review_measure

    with Image.open(proxy) as image:
        pixels = np.asarray(image.convert("RGB"))
    shot = {"as_shot_temperature_k": as_shot["temperature_k"], "as_shot_tint": as_shot["tint"]}
    result = auto.measure(pixels, **shot)
    return {
        "version": AUTO_VERSION,
        "key_ev": round(result.key_ev, 4),
        "exposure_anchor_ev": round(result.exposure_anchor_ev, 4),
        "wb_mired_shift": round(result.wb_mired_shift, 3),
        "wb_tint_shift": round(result.wb_tint_shift, 3),
        **review_measure.measure(pixels, **shot),
    }


def describe_photo(
    scene_features: bytes | None,
    embedding: bytes | None,
    analysis: dict | None,
    proxy_path: str | None,
) -> tuple[SceneDescription, dict]:
    """The scene description of an analysed photo, and its ``auto`` measurement.

    Raises:
        ValueError: the photo has not been analysed yet.
    """
    features = decode_features(scene_features)
    proxy = Path(proxy_path) if proxy_path else None
    if features is None or proxy is None or not proxy.is_file():
        raise ValueError("la foto non è ancora stata analizzata")
    analysis = analysis or {}
    as_shot = analysis.get("as_shot") or {"temperature_k": 5500.0, "tint": 0.0}
    measured = analysis.get("auto")
    if not measured or measured.get("version") != AUTO_VERSION:
        measured = _measure(proxy, as_shot)
    context = sv.StyleContext(
        as_shot_temperature_k=float(as_shot["temperature_k"]),
        as_shot_tint=float(as_shot["tint"]),
        exposure_anchor_ev=float(measured["exposure_anchor_ev"]),
    )
    scene = SceneDescription(
        features=features,
        embedding=decode_embedding(embedding),
        context=context,
        auto=auto.AutoResult(
            measured["key_ev"],
            measured["exposure_anchor_ev"],
            measured["wb_mired_shift"],
            measured["wb_tint_shift"],
        ),
    )
    return scene, measured


def params_for(profile, scene: SceneDescription, base: EditParams) -> EditParams:
    """One profile's parameters for one photo, on top of its own non-style ones."""
    prediction = profile.predict(scene)
    return sv.to_params(prediction.vector, scene.context, base=base)
