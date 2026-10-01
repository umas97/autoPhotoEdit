# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The eraser's IA engine (docs/SPEC_rimozione.md 5 and 8).

Always: the model is pinned and declared, inside the budget of section 26.
With the model on this machine (``slow``): an erased object looks like the
background it hid (test 3), also across a real shadow's edge, the same at
1024 px and at full size (test 4), the same bytes twice and another fill
for another seed (test 5).

The model is looked for in this machine's models folder, or in the folder
named by ``APE_TEST_MODELS`` -- a copy verified against the pin, so the tests
can run without the interface's download.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ape import models_registry

_REAL = Path(os.environ.get("APE_TEST_MODELS")
             or Path.home() / ".local" / "share" / "autophotoedit" / "models")
_ENTRY = models_registry.entry_for("inpaint")


def test_the_model_is_pinned_declared_and_inside_the_budget():
    assert _ENTRY is not None and _ENTRY.sha256 and _ENTRY.download_url
    assert _ENTRY.sha256 in _ENTRY.download_url or "resolve/" in _ENTRY.download_url
    assert _ENTRY.size_bytes <= 120 * 1000**2, "oltre i 120 MB per modello di §26"
    pinned = [m.size_bytes for m in models_registry.MODELS if m.size_bytes]
    assert sum(pinned) <= 350 * 1000**2, "oltre i 350 MB in tutto di §26"
    licences = (Path(__file__).parents[1] / "models" / "LICENSES.md").read_text(encoding="utf-8")
    assert "opencv/inpainting_lama" in licences and "Places2" in licences
    assert _ENTRY.notice == "places2_terms"


@pytest.fixture
def model(monkeypatch):
    from ape.config import get_settings

    if not (_REAL / _ENTRY.filename).is_file():
        pytest.skip("modello IA della gomma non scaricato su questa macchina")
    monkeypatch.setattr(type(get_settings()), "models_dir", property(lambda _self: _REAL))
    ok, reason = __import__("ape.retouch.ml", fromlist=["available"]).available()
    assert ok, reason
    yield
    from ape.retouch import ml

    ml.release_if_idle(force=True)


@pytest.mark.slow
def test_an_object_erased_by_the_model_looks_like_the_background(xdg_home, model):
    """ΔE2000 under 5 after a 2 px blur, where the object stands at more than 30.

    Looser than the classic engine's 3 (``test_retouch_fill.py``) on purpose:
    the int8 LaMa drifts in colour towards the middle of a hole
    on a smooth surface -- 4.7 here, against 1.7 for the same model in FP32 and
    0.45 for the classic engine (measured). The bar
    keeps that drift from growing and catches what would really be wrong: the
    object left in, or the wrong colour space (BGR, display for linear), which
    land far above 5.
    """
    import cv2

    from ape.pipeline.colorspace import delta_e_2000
    from ape.pipeline.render import RenderOptions, render
    from ape.pipeline.retouch_params import EraseItem
    from conftest import as_decoded
    from test_retouch_fill import (
        _OBJECT,
        _background,
        _filled,
        _lab,
        _params,
        _raster,
        _save_raster,
        _with_object,
    )

    truth = _background(2048)
    proxy = _with_object(truth)
    params = _filled(proxy, _params(EraseItem(id="e", area=_save_raster(_raster()), engine="ml")))
    options = RenderOptions(stop_before_output=True)
    erased = render(as_decoded(proxy), params, options)
    true = render(as_decoded(truth), _params(), options)
    h, w = erased.shape[:2]
    cx, cy, r = _OBJECT
    y, x = np.mgrid[0:h, 0:w]
    area = np.hypot(x - cx * w, y - cy * h) < r * w

    def blurred(image):
        return cv2.GaussianBlur(image, (0, 0), 2.0)

    after = float(np.mean(delta_e_2000(_lab(blurred(erased))[area], _lab(blurred(true))[area])))
    assert after < 5.0, after


@pytest.mark.slow
def test_the_model_s_fill_looks_the_same_at_every_size(xdg_home, model):
    from ape.pipeline.colorspace import delta_e_2000
    from ape.pipeline.filters import resize_long_edge
    from ape.pipeline.render import RenderOptions, render
    from ape.pipeline.retouch_params import EraseItem
    from conftest import as_decoded
    from test_retouch_fill import (
        _OBJECT,
        _background,
        _filled,
        _lab,
        _params,
        _raster,
        _save_raster,
        _with_object,
    )

    full = _with_object(_background(4096))
    proxy = resize_long_edge(full, 2048)
    item = EraseItem(id="e", area=_save_raster(_raster()), engine="ml", feather=0.3)
    params = _filled(proxy, _params(item))
    options = RenderOptions(stop_before_output=True, long_edge=1024)
    preview = render(as_decoded(resize_long_edge(proxy, 1024)), params, options)
    exported = render(as_decoded(full), params, options)
    h, w = preview.shape[:2]
    cx, cy, r = _OBJECT
    y, x = np.mgrid[0:h, 0:w]
    area = np.hypot(x - cx * w, y - cy * h) < r * w * 1.1
    assert float(np.mean(delta_e_2000(_lab(preview[area]), _lab(exported[area])))) < 1.5


@pytest.mark.slow
def test_the_model_gives_the_same_bytes_and_another_seed_another_fill(model):
    from ape.pipeline.ops.erase import area_alpha
    from ape.retouch.ml import fill_ml
    from test_retouch_fill import _background, _raster, _with_object

    proxy = _with_object(_background(2048))
    area = area_alpha(_raster(), 0.002, 0.25)
    first, again = fill_ml(proxy, area, 0), fill_ml(proxy, area, 0)
    assert first.bbox == again.bbox
    assert first.structure.tobytes() == again.structure.tobytes()
    assert first.offsets.tobytes() == again.offsets.tobytes()
    assert fill_ml(proxy, area, 1).structure.tobytes() != first.structure.tobytes()


@pytest.mark.slow
def test_an_object_on_a_real_shadow_s_edge_is_filled_without_a_fog(model):
    """A dark disc pasted across the shadow's edge on the frozen pond of
    DSC05634, where the truth is the photo itself.

    ``ML_VERSION`` 2 carried the colour of a ring just outside the hole into it
    as a membrane; where a sharp edge crossed the outline the ring mixed light
    and shadow and the fill was multiplied by 0.4 to 3.4 -- a grey fog on the
    real duck. Measured here: 2.6 and 2.8 now, 5.1 and 5.0 with the membrane.
    """
    import cv2

    from ape.pipeline import lens
    from ape.pipeline.colorspace import delta_e_2000
    from ape.pipeline.ops.erase import area_alpha, erase_into
    from ape.pipeline.params import EditParams
    from ape.pipeline.render import RenderOptions, render
    from ape.pipeline.retouch_params import EraseItem
    from ape.raw.decode import decoded_from_array
    from ape.raw.proxy import editing_proxy
    from ape.retouch.ml import fill_ml
    from conftest import FIXTURES
    from test_retouch_fill import _lab

    raw = FIXTURES / "DSC05634.ARW"
    if not raw.is_file():
        pytest.skip("manca il RAW di esempio DSC05634")
    proxy = editing_proxy(raw)
    truth = np.array(lens.apply(proxy.rgb, proxy.lens, True), np.float32)
    h, w = truth.shape[:2]
    params = EditParams.model_validate({"geometry": {"lens_correction": False}})

    def shown(frame):
        decoded = decoded_from_array(frame, temperature_k=6504, tint=0)
        decoded.camera, decoded.baseline_exposure_ev = proxy.camera, proxy.baseline_exposure_ev
        image = render(decoded, params, RenderOptions(stop_before_output=True))
        return _lab(cv2.GaussianBlur(image, (0, 0), 2.0))

    y, x = np.mgrid[0:h, 0:w]
    for cx in (0.40, 0.70):
        cy, r = 0.4905, 0.02  # the edge of the trees' shadow on the ice
        disc = np.hypot(x - cx * w, y - cy * h) < r * w
        photo = truth.copy()
        photo[disc] = [0.02, 0.015, 0.012]
        raster = np.zeros((h, w), np.uint8)
        cv2.circle(raster, (int(cx * w), int(cy * h)), int(r * w * 1.1), 255, -1)
        area = area_alpha(raster, 0.002, 0.25)
        item = EraseItem(id="e", area="0" * 64, engine="ml")
        erase_into(photo, item, fill_ml(photo, area, 0), area)
        after = float(np.mean(delta_e_2000(shown(photo)[disc], shown(truth)[disc])))
        assert after < 4.0, (cx, after)
