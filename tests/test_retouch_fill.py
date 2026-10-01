# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Removals, the eraser (docs/SPEC_rimozione.md section 8): fills, keys, export.

As numbered in the prompt: an erased object looks like the background it
hid (3); the same at 1024 px and at full size (4); the same seed gives the
same bytes, recomputed from nothing, and another seed another fill (5); a
fill goes stale when what is upstream of it changes -- the lens, a removal
before it -- and never for a slider of the development (7); an export
without the fill computes it, to the same image (8), and fails with the
reason when the engine asked for cannot run. The keys also say what order
means (6): an eraser sees the removals before it, not those after.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ape.pipeline.colorspace import delta_e_2000, display_decode, to_lab
from ape.pipeline.filters import resize_long_edge
from ape.pipeline.ops.erase import area_alpha
from ape.pipeline.params import EditParams
from ape.pipeline.render import RenderOptions, render
from ape.pipeline.retouch_params import EraseItem, HealItem
from ape.retouch import fills
from ape.retouch.classic import fill_classic
from conftest import as_decoded, scene_shape

#: The object of every test: a dark disc, a fifth of the frame's height across.
_OBJECT = (0.55, 0.45, 0.06)


def _background(long_edge: int, seed: int = 3) -> np.ndarray:
    """A lit surface with a gradient, a soft stripe and grain, built at 4096 and reduced."""
    base = 4096
    h, w = scene_shape(base)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 0.10 + 0.12 * (x / w) + 0.05 * np.sin(2 * np.pi * y / (0.35 * h))
    grain = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
    scene = light[..., None] * np.array([1.1, 1.0, 0.85], np.float32) * (1.0 + 0.05 * grain)
    scene = np.clip(scene, 1e-4, None).astype(np.float32)
    return scene if long_edge == base else resize_long_edge(scene, long_edge)


def _with_object(scene: np.ndarray) -> np.ndarray:
    h, w = scene.shape[:2]
    cx, cy, r = _OBJECT
    y, x = np.mgrid[0:h, 0:w]
    inside = np.hypot(x - cx * w, y - cy * h) < r * max(h, w)
    out = scene.copy()
    out[inside] = np.array([0.02, 0.015, 0.012], np.float32)
    return out


def _raster(h: int = 1366, w: int = 2048, grow: float = 1.1) -> np.ndarray:
    """The painted area at the brush canvas's size: the object and a little more."""
    cx, cy, r = _OBJECT
    raster = np.zeros((h, w), np.uint8)
    cv2.circle(raster, (int(cx * w), int(cy * h)), int(r * w * grow), 255, -1)
    return raster


def _save_raster(raster: np.ndarray) -> str:
    from ape import masks_store

    ok, encoded = cv2.imencode(".png", raster)
    assert ok
    return masks_store.save(encoded.tobytes())


def _params(*items, **sections) -> EditParams:
    return EditParams.model_validate(
        {"geometry": {"lens_correction": False}, **sections,
         "retouch": [item.model_dump() for item in items]}
    )


def _lab(display: np.ndarray) -> np.ndarray:
    return to_lab(display_decode(display))


_BASE = fills.upstream_base("test:photo", None)


def _filled(proxy: np.ndarray, params: EditParams) -> EditParams:
    keys = fills.compute_fills(as_decoded(proxy), params, _BASE)
    return fills.resolve(params, keys)


# --------------------------------------------------------------------------- #
# 3. Efficacy
# --------------------------------------------------------------------------- #


def test_an_erased_object_looks_like_the_background_it_hid(xdg_home):
    """ΔE2000 against the true background after a 2 px blur (the fill's grain is
    the photo's, not the grain that was under the object, so the comparison is
    of the surface): under 3 -- a difference seen side by side but not across a
    room -- where the object stands at more than 30. The grain comes back too."""
    truth = _background(2048)
    proxy = _with_object(truth)
    name = _save_raster(_raster())
    params = _filled(proxy, _params(EraseItem(id="e", area=name)))
    assert params.retouch[0].fill is not None

    options = RenderOptions(stop_before_output=True)
    erased = render(as_decoded(proxy), params, options)
    before = render(as_decoded(proxy), _params(), options)
    true = render(as_decoded(truth), _params(), options)
    h, w = erased.shape[:2]
    cx, cy, r = _OBJECT
    y, x = np.mgrid[0:h, 0:w]
    area = np.hypot(x - cx * w, y - cy * h) < r * w

    def blurred(image):
        return cv2.GaussianBlur(image, (0, 0), 2.0)

    after = float(np.mean(delta_e_2000(_lab(blurred(erased))[area], _lab(blurred(true))[area])))
    object_ = float(np.mean(delta_e_2000(_lab(blurred(before))[area], _lab(blurred(true))[area])))
    assert object_ > 30.0, object_
    assert after < 3.0, after
    grain = float(np.std((erased - blurred(erased))[area]))
    true_grain = float(np.std((true - blurred(true))[area]))
    assert grain == pytest.approx(true_grain, rel=0.35)


# --------------------------------------------------------------------------- #
# 4. Resolution invariance
# --------------------------------------------------------------------------- #


def test_an_erased_area_looks_the_same_at_every_size(xdg_home):
    """Test 1 of section 13 inside the area: the fill is made once, on the 2048 px
    proxy; the preview renders it at 1024 px, the export at 4096 px reduced."""
    full = _with_object(_background(4096))
    proxy = resize_long_edge(full, 2048)
    name = _save_raster(_raster())
    params = _filled(proxy, _params(EraseItem(id="e", area=name, feather=0.3)))
    options = RenderOptions(stop_before_output=True, long_edge=1024)
    preview = render(as_decoded(resize_long_edge(proxy, 1024)), params, options)
    exported = render(as_decoded(full), params, options)
    h, w = preview.shape[:2]
    cx, cy, r = _OBJECT
    y, x = np.mgrid[0:h, 0:w]
    area = np.hypot(x - cx * w, y - cy * h) < r * w * 1.1
    delta = delta_e_2000(_lab(preview[area]), _lab(exported[area]))
    assert float(np.mean(delta)) < 1.5


# --------------------------------------------------------------------------- #
# 5. Determinism
# --------------------------------------------------------------------------- #


def test_the_same_seed_gives_the_same_fill_and_another_seed_another():
    proxy = _with_object(_background(2048))
    area = area_alpha(_raster(), 0.002, 0.25)
    first = fill_classic(proxy, area, 0)
    again = fill_classic(proxy, area, 0)
    assert first.bbox == again.bbox
    assert first.structure.tobytes() == again.structure.tobytes()
    assert first.offsets.tobytes() == again.offsets.tobytes()
    other = fill_classic(proxy, area, 1)
    assert other.structure.tobytes() != first.structure.tobytes()


def test_a_render_is_the_same_bytes_with_the_fill_made_again(xdg_home, tmp_path):
    from ape.config import get_settings

    proxy = _with_object(_background(2048))
    name = _save_raster(_raster())
    params = _filled(proxy, _params(EraseItem(id="e", area=name)))
    first = render(as_decoded(proxy), params).tobytes()
    for patch in get_settings().retouch_dir.glob("*.patch"):
        patch.unlink()
    from ape import retouch_store

    retouch_store.load.cache_clear()
    remade = _filled(proxy, _params(EraseItem(id="e", area=name)))
    assert remade == params
    assert render(as_decoded(proxy), remade).tobytes() == first


# --------------------------------------------------------------------------- #
# 6. Order and 7. staleness, through the keys
# --------------------------------------------------------------------------- #


_AREA = "ab" * 32


def _keys(*items, **sections) -> dict[str, str]:
    return fills.item_keys(_params(*items, **sections), _BASE)


def test_an_eraser_sees_the_removals_before_it_and_not_those_after():
    erase = EraseItem(id="e", area=_AREA)
    heal = HealItem(id="h", cx=0.2, cy=0.2, sx=0.25, sy=0.2)
    moved = heal.model_copy(update={"cx": 0.3})
    alone = _keys(erase)["e"]
    assert _keys(heal, erase)["e"] != alone
    assert _keys(moved, erase)["e"] != _keys(heal, erase)["e"]
    assert _keys(erase, heal)["e"] == alone
    assert _keys(heal.model_copy(update={"visible": False}), erase)["e"] == alone


def test_the_development_never_makes_a_fill_stale_and_the_lens_does():
    erase = EraseItem(id="e", area=_AREA)
    key = _keys(erase)["e"]
    for sections in (
        {"white_balance": {"mode": "custom", "temperature_k": 3200.0}},
        {"exposure": {"ev": 1.3}},
        {"masks": [{"kind": "radial", "definition": {}, "exposure": {"ev": 0.5}}]},
        {"tone": {"contrast": 1.8}},
    ):
        assert _keys(erase, **sections)["e"] == key, sections
    assert _keys(erase.model_copy(update={"opacity": 0.5, "visible": False}))["e"] == key
    other_lens = fills.upstream_base("test:photo", ["lens", "another profile"])
    assert fills.item_keys(_params(erase), other_lens)["e"] != key
    for change in ({"expand": 0.01}, {"seed": 4}, {"engine": "ml"}, {"feather": 0.6}):
        assert _keys(erase.model_copy(update=change))["e"] != key, change


def test_an_old_fill_shows_until_the_new_one_arrives(xdg_home):
    proxy = _with_object(_background(2048))
    name = _save_raster(_raster())
    first = _filled(proxy, _params(EraseItem(id="e", area=name)))
    old_fill = first.retouch[0].fill
    grown = first.retouch[0].model_copy(update={"expand": 0.01, "fill": None})
    edited = _params(grown)
    keys = fills.item_keys(edited, _BASE)
    assert fills.missing(keys)
    # The gesture's new parameters name no fill; the photo's version does.
    resolved = fills.resolve(edited, keys, previous=first)
    assert resolved.retouch[0].fill == old_fill


# --------------------------------------------------------------------------- #
# 8. Export
# --------------------------------------------------------------------------- #


def _work(params: EditParams, tmp_path):
    from ape.export.run import ExportWork
    from ape.export.settings import ExportSettings

    return ExportWork(
        source=tmp_path / "photo.ARW", filename="photo.ARW", name="photo.jpg",
        params=params, settings=ExportSettings(), policy=None, photo_token="test:photo",
    )


def test_an_export_without_the_fill_makes_it_and_gives_the_same_image(
    xdg_home, tmp_path, monkeypatch
):
    from ape.export import run
    from ape.raw import proxy as proxy_module

    full = _with_object(_background(4096))
    proxy = resize_long_edge(full, 2048)
    monkeypatch.setattr(proxy_module, "editing_proxy", lambda *a, **k: as_decoded(proxy))
    name = _save_raster(_raster())
    params = _params(EraseItem(id="e", area=name))
    timings: dict[str, float] = {}
    exported = run._with_fills(_work(params, tmp_path), as_decoded(full), timings)
    assert "retouch" in timings, "calcolata dentro l'export"
    ready = _filled(proxy, params)
    assert exported == ready
    options = RenderOptions(long_edge=1024)
    assert np.array_equal(render(as_decoded(full), exported, options),
                          render(as_decoded(full), ready, options))


def test_an_export_asking_for_a_missing_model_fails_with_the_reason(
    xdg_home, tmp_path, monkeypatch
):
    from ape.export import run
    from ape.raw import proxy as proxy_module

    proxy = _with_object(_background(2048))
    monkeypatch.setattr(proxy_module, "editing_proxy", lambda *a, **k: as_decoded(proxy))
    params = _params(EraseItem(id="e", area=_save_raster(_raster()), engine="ml"))
    with pytest.raises(run.ExportFailure) as caught:
        run._with_fills(_work(params, tmp_path), as_decoded(proxy), {})
    assert caught.value.photo_fault
    assert "rimozione" in str(caught.value)


def test_the_stage_cache_gives_the_same_pixels_with_an_eraser(xdg_home):
    """``StageRenderer`` against a fresh render, erasers included and moved in the list."""
    from ape.pipeline.render import StageRenderer

    proxy = _with_object(_background(1024))
    name = _save_raster(_raster(683, 1024))
    heal = HealItem(id="h", cx=0.2, cy=0.3, radius=0.02, sx=0.26, sy=0.3)
    erase = EraseItem(id="e", area=name)
    ready = _filled(proxy, _params(heal, erase))
    decoded = as_decoded(proxy)
    renderer = StageRenderer(decoded)
    swapped = _filled(proxy, _params(erase, heal))
    brighter = EditParams.model_validate({**ready.model_dump(), "exposure": {"ev": 0.4}})
    for params in (ready, brighter, swapped, _params()):
        assert np.array_equal(renderer.render(params), render(decoded, params))
