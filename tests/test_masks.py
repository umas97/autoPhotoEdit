# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Masks (section 6.3, phase 9): the selections, and what the render does with them.

The claims, in order: each shape covers what it says; combinations, invert and
opacity compose as documented; painted rasters are stored safely and by
content; a mask that adjusts nothing changes no bit; outside a selection the
render is the global render to the bit; the banded full-resolution path and the
stage cache give the same pixels as a plain render; and the same masks look the
same at 1024 px and at 2048 px (test 1 of section 13, extended to masks).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ape.pipeline import bands
from ape.pipeline.colorspace import delta_e_2000, display_decode, to_lab
from ape.pipeline.filters import resize_long_edge
from ape.pipeline.ops import masks as mask_ops
from ape.pipeline.params import EditParams, MaskParams, ToneParams
from ape.pipeline.render import RenderOptions, StageRenderer, render, render_stages
from conftest import as_decoded, build_scene


def _selection(img: np.ndarray, **mask) -> np.ndarray:
    return mask_ops.evaluate(
        img, MaskParams(**mask), ToneParams(), lambda name: pytest.fail(name)
    ).astype(np.float32)


def _flat(height: int = 60, width: int = 90, value: float = 0.18) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #


def test_linear_gradient_is_full_before_its_start_and_empty_past_its_end():
    m = _selection(_flat(), kind="linear", definition={"x0": 0.5, "y0": 0.2, "x1": 0.5, "y1": 0.6})
    column = m[:, 45]
    assert column[:10].min() == pytest.approx(1.0)  # above y0 = 0.2 (row 12)
    assert column[40:].max() == pytest.approx(0.0)  # below y1 = 0.6 (row 36)
    assert column[24] == pytest.approx(0.5, abs=0.05)  # the midpoint
    assert np.all(np.diff(column) <= 1e-6), "monotona"
    assert np.allclose(m, m[:, :1]), "uguale su ogni riga"


def test_radial_is_an_ellipse_in_long_edge_units_and_turns_counter_clockwise():
    img = _flat(100, 200)
    wide = {"cx": 0.5, "cy": 0.5, "rx": 0.2, "ry": 0.05, "feather": 0.0}
    m = _selection(img, kind="radial", definition=wide)
    # rx = 0.2 of the 200 px long edge = 40 px across; ry = 10 px down.
    assert m[50, 100 + 35] > 0.99 and m[50, 100 + 45] < 0.01
    assert m[50 + 7, 100] > 0.99 and m[50 + 13, 100] < 0.01
    turned = _selection(img, kind="radial", definition={**wide, "angle": 90.0})
    assert turned[50 + 35, 100] > 0.99 and turned[50, 100 + 35] < 0.01


def test_invert_opacity_and_combinations():
    img = _flat()
    left = {"x0": 0.4, "y0": 0.5, "x1": 0.6, "y1": 0.5}  # 1 on the left
    base = _selection(img, kind="linear", definition=left)
    assert np.allclose(
        _selection(img, kind="linear", definition=left, invert=True, opacity=0.5),
        0.5 * (1 - base),
        atol=1e-3,
    )
    top = {"kind": "linear", "definition": {"x0": 0.5, "y0": 0.4, "x1": 0.5, "y1": 0.6}}
    other = _selection(img, **top)
    for op, expected in (
        ("intersect", base * other),
        ("add", np.maximum(base, other)),
        ("subtract", base * (1 - other)),
    ):
        combined = _selection(
            img, kind="linear", definition={**left, "combine": [{"op": op, **top}]}
        )
        assert np.allclose(combined, expected, atol=2e-3), op


def test_luminance_range_selects_by_how_bright_the_pixel_will_look():
    img = _flat(40, 120)
    img[:, 60:] = 4.0  # a highlight, two stops over grey... and then some
    m = _selection(
        img, kind="parametric", definition={"luminance": {"low": 0.7, "high": 1.0}}
    )
    assert m[:, :50].max() < 0.01 and m[:, 70:].min() > 0.99


def test_hue_range_selects_blue_and_leaves_greys_out():
    img = _flat(40, 120, 0.3)
    img[:, :40] = [0.05, 0.12, 0.45]  # sky blue, linear Rec.2020
    img[:, 40:80] = [0.45, 0.15, 0.05]  # orange
    m = _selection(
        img, kind="parametric", definition={"hue": {"center": 220, "width": 60, "feather": 20}}
    )
    assert m[:, 5:35].min() > 0.95
    assert m[:, 45:75].max() < 0.01, "arancio"
    assert m[:, 85:].max() < 0.01, "un grigio non ha tinta"


# --------------------------------------------------------------------------- #
# Painted rasters
# --------------------------------------------------------------------------- #


def test_rasters_are_stored_by_content_as_one_channel(xdg_home):
    from ape import masks_store

    rgba = np.zeros((30, 40, 4), dtype=np.uint8)
    rgba[:, :20, 3] = 255  # the browser paints alpha on a transparent canvas
    ok, png = cv2.imencode(".png", rgba)
    name = masks_store.save(png.tobytes())
    assert masks_store.path_for(name).parent == xdg_home / "data" / "autophotoedit" / "masks"
    assert masks_store.save(png.tobytes()) == name, "stesso contenuto, stesso file"
    raster = masks_store.load(name)
    assert raster.shape == (30, 40) and raster[:, :20].min() == 255 and raster[:, 20:].max() == 0
    assert not list(masks_store.path_for(name).parent.glob(".*part")), "nessun file parziale"


def test_bad_rasters_are_refused(xdg_home):
    from ape import masks_store

    with pytest.raises(masks_store.RasterError, match="PNG"):
        masks_store.save(b"GIF89a non una maschera")
    ok, huge = cv2.imencode(".png", np.zeros((10, masks_store.MAX_EDGE + 1), np.uint8))
    with pytest.raises(masks_store.RasterError, match="lato lungo"):
        masks_store.save(huge.tobytes())
    with pytest.raises(masks_store.RasterError, match="non valido"):
        masks_store.path_for("../../etc/passwd")
    with pytest.raises(masks_store.RasterError, match="non è più"):
        masks_store.load("0" * 64)


def test_a_painted_mask_renders_where_it_was_painted(xdg_home):
    from ape import masks_store

    painted = np.zeros((20, 30), dtype=np.uint8)
    painted[:, :15] = 255
    ok, png = cv2.imencode(".png", painted)
    name = masks_store.save(png.tobytes())
    params = EditParams.model_validate(
        {"masks": [{"kind": "brush", "definition": {"raster": name}, "exposure": {"ev": 1.0}}]}
    )
    decoded = as_decoded(_flat(40, 60))
    out = render(decoded, params, RenderOptions(stop_before_output=True))
    plain = render(decoded, EditParams(), RenderOptions(stop_before_output=True))
    assert out[:, :25].mean() > plain[:, :25].mean() + 0.05
    assert np.array_equal(out[:, 35:], plain[:, 35:]), "fuori dal pennello, identica"


# --------------------------------------------------------------------------- #
# The render
# --------------------------------------------------------------------------- #

_RADIAL = {"cx": 0.3, "cy": 0.4, "rx": 0.15, "ry": 0.1, "angle": 20.0, "feather": 0.4}


def _masked(**adjust) -> EditParams:
    return EditParams.model_validate(
        {
            "tone_shaping": {"shadows": 0.2},
            "color": {"vibrance": 0.2},
            "masks": [{"kind": "radial", "definition": _RADIAL, **adjust}],
        }
    )


def _everything() -> dict:
    return {
        "exposure": {"ev": 0.7},
        "tone_shaping": {"highlights": -0.5, "shadows": 0.4},
        "color": {"saturation": -0.4, "hsl": {"blue": {"luminance": -0.3}}},
        "local_contrast": {"clarity": 0.5},
    }


def test_a_mask_that_adjusts_nothing_changes_no_bit():
    decoded = as_decoded(build_scene(2, 512))
    plain = EditParams.model_validate(
        {"tone_shaping": {"shadows": 0.2}, "color": {"vibrance": 0.2}}
    )
    assert np.array_equal(render(decoded, _masked(), None), render(decoded, plain, None))


def test_outside_the_selection_the_render_is_the_global_one_to_the_bit():
    decoded = as_decoded(build_scene(3, 512))
    options = RenderOptions(stop_before_output=True)
    plain = EditParams.model_validate(
        {"tone_shaping": {"shadows": 0.2}, "color": {"vibrance": 0.2}}
    )
    out = render(decoded, _masked(**_everything()), options)
    base = render(decoded, plain, options)
    # Far from the ellipse, bar the reach of the clarity's guided filter.
    assert np.array_equal(out[:, 300:], base[:, 300:])
    inside = (slice(int(0.4 * out.shape[0]) - 5, int(0.4 * out.shape[0]) + 5), slice(148, 158))
    assert not np.allclose(out[inside], base[inside], atol=1e-3)


def test_banded_full_resolution_matches_the_plain_chain(monkeypatch):
    monkeypatch.setattr(bands, "BAND_MIN_PIXELS", 1000)
    monkeypatch.setattr(bands, "BAND_PIXELS", 16 * 512)  # many bands on a small frame
    decoded = as_decoded(build_scene(4, 512))
    params = _masked(**_everything())
    banded = render(decoded, params)
    plain = render_stages(decoded, params)["output"]
    assert np.array_equal(banded, plain)


def test_the_stage_cache_gives_the_same_pixels_as_a_fresh_render():
    decoded = as_decoded(build_scene(5, 512))
    renderer = StageRenderer(decoded=decoded)
    first = _masked(**_everything())
    renderer.render(first)
    # A later-stage change on the mask: resumes past the masks stage.
    second = first.model_copy(deep=True)
    second.masks[0].color.saturation = 0.6
    assert np.array_equal(renderer.render(second), render(decoded, second))
    # Reordered masks must not swap selections in the cache.
    third = second.model_copy(deep=True)
    third.masks = [
        MaskParams(kind="linear", definition={"x0": 0.5, "y0": 0.0, "x1": 0.5, "y1": 0.5},
                   exposure={"ev": -1.0}),
        third.masks[0],
    ]
    renderer.render(third)
    swapped = third.model_copy(deep=True)
    swapped.masks = swapped.masks[::-1]
    assert np.array_equal(renderer.render(swapped), render(decoded, swapped))


def test_through_a_stage_is_that_stage_of_the_chain():
    decoded = as_decoded(build_scene(6, 512))
    params = _masked(**_everything())
    renderer = StageRenderer(decoded=decoded)
    stages = render_stages(decoded, params)
    assert np.array_equal(renderer.through(params, "exposure"), stages["exposure"])
    renderer.render(params)  # the checkpoint is now the cached one
    assert np.array_equal(renderer.through(params, "exposure"), stages["exposure"])
    assert np.array_equal(renderer.render(params), stages["output"])


def test_a_masks_local_contrast_costs_no_rgb_frame():
    """The masks' clarity is blended a band at a time, from one-channel maps.

    Made as a second full-frame operation, it held the global result, the local
    one and their intermediates at once: 4.75 frames for two masks, and a 24 MP
    export over the memory of section 26 (2.07 GB against 1.48 without masks).
    """
    import tracemalloc

    from ape.pipeline import local

    img = np.random.default_rng(0).random((1000, 1500, 3), dtype=np.float32)
    masks = [
        MaskParams(kind="radial", definition={}, local_contrast={"clarity": 0.4, "shadows": 0.3})
    ] * 2
    selections = {
        i: mask_ops.evaluate(img, mask, ToneParams(), pytest.fail) for i, mask in enumerate(masks)
    }
    tracemalloc.start()
    try:
        local.blend_local_contrast(img, masks, selections, owned=True)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2.5 * img.nbytes


def test_a_missing_segmentation_is_an_error_not_an_absent_mask():
    params = EditParams.model_validate(
        {"masks": [{"kind": "segment", "definition": {"subject": "sky"}, "exposure": {"ev": -1}}]}
    )
    with pytest.raises(ValueError, match="non è ancora stata calcolata"):
        render(as_decoded(_flat()), params)


@pytest.mark.parametrize("scene_index", [0, 6, 9])
def test_masks_look_the_same_at_every_size(scene_index):
    """Test 1 of section 13 with every kind of mask computed from the frame."""
    scene = build_scene(scene_index, 2048)
    params = EditParams.model_validate(
        {
            "masks": [
                {"kind": "radial", "definition": _RADIAL, **_everything()},
                {
                    "kind": "linear",
                    "definition": {"x0": 0.5, "y0": 0.0, "x1": 0.6, "y1": 0.5,
                                   "combine": [{"op": "intersect", "kind": "parametric",
                                                "definition": {"luminance": {"low": 0.4}}}]},
                    "exposure": {"ev": -0.8},
                    "color": {"vibrance": 0.5},
                },
                {
                    "kind": "parametric",
                    "definition": {"hue": {"center": 200, "width": 80}},
                    "tone_shaping": {"highlights": -0.6},
                },
            ]
        }
    )
    options = RenderOptions(stop_before_output=True, long_edge=1024)
    preview = render(as_decoded(resize_long_edge(scene, 1024)), params, options)
    full = render(as_decoded(scene), params, options)
    height, width = (min(a, b) - 24 for a, b in zip(preview.shape[:2], full.shape[:2], strict=True))
    sampled = (slice(24, height, 2), slice(24, width, 2))
    delta = delta_e_2000(
        to_lab(display_decode(preview[sampled])), to_lab(display_decode(full[sampled]))
    )
    assert float(np.mean(delta)) < 1.5
