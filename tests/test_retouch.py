# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Removals, the pipeline half (docs/SPEC_rimozione.md section 8): the spot healing.

The claims, as numbered in the prompt: no removal renders to the bit as
before (1); outside a spot nothing changes (2); a healed spot looks like what
was under it (3); it looks the same at 1024 px and at full size (4); the same
parameters give the same bytes (5); the removals apply in order, each on the
result of the ones before (6); and the stage cache gives what a fresh render
gives. The automatic source: never on the spot, same surface, "another"
really is another, and fast enough to run on a click. The eraser's fills are
in ``test_retouch_fill.py``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ape.pipeline.colorspace import delta_e_2000, display_decode, to_lab
from ape.pipeline.filters import resize_long_edge
from ape.pipeline.ops.heal import RING, find_source, heal_into
from ape.pipeline.params import EditParams, NoiseParams
from ape.pipeline.render import RenderOptions, StageRenderer, render, render_stages, resume
from ape.pipeline.retouch_params import HealItem
from conftest import as_decoded, build_scene, scene_shape

#: The spot of every test: a dark speck on a surface with grain, the sensor
#: dust or the blemish the tool is for.
_SPOT = {"cx": 0.4, "cy": 0.5, "radius": 0.012}


def _surface(long_edge: int, *, blemish: bool = True, seed: int = 7) -> np.ndarray:
    """A lit surface with fine grain, and optionally a dark blemish at ``_SPOT``.

    Built at 2048 px and reduced, so that every size is the same picture: the
    grain of the smaller one is the larger one's, averaged -- as a proxy's is.
    """
    base = 2048
    h, w = scene_shape(base)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 0.12 + 0.10 * (x / w) + 0.04 * (y / h)
    tint = np.array([1.25, 1.0, 0.8], dtype=np.float32)
    grain = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
    scene = light[..., None] * tint * (1.0 + 0.06 * grain)
    if blemish:
        # Sized as the user sizes the circle: the blemish fades out inside it.
        cx, cy, r = _SPOT["cx"] * w, _SPOT["cy"] * h, _SPOT["radius"] * base * 0.3
        dark = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))[..., None]
        scene = scene * (1.0 - 0.85 * dark)
    scene = np.ascontiguousarray(np.clip(scene, 1e-4, None), dtype=np.float32)
    return scene if long_edge == base else resize_long_edge(scene, long_edge)


def _heal(**overrides) -> HealItem:
    values = {"id": "h1", **_SPOT, "sx": 0.44, "sy": 0.5, **overrides}
    return HealItem(**values)


def _params(*items, **sections) -> EditParams:
    return EditParams.model_validate(
        {**sections, "retouch": [item.model_dump() for item in items]}
    )


def _lab(display: np.ndarray) -> np.ndarray:
    return to_lab(display_decode(display))


# --------------------------------------------------------------------------- #
# 1. Neutrality
# --------------------------------------------------------------------------- #


def test_no_removal_renders_to_the_bit_as_the_chain_without_the_stage():
    decoded = as_decoded(build_scene(6, 512))
    params = EditParams()
    stages = render_stages(decoded, params)
    assert stages["retouch"] is stages["lens"], "nessuna copia senza rimozioni"
    skipped = resume(stages["lens"], decoded, params, after="retouch")
    assert np.array_equal(render(decoded, params), skipped)


def test_hidden_or_transparent_removals_change_no_bit():
    decoded = as_decoded(_surface(512))
    plain = render(decoded, EditParams())
    hidden = _params(_heal(visible=False), _heal(id="h2", opacity=0.0))
    assert np.array_equal(render(decoded, hidden), plain)


def test_the_show_removals_switch_suspends_them():
    decoded = as_decoded(_surface(512))
    params = _params(_heal())
    off = render(decoded, params, RenderOptions(retouch=False))
    assert np.array_equal(off, render(decoded, EditParams()))
    assert not np.array_equal(render(decoded, params), off)


# --------------------------------------------------------------------------- #
# 2. Locality
# --------------------------------------------------------------------------- #


def test_outside_the_spot_nothing_changes():
    """At the stage's output, to the bit; and at the render's too, where no
    spatial filter after the stage reaches back into the spot."""
    decoded = as_decoded(_surface(1024))
    item = _heal()
    stages = render_stages(decoded, _params(item))
    h, w = stages["lens"].shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    rho = np.hypot(x + 0.5 - item.cx * w, y + 0.5 - item.cy * h) / (item.radius * max(h, w))
    outside = rho >= 1.0
    assert np.array_equal(stages["retouch"][outside], stages["lens"][outside])
    assert not np.array_equal(stages["retouch"][~outside], stages["lens"][~outside])

    quiet = {"noise": NoiseParams(chrominance=0.0).model_dump()}
    with_spot = render(decoded, _params(item, **quiet))
    without = render(decoded, _params(**quiet))
    assert np.array_equal(with_spot[outside], without[outside])


# --------------------------------------------------------------------------- #
# 3. Efficacy
# --------------------------------------------------------------------------- #


def test_a_healed_blemish_looks_like_the_surface_under_it():
    """ΔE2000 against the surface without the blemish, after a blur of a
    quarter radius: the colour and the light must be the surface's (under 2,
    the threshold at which two flat patches side by side start to read as
    different), while the blemish itself stands at more than 10. The grain is
    checked apart: the copy keeps it (within 25% of the true one) instead of
    smearing it, which is what a colour match alone would not see."""
    import cv2

    long_edge = 2048
    blemished = _surface(long_edge)
    truth = _surface(long_edge, blemish=False)
    sx, sy = find_source(blemished, _SPOT["cx"], _SPOT["cy"], _SPOT["radius"])
    item = _heal(sx=sx, sy=sy)
    options = RenderOptions(stop_before_output=True)
    healed = render(as_decoded(blemished), _params(item), options)
    before = render(as_decoded(blemished), EditParams(), options)
    true = render(as_decoded(truth), EditParams(), options)

    h, w = healed.shape[:2]
    r = item.radius * long_edge
    y, x = np.mgrid[0:h, 0:w]
    disc = np.hypot(x + 0.5 - item.cx * w, y + 0.5 - item.cy * h) < r * 0.6
    sigma = r / 4.0

    def blurred(image):
        return cv2.GaussianBlur(image, (0, 0), sigma)

    after = float(np.mean(delta_e_2000(_lab(blurred(healed))[disc], _lab(blurred(true))[disc])))
    blemish = float(np.mean(delta_e_2000(_lab(blurred(before))[disc], _lab(blurred(true))[disc])))
    assert blemish > 10.0, blemish
    assert after < 2.0, after
    grain_healed = float(np.std((healed - blurred(healed))[disc]))
    grain_true = float(np.std((true - blurred(true))[disc]))
    assert grain_healed == pytest.approx(grain_true, rel=0.25)


# --------------------------------------------------------------------------- #
# 4. Resolution invariance
# --------------------------------------------------------------------------- #


def test_a_spot_looks_the_same_at_every_size():
    """Test 1 of section 13, inside the spot: 1024 px against 2048 px reduced."""
    full_scene = _surface(2048)
    item = _heal(sx=0.445, sy=0.49, feather=0.3)
    options = RenderOptions(stop_before_output=True, long_edge=1024)
    preview = render(as_decoded(resize_long_edge(full_scene, 1024)), _params(item), options)
    full = render(as_decoded(full_scene), _params(item), options)
    h, w = preview.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    rho = np.hypot(x + 0.5 - item.cx * w, y + 0.5 - item.cy * h) / (item.radius * 1024)
    spot = rho < 1.0
    delta = delta_e_2000(_lab(preview[spot]), _lab(full[spot]))
    assert float(np.mean(delta)) < 1.5


# --------------------------------------------------------------------------- #
# 5. Determinism, 6. order, and the stage cache
# --------------------------------------------------------------------------- #


def test_the_same_parameters_give_the_same_bytes():
    decoded = as_decoded(_surface(1024))
    params = _params(_heal(), _heal(id="h2", cx=0.6, sx=0.63))
    assert render(decoded, params).tobytes() == render(decoded, params).tobytes()


def test_removals_apply_in_order_each_on_the_result_of_the_one_before():
    """B copies from where the blemish was. After A has healed it, B copies
    clean surface; before A, B copies the blemish."""
    scene = _surface(1024)
    heal_a = _heal(id="a", sx=0.44, sy=0.5)
    heal_b = _heal(id="b", cx=0.36, cy=0.5, sx=0.4, sy=0.5)
    decoded = as_decoded(scene)
    first_a = render_stages(decoded, _params(heal_a, heal_b))["retouch"]
    first_b = render_stages(decoded, _params(heal_b, heal_a))["retouch"]
    assert not np.array_equal(first_a, first_b)

    h, w = scene.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    near_b = np.hypot(x + 0.5 - heal_b.cx * w, y + 0.5 - heal_b.cy * h) < heal_b.radius * w * 0.4
    clean = float(first_a[near_b].mean())
    dirty = float(first_b[near_b].mean())
    assert dirty < clean * 0.8, (dirty, clean)


def test_the_stage_cache_gives_the_same_pixels_as_a_fresh_render():
    decoded = as_decoded(_surface(512))
    renderer = StageRenderer(decoded)
    sequence = [
        _params(_heal()),
        _params(_heal(radius=0.02)),
        _params(_heal(radius=0.02), exposure={"ev": 0.5}),
        _params(_heal(radius=0.02), _heal(id="h2", cx=0.7, sx=0.73), exposure={"ev": 0.5}),
        _params(exposure={"ev": 0.5}),
    ]
    for params in sequence:
        assert np.array_equal(renderer.render(params), render(decoded, params))


def test_a_spot_at_the_edge_of_the_frame_stays_inside_it():
    frame = _surface(512)
    item = _heal(cx=0.995, cy=0.01, sx=0.96, sy=0.03, radius=0.02)
    out = frame.copy()
    heal_into(out, item)
    assert np.isfinite(out).all()
    assert out.shape == frame.shape


# --------------------------------------------------------------------------- #
# The automatic source
# --------------------------------------------------------------------------- #


def test_the_source_is_never_on_the_spot_and_stays_near():
    frame = _surface(2048)
    sx, sy = find_source(frame, _SPOT["cx"], _SPOT["cy"], _SPOT["radius"])
    h, w = frame.shape[:2]
    distance = np.hypot((sx - _SPOT["cx"]) * w, (sy - _SPOT["cy"]) * h) / (_SPOT["radius"] * w)
    assert 2.0 * (1.0 + RING) - 0.1 <= distance <= 8.0


def test_the_source_stays_on_the_same_surface():
    """Left of a vertical edge the surface is dark, right of it bright: a spot
    just left of the edge takes its source on the left."""
    frame = _surface(1024, blemish=False)
    h, w = frame.shape[:2]
    frame[:, int(0.43 * w) :] *= 3.0
    sx, _sy = find_source(frame, 0.4, 0.5, 0.008)
    assert sx < 0.43 - 0.008 * 1.3


def test_another_source_is_another():
    frame = _surface(2048)
    first = find_source(frame, _SPOT["cx"], _SPOT["cy"], _SPOT["radius"])
    second = find_source(frame, _SPOT["cx"], _SPOT["cy"], _SPOT["radius"], avoid=[first])
    h, w = frame.shape[:2]
    apart = np.hypot((first[0] - second[0]) * w, (first[1] - second[1]) * h)
    assert apart >= _SPOT["radius"] * w * 0.99


@pytest.mark.parametrize("radius", [0.003, 0.012, 0.08])
def test_the_source_is_found_in_time_on_the_proxy(radius):
    """Under 300 ms on the 2048 px proxy (docs/SPEC_rimozione.md section 9)."""
    frame = _surface(2048)
    find_source(frame, 0.5, 0.5, radius)  # warm the OpenCV kernels
    start = time.perf_counter()
    find_source(frame, 0.5, 0.5, radius)
    assert time.perf_counter() - start < 0.3
