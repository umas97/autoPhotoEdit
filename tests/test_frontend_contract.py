# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The two places where the interface repeats something the backend owns.

Duplication between a Python schema and a TypeScript one is unavoidable -- the
browser cannot import pydantic -- but silent divergence is not. Two facts are
checked here against the real source of truth:

* **every slider's range is the range of its field.** ``EditParams`` is the one
  contract between subsystems (section 5), and the server answers 400 to a value
  outside it. A slider that can reach such a value is a control that breaks under
  the user's hand, and the failure would look like a bug in the renderer;
* **every string the interface shows exists in the table.** Section 4 forbids
  hardcoded strings in components; the corresponding risk is a ``t('key')`` for a
  key nobody ever wrote, which shows the key itself on screen.

Both read the TypeScript as text. A parser would be better and is not worth it:
the two files these tests read are written in a fixed shape, and a test that
starts failing because the shape changed is a test doing its job.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ape.pipeline.params import EditParams

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
PARAMS_TS = FRONTEND / "src" / "lib" / "params.ts"
CONTROLS_TS = FRONTEND / "src" / "lib" / "controls.ts"
STRINGS_TS = FRONTEND / "src" / "i18n" / "it.ts"

pytestmark = pytest.mark.skipif(not PARAMS_TS.is_file(), reason="frontend assente")


def _controls(source: Path = CONTROLS_TS) -> list[dict[str, object]]:
    """Every control declared in a table of ``source``, as dictionaries.

    The files declare them in two shapes -- object literals and the ``unit()``
    helper for the -1..1 controls -- so both are read.
    """
    text = source.read_text(encoding="utf-8")
    controls: list[dict[str, object]] = []

    for block in re.finditer(
        r"\{\s*path:\s*'([^']+)',.*?min:\s*(-?[\d.]+),\s*max:\s*(-?[\d.]+),"
        r"\s*step:\s*(-?[\d.]+),\s*neutral:\s*(-?[\d.]+)",
        text,
        re.DOTALL,
    ):
        path, minimum, maximum, step, neutral = block.groups()
        controls.append(
            {
                "path": path,
                "min": float(minimum),
                "max": float(maximum),
                "step": float(step),
                "neutral": float(neutral),
            }
        )

    for block in re.finditer(r"unit\('([^']+)',\s*'[^']+'(?:,\s*(-?[\d.]+))?\)", text):
        path, neutral = block.groups()
        controls.append(
            {
                "path": path,
                "min": -1.0,
                "max": 1.0,
                "step": 0.01,
                "neutral": float(neutral or 0.0),
            }
        )
    return controls


def _field_bounds(
    path: str, root: object = EditParams
) -> tuple[float | None, float | None, object]:
    """The declared range of a dotted path into ``root``, and its default."""
    model: object = root
    field = None
    for name in path.split("."):
        fields = model.model_fields  # type: ignore[union-attr]
        assert name in fields, f"campo inesistente in {root.__name__}: {path}"
        field = fields[name]
        model = field.annotation

    low = high = None
    for item in getattr(field, "metadata", []):
        low = getattr(item, "ge", None) if getattr(item, "ge", None) is not None else low
        high = getattr(item, "le", None) if getattr(item, "le", None) is not None else high
        if getattr(item, "gt", None) is not None:
            low = item.gt
        if getattr(item, "lt", None) is not None:
            high = item.lt
    return low, high, field.default


def test_every_slider_exists_in_edit_params():
    controls = _controls()
    assert len(controls) > 25, "la tabella dei controlli sembra troncata"
    for control in controls:
        _field_bounds(str(control["path"]))


def test_no_slider_can_reach_a_value_the_server_refuses():
    for control in _controls():
        path = str(control["path"])
        low, high, _ = _field_bounds(path)
        if low is not None:
            assert control["min"] >= low, f"{path}: minimo {control['min']} sotto {low}"
        if high is not None:
            assert control["max"] <= high, f"{path}: massimo {control['max']} sopra {high}"


def test_the_neutral_position_is_the_default_of_the_field():
    """Double-clicking a slider must land exactly on ``neutral_params()``."""
    for control in _controls():
        path = str(control["path"])
        _, _, default = _field_bounds(path)
        if isinstance(default, int | float):
            assert control["neutral"] == pytest.approx(float(default)), path


def test_the_typescript_neutral_document_matches_neutral_params():
    """``neutralParams()`` in the browser is ``neutral_params()`` on the server."""
    text = PARAMS_TS.read_text(encoding="utf-8")
    start = text.index("export function neutralParams()")
    body = text[text.index("return {", start) : text.index("\n}", start)]

    # Pull the numbers out of the literal and compare them field by field, which
    # is cheaper than teaching a test to parse TypeScript.
    server = json.loads(EditParams().model_dump_json())
    for group, values in server.items():
        if not isinstance(values, dict):
            continue
        for key, expected in values.items():
            if not isinstance(expected, int | float) or isinstance(expected, bool):
                continue
            found = re.search(rf"{group}:\s*\{{[^}}]*?{key}:\s*(-?[\d.]+)", body, re.DOTALL)
            if found:
                assert float(found.group(1)) == pytest.approx(float(expected)), f"{group}.{key}"


def test_every_string_the_interface_asks_for_is_in_the_table():
    """Section 4: no visible string outside the Italian table.

    The table is ``i18n/it.ts`` plus the sections it spreads in from
    ``i18n/it.*.ts`` -- one table, several files, so none grows past 400 lines.
    """
    tables = sorted(STRINGS_TS.parent.glob("it*.ts"))
    table = {
        key
        for path in tables
        for key in re.findall(r"^  '([^']+)':", path.read_text(encoding="utf-8"), re.M)
    }
    assert len(table) > 100, "la tabella delle stringhe sembra troncata"
    spread = STRINGS_TS.read_text(encoding="utf-8")
    for extra in tables:
        if extra != STRINGS_TS:
            name = extra.stem.split(".", 1)[1]
            assert f"...{name}," in spread, f"{extra.name} non è incluso nella tabella"

    missing: dict[str, str] = {}
    for source in (FRONTEND / "src").rglob("*.ts*"):
        if source in tables:
            continue
        for key in re.findall(r"\bt\(\s*'([^']+)'", source.read_text(encoding="utf-8")):
            if key not in table:
                missing[key] = source.name
    assert not missing, f"chiavi usate e non tradotte: {missing}"


def test_the_panel_covers_the_parameters_a_person_edits():
    """Every group of section 6.2 that phase 3 supports is reachable in the UI."""
    paths = {str(control["path"]).split(".")[0] for control in _controls()}
    expected = {
        "white_balance",
        "exposure",
        "highlight_recovery",
        "noise",
        "tone",
        "tone_shaping",
        "tone_curve",
        "color",
        "local_contrast",
        "sharpen",
    }
    assert expected <= paths, f"gruppi assenti dal pannello: {sorted(expected - paths)}"


CULLING_PANEL = FRONTEND / "src" / "components" / "culling" / "CullingPanel.tsx"


def _string_table() -> set[str]:
    """Every key of ``i18n/it.ts`` and of the sections it spreads in."""
    return {
        key
        for path in STRINGS_TS.parent.glob("it*.ts")
        for key in re.findall(r"^  '([^']+)':", path.read_text(encoding="utf-8"), re.M)
    }


def test_the_culling_weights_return_where_the_server_starts():
    """Double-clicking a weight lands on ``select.DEFAULT_WEIGHTS``."""
    from ape.culling.select import DEFAULT_WEIGHTS

    text = CULLING_PANEL.read_text(encoding="utf-8")
    block = text[text.index("const DEFAULT_WEIGHTS"):]
    for name, value in DEFAULT_WEIGHTS.items():
        found = re.search(rf"\b{name}:\s*([\d.]+)", block)
        assert found, f"peso {name} assente dal pannello"
        assert float(found.group(1)) == pytest.approx(value), name


def test_every_code_the_server_can_send_has_words():
    """Reasons, criteria, modes, states and model problems are codes on the
    wire and sentences on screen. The keys are built from the codes at run
    time, so the literal check above cannot see them: this one enumerates the
    codes from the server side and looks for each key."""
    from ape.culling.select import CRITERIA, Reason
    from ape.db.enums import CullingMode, MergeDecision, MergeKind, PhotoStatus
    from ape.models_registry import MODELS, Unavailable

    table = _string_table()
    expected = {f"culling.reason.{value}" for key, value in vars(Reason).items()
                if not key.startswith("_")}
    expected |= {f"culling.criterion.{name}" for name in CRITERIA}
    expected |= {f"culling.criterion.{name}.hint" for name in CRITERIA}
    expected |= {f"culling.mode.{mode.value}" for mode in CullingMode}
    expected |= {f"culling.mode.{mode.value}.hint" for mode in CullingMode}
    expected |= {f"culling.merge.kind.{kind.value}" for kind in MergeKind}
    expected |= {f"merges.kind.{kind.value}" for kind in MergeKind}
    expected |= {f"merges.decision.{decision.value}" for decision in MergeDecision}
    expected |= {f"culling.unavailable.{reason.value}" for reason in Unavailable}
    expected |= {f"culling.notice.{m.notice}" for m in MODELS if m.notice}
    expected |= {f"status.{status.value}" for status in PhotoStatus}
    missing = sorted(expected - table)
    assert not missing, f"codici senza traduzione: {missing}"


MASKS_TS = FRONTEND / "src" / "lib" / "masks.ts"


def _mask_model(path: str) -> tuple[object, str]:
    """The model a masks-panel control is checked against, and its path in it.

    ``MASK_GROUPS`` paths are into ``MaskParams``; ``SELECTION_CONTROLS``
    paths start with the model: ``mask``, ``radial``, ``range``, ``hue``.
    """
    from ape.pipeline.mask_defs import HueRange, RadialDef, ValueRange
    from ape.pipeline.params import MaskParams

    head, _, rest = path.partition(".")
    models = {"mask": MaskParams, "radial": RadialDef, "range": ValueRange, "hue": HueRange}
    if head in models:
        return models[head], rest
    return MaskParams, path


def test_no_mask_slider_can_reach_a_value_the_server_refuses():
    controls = _controls(MASKS_TS)
    assert len(controls) >= 18, "la tabella dei controlli delle maschere sembra troncata"
    for control in controls:
        model, path = _mask_model(str(control["path"]))
        low, high, default = _field_bounds(path, model)
        name = control["path"]
        if low is not None:
            assert control["min"] >= low, f"{name}: minimo {control['min']} sotto {low}"
        if high is not None:
            assert control["max"] <= high, f"{name}: massimo {control['max']} sopra {high}"
        if isinstance(default, int | float) and not isinstance(default, bool):
            assert control["neutral"] == pytest.approx(float(default)), name


RETOUCH_TS = FRONTEND / "src" / "lib" / "retouch.ts"
RETOUCH_TYPES_TS = FRONTEND / "src" / "lib" / "retouchTypes.ts"


def test_no_retouch_slider_can_reach_a_value_the_server_refuses():
    """The Rimozione panel's controls against ``HealItem`` and ``EraseItem``."""
    from ape.pipeline.retouch_params import EraseItem, HealItem

    controls = _controls(RETOUCH_TS)
    assert len(controls) >= 6, "la tabella dei controlli della rimozione sembra troncata"
    models = {"heal": HealItem, "erase": EraseItem}
    for control in controls:
        head, _, path = str(control["path"]).partition(".")
        low, high, default = _field_bounds(path, models[head])
        name = control["path"]
        if low is not None:
            assert control["min"] >= low, f"{name}: minimo {control['min']} sotto {low}"
        if high is not None:
            assert control["max"] <= high, f"{name}: massimo {control['max']} sopra {high}"
        if isinstance(default, int | float) and not isinstance(default, bool):
            assert control["neutral"] == pytest.approx(float(default)), name


def _interface_fields(text: str, name: str) -> set[str]:
    start = text.index(f"interface {name}")
    body = text[text.index("{", start) + 1 : text.index("\n}", start)]
    return set(re.findall(r"^\s*(\w+)\??:", body, re.MULTILINE))


def test_the_removal_types_have_the_server_s_fields():
    from ape.pipeline.retouch_params import EraseItem, HealItem

    text = RETOUCH_TYPES_TS.read_text(encoding="utf-8")
    base = _interface_fields(text, "ItemBase")
    for interface, model in (("HealItem", HealItem), ("EraseItem", EraseItem)):
        fields = base | _interface_fields(text, interface)
        assert fields == set(model.model_fields), interface


def test_the_typescript_neutral_document_has_no_removals_and_the_version():
    from ape.pipeline.params import PARAMS_VERSION

    text = PARAMS_TS.read_text(encoding="utf-8")
    start = text.index("export function neutralParams()")
    body = text[text.index("return {", start) : text.index("\n}", start)]
    assert re.search(r"retouch:\s*\[\]", body)
    assert re.search(rf"params_version:\s*{PARAMS_VERSION}\b", body)
