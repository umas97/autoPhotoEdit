# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 13 of section 13: export names and collisions (section 16).

"A template that uses every token produces valid names on the filesystem; with
``on_conflict=rename`` no file already in the destination changes hash; with
``skip`` the file is left intact and the job is reported as skipped, not
failed."
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pytest

from ape.db.enums import ExportConflict
from ape.export.naming import (
    PART_SUFFIX,
    NameContext,
    TemplateError,
    render_name,
    render_names,
    sanitise,
    write_exclusive,
)

EVERY_TOKEN = (
    "{project}_{basename}_{counter:03}_{seq}_{date:%Y%m%d}_{time:%H%M%S}"
    "_{camera}_{lens}_{iso}_{focal}.{ext}"
)


def _ctx(**changes) -> NameContext:
    values = {
        "filename": "DSC06312.ARW",
        "ext": "jpg",
        "counter": 7,
        "project": "Evento/Prova: Roma?",
        "camera": "ILCE-7M3",
        "lens": 'E 28-75mm F2.8 "A063"',
        "iso": 250,
        "focal_length": 75.0,
        "shot_at": datetime(2026, 2, 26, 14, 37, 59),
    }
    values.update(changes)
    return NameContext(**values)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_token_gives_a_name_legal_everywhere(tmp_path):
    name = render_name(EVERY_TOKEN, _ctx())
    assert name == (
        "Evento_Prova_ Roma__DSC06312_007_06312_20260226_143759"
        "_ILCE-7M3_E 28-75mm F2.8 _A063__250_75mm.jpg"
    )
    assert not set('<>:"/\\|?*') & set(name)
    assert len(name.encode()) <= 255
    # And the filesystem agrees.
    (tmp_path / name).write_bytes(b"x")
    assert (tmp_path / name).read_bytes() == b"x"


def test_the_default_template_keeps_the_raw_name():
    assert render_name("{basename}.{ext}", _ctx()) == "DSC06312.jpg"


def test_a_template_without_the_extension_still_gets_one():
    assert render_name("{basename}_web", _ctx(ext="tif")) == "DSC06312_web.tif"


def test_path_separators_do_not_make_folders():
    assert render_name("../{basename}/x.{ext}", _ctx()) == ".._DSC06312_x.jpg"


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("foto.", "foto"),
        ("CON.jpg", "_CON.jpg"),
        ("a\x00b\x1fc.jpg", "a_b_c.jpg"),
        ("é.jpg", "é.jpg"),  # NFC: one name, however it was typed
    ],
)
def test_sanitise(raw, clean):
    assert sanitise(raw) == clean


def test_long_names_are_cut_at_255_bytes_keeping_the_extension():
    name = render_name("{basename}_" + "è" * 300 + ".{ext}", _ctx())
    assert len(name.encode()) <= 255 and name.endswith(".jpg")


@pytest.mark.parametrize(
    "template",
    ["", "   ", "{nome}.{ext}", "{basename.{ext}", "{counter:zz}.{ext}"],
)
def test_invalid_templates_are_refused(template):
    with pytest.raises(TemplateError):
        render_name(template, _ctx())


def test_a_template_that_gives_two_photos_one_name_is_refused():
    contexts = [_ctx(filename="DSC00001.ARW", counter=1), _ctx(filename="DSC00002.ARW", counter=2)]
    with pytest.raises(TemplateError, match="stesso nome"):
        render_names("{camera}.{ext}", contexts)
    assert render_names("{camera}_{counter:03}.{ext}", contexts) == [
        "ILCE-7M3_001.jpg",
        "ILCE-7M3_002.jpg",
    ]


def test_duplicates_are_compared_ignoring_case():
    """An exFAT card thinks ``A.jpg`` and ``a.jpg`` are the same file."""
    contexts = [_ctx(filename="A.ARW"), _ctx(filename="a.ARW")]
    with pytest.raises(TemplateError):
        render_names("{basename}.{ext}", contexts)


def test_rename_never_changes_an_existing_file(tmp_path):
    existing = tmp_path / "DSC06312.jpg"
    existing.write_bytes(b"the user's own file")
    taken = tmp_path / "DSC06312_1.jpg"
    taken.write_bytes(b"another one")
    before = {p.name: _digest(p) for p in tmp_path.iterdir()}

    outcome = write_exclusive(
        tmp_path, "DSC06312.jpg", b"new", ExportConflict.RENAME, reserved=["DSC06312_2.jpg"]
    )

    assert outcome.outcome == "renamed"
    # _1 exists and _2 belongs to another photo of the same batch.
    assert outcome.path == tmp_path / "DSC06312_3.jpg"
    assert outcome.path.read_bytes() == b"new"
    assert {p: _digest(tmp_path / p) for p in before} == before
    assert not list(tmp_path.glob(f"*{PART_SUFFIX}"))


def test_skip_leaves_the_file_alone(tmp_path):
    existing = tmp_path / "DSC06312.jpg"
    existing.write_bytes(b"keep me")
    outcome = write_exclusive(tmp_path, "DSC06312.jpg", b"new", ExportConflict.SKIP)
    assert outcome.outcome == "skipped" and outcome.path is None
    assert existing.read_bytes() == b"keep me"
    assert not list(tmp_path.glob(f"*{PART_SUFFIX}"))


def test_overwrite_replaces_only_when_asked(tmp_path):
    existing = tmp_path / "DSC06312.jpg"
    existing.write_bytes(b"old")
    outcome = write_exclusive(tmp_path, "DSC06312.jpg", b"new", ExportConflict.OVERWRITE)
    assert outcome.outcome == "overwritten" and existing.read_bytes() == b"new"
    fresh = write_exclusive(tmp_path, "other.jpg", b"n", ExportConflict.OVERWRITE)
    assert fresh.outcome == "written"


def test_a_name_that_is_free_is_simply_written(tmp_path):
    outcome = write_exclusive(tmp_path, "a.jpg", b"data", ExportConflict.RENAME)
    assert outcome.outcome == "written" and (tmp_path / "a.jpg").read_bytes() == b"data"


# --------------------------------------------------------------------------- #
# The same, through a whole batch: the policies as the queue applies them.


@pytest.fixture
def project(catalog, tmp_path, monkeypatch):
    from export_helpers import make_project, synthetic_decode

    synthetic_decode(monkeypatch)
    return make_project(catalog, tmp_path / "card")


def _batch_items(catalog, batch_id):
    from ape.db.models import ExportItem

    with catalog() as session:
        return {
            i.name: (i.state.value, i.outcome, i.path)
            for i in session.query(ExportItem).filter_by(batch_id=batch_id)
        }


def test_a_batch_with_rename_keeps_every_existing_file(catalog, project, tmp_path):
    from export_helpers import run_queue, start_batch

    out = tmp_path / "export"
    out.mkdir()
    (out / "DSC00002.jpg").write_bytes(b"exported last week, then retouched by hand")
    (out / "notes.txt").write_text("appunti")
    before = {p.name: _digest(p) for p in out.iterdir()}

    batch = start_batch(catalog, project, output_dir=str(out), on_conflict=ExportConflict.RENAME)
    run_queue()

    items = _batch_items(catalog, batch)
    assert items["DSC00002.jpg"][:2] == ("done", "renamed")
    assert Path(items["DSC00002.jpg"][2]).name == "DSC00002_1.jpg"
    assert {n: _digest(out / n) for n in before} == before
    assert sorted(p.name for p in out.iterdir()) == [
        "DSC00001.jpg",
        "DSC00002.jpg",
        "DSC00002_1.jpg",
        "DSC00003.jpg",
        "notes.txt",
    ]


def test_a_batch_with_skip_reports_skipped_not_failed(catalog, project, tmp_path):
    from ape.db.models import ExportBatch
    from ape.export.batch import batch_summary
    from export_helpers import run_queue, start_batch

    out = tmp_path / "export"
    out.mkdir()
    kept = out / "DSC00001.jpg"
    kept.write_bytes(b"do not touch")

    batch = start_batch(catalog, project, output_dir=str(out), on_conflict=ExportConflict.SKIP)
    run_queue()

    assert kept.read_bytes() == b"do not touch"
    items = _batch_items(catalog, batch)
    assert items["DSC00001.jpg"][:2] == ("skipped", "skipped")
    with catalog() as session:
        summary = batch_summary(session, session.get(ExportBatch, batch))
    assert summary["state"] == "done"
    assert summary["counts"]["failed"] == 0
    assert summary["counts"]["skipped"] == 1 and summary["counts"]["done"] == 2
