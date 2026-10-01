# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 8 of docs/SPEC.md section 13: non-destructiveness.

This test is not optional and is never skipped. It is the one that says the
program did not touch the user's originals.

Two halves:

* **the guard**, ``assert_outside_source``, must refuse every route into a
  protected folder, including the ones that go through a symlink;
* **the pipeline**, run end to end, must leave every byte and every mtime of the
  source folder exactly as it found them.

The end-to-end half covers what the program has today -- import into the
catalogue, the culling analysis and selection, decode, render, export -- and
grows as the phases land, so that the cycle this test verifies is always the
whole program. When real ARW files are available one of them joins the source
folder, so that the culling job reads a real embedded preview and not only a
file it has to reject.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from ape.export.image import ExportFormat, save_image
from ape.pipeline.params import neutral_params
from ape.pipeline.render import render
from ape.safety import (
    SourceWriteError,
    assert_outside_source,
    guarded_open,
    register_protected_root,
    unregister_protected_root,
)


def _fingerprint(directory: Path) -> dict[str, tuple[str, int, int]]:
    """SHA-256, size and mtime_ns of every file under ``directory``."""
    result: dict[str, tuple[str, int, int]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[str(path.relative_to(directory))] = (digest, stat.st_size, stat.st_mtime_ns)
    return result


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """A stand-in source folder with a few files, none of which may change."""
    import shutil

    from conftest import available_raws

    source = tmp_path / "scheda-sd" / "DCIM"
    source.mkdir(parents=True)
    (source / "DSC00001.ARW").write_bytes(b"non e' un RAW vero, ma e' un file dell'utente")
    (source / "DSC00001.JPG").write_bytes(b"jpeg affiancato")
    (source / "note.txt").write_text("appunti dello scatto", encoding="utf-8")
    nested = source / "sottocartella"
    nested.mkdir()
    (nested / "DSC00002.ARW").write_bytes(b"un altro file")
    real = available_raws()
    if real:
        shutil.copy2(real[0], source / "DSC00003.ARW")
    return source


def test_guard_refuses_a_direct_path(source_dir):
    with pytest.raises(SourceWriteError):
        assert_outside_source(source_dir / "uscita.jpg", source_dir)


def test_guard_refuses_the_folder_itself(source_dir):
    with pytest.raises(SourceWriteError):
        assert_outside_source(source_dir, source_dir)


def test_guard_refuses_a_nested_path(source_dir):
    with pytest.raises(SourceWriteError):
        assert_outside_source(source_dir / "a" / "b" / "c.tif", source_dir)


def test_guard_refuses_a_path_reached_through_a_symlink(source_dir, tmp_path):
    """The spec calls this out by name, and it is the case a naive check misses."""
    link = tmp_path / "scorciatoia"
    link.symlink_to(source_dir, target_is_directory=True)
    with pytest.raises(SourceWriteError):
        assert_outside_source(link / "uscita.jpg", source_dir)


def test_guard_refuses_a_file_whose_parent_is_a_symlink(source_dir, tmp_path):
    """Writing into a symlinked directory writes wherever the link points."""
    link = tmp_path / "esporta-qui"
    link.symlink_to(source_dir, target_is_directory=True)
    with pytest.raises(SourceWriteError):
        assert_outside_source(link / "sub" / "uscita.jpg", source_dir)


def test_guard_refuses_relative_paths_that_climb_back_in(source_dir, tmp_path):
    outside = tmp_path / "export"
    outside.mkdir()
    sneaky = outside / ".." / source_dir.relative_to(tmp_path) / "uscita.jpg"
    with pytest.raises(SourceWriteError):
        assert_outside_source(sneaky, source_dir)


def test_guard_allows_a_sibling_folder(source_dir, tmp_path):
    destination = tmp_path / "export" / "uscita.jpg"
    assert assert_outside_source(destination, source_dir) == destination.resolve()


def test_guard_allows_a_lookalike_sibling(source_dir, tmp_path):
    """``/DCIM-export`` is not inside ``/DCIM``: a prefix check would say it is."""
    destination = source_dir.parent / f"{source_dir.name}-export" / "uscita.jpg"
    assert assert_outside_source(destination, source_dir)


def test_process_wide_registry_protects_without_being_passed(source_dir, tmp_path):
    """Forgetting to thread the project through a call must not disarm the guard."""
    register_protected_root(source_dir)
    try:
        with pytest.raises(SourceWriteError):
            assert_outside_source(source_dir / "uscita.jpg")
    finally:
        unregister_protected_root(source_dir)
    assert assert_outside_source(source_dir / "uscita.jpg") is not None


def test_guarded_open_refuses_to_write_inside(source_dir):
    with pytest.raises(SourceWriteError):  # noqa: SIM117 - the context is the subject
        with guarded_open(source_dir / "uscita.bin", "wb", source=source_dir):
            pass


def test_guarded_open_rejects_read_modes(tmp_path):
    with pytest.raises(ValueError, match="read-only"):  # noqa: SIM117
        with guarded_open(tmp_path / "x", "rb"):
            pass


def test_guarded_open_writes_outside_and_creates_parents(source_dir, tmp_path):
    destination = tmp_path / "export" / "nested" / "uscita.bin"
    with guarded_open(destination, "wb", source=source_dir) as handle:
        handle.write(b"ok")
    assert destination.read_bytes() == b"ok"


def _run_culling(source: Path) -> None:
    """Import the folder, analyse it for culling, group, select, confirm."""
    from ape.culling import decisions, service
    from ape.db.models import Project
    from ape.db.session import session_scope
    from ape.importer import import_folder
    from ape.jobs.worker import worker_loop

    with session_scope() as session:
        project = Project(name="ciclo", source_dir=str(source))
        session.add(project)
        session.flush()
        import_folder(session, project)
        service.enqueue_culling(session, project)
        project_id = project.id

    class _Never:
        @staticmethod
        def is_set() -> bool:
            return False

    from ape.config import get_settings

    # The unreadable stand-ins fail, as they must; the loop moves on.
    worker_loop(get_settings().db_path, _Never(), idle_exit=True)

    with session_scope() as session:
        project = session.get(Project, project_id)
        service.ensure_grouped(session, project)
        state = service.apply_selection(session, project)
        decisions.set_decisions(session, project, [(d.id, "keep") for d in state.decisions])
        decisions.confirm_selection(session, project)
        analysed = service.state(session, project)["analysed"]

    from conftest import available_raws

    # Not a vacuous pass: with a real file in the folder, the culling job must
    # actually have read it.
    if available_raws():
        assert analysed >= 1, "il RAW reale non è stato analizzato"


def _run_retouch() -> None:
    """The removals: a spot and an eraser on the real ARW, the fill made by a worker.

    Every file they write -- the painted area, the patch -- goes through the
    guard; the export then develops the photo with both.
    """
    import cv2
    from sqlalchemy import select

    from ape import masks_store
    from ape.api.routes_photos import add_version
    from ape.config import get_settings
    from ape.db.models import EditVersionSource, Photo
    from ape.db.session import session_scope
    from ape.jobs.worker import worker_loop
    from ape.pipeline.params import EditParams
    from ape.retouch.service import enqueue_fills

    with session_scope() as session:
        photo = session.scalars(select(Photo).where(Photo.filename == "DSC00003.ARW")).first()
        if photo is None or photo.status.value == "failed":
            return  # no real ARW in the folder: the stand-ins cannot be decoded
        raster = np.zeros((683, 1024), np.uint8)
        cv2.circle(raster, (512, 340), 20, 255, -1)
        ok, encoded = cv2.imencode(".png", raster)
        assert ok
        area = masks_store.save(encoded.tobytes())
        params = EditParams.model_validate({"retouch": [
            {"kind": "heal", "id": "h", "cx": 0.3, "cy": 0.4, "radius": 0.01,
             "sx": 0.34, "sy": 0.4},
            {"kind": "erase", "id": "e", "area": area},
        ]})
        add_version(session, photo, params, EditVersionSource.USER_EDITED)
        enqueue_fills(session, photo)

    class _Never:
        @staticmethod
        def is_set() -> bool:
            return False

    worker_loop(get_settings().db_path, _Never(), idle_exit=True)
    assert list(get_settings().retouch_dir.glob("*.patch")), "la gomma non è stata riempita"


def _run_export(destination: Path) -> None:
    """Phase 8: an export batch through the queue, images and both sidecars.

    The stand-in files have failed their analysis and are not selected; the
    real ARW, when there is one, is developed at full resolution and written
    with its darktable and Adobe sidecars.
    """
    from sqlalchemy import select

    from ape.db.models import Project
    from ape.db.session import session_scope
    from ape.export import service
    from ape.export.settings import ExportSettings, load_settings
    from ape.jobs.worker import worker_loop

    with session_scope() as session:
        project = session.scalars(select(Project)).first()
        settings = ExportSettings.model_validate(
            load_settings(project).model_dump(mode="json")
            | {
                "output_dir": str(destination / "batch"),
                "on_conflict": "rename",
                "long_edge": 1024,
                "xmp_darktable": True,
                "xmp_adobe": True,
            }
        )
        try:
            service.start(session, project, settings)
        except service.PlanError:
            return  # no readable photo in the folder: nothing to export

    class _Never:
        @staticmethod
        def is_set() -> bool:
            return False

    from ape.config import get_settings

    worker_loop(get_settings().db_path, _Never(), idle_exit=True)

    from conftest import available_raws

    if available_raws():
        written = sorted(p.name for p in (destination / "batch").iterdir())
        assert written == ["DSC00003.ARW.xmp", "DSC00003.jpg", "DSC00003.xmp"], written


def _run_full_cycle(source: Path, destination: Path) -> None:
    """Everything the program does today, end to end, over a source folder.

    Grows with each phase. Today: the import and the catalogue, the culling
    analysis and selection, the decode, render and export of a frame, the
    removals (a spot and an eraser, filled by a worker), and an export batch
    with its sidecars (phase 8).
    """
    from conftest import as_decoded

    register_protected_root(source)
    try:
        # Read every file exactly as the import scan will (section 15).
        for path in sorted(source.rglob("*")):
            if path.is_file():
                with open(path, "rb") as handle:
                    hashlib.sha256(handle.read(1 << 20)).hexdigest()

        _run_culling(source)

        rng = np.random.default_rng(3)
        scene = rng.uniform(0.01, 0.9, size=(96, 128, 3)).astype(np.float32)
        image = render(as_decoded(scene), neutral_params())
        save_image(image, destination / "sviluppata.jpg", ExportFormat.JPEG, source=source)
        save_image(image, destination / "sviluppata.tif", ExportFormat.TIFF16, source=source)
        _run_retouch()
        _run_export(destination)
    finally:
        unregister_protected_root(source)


def test_a_full_cycle_leaves_the_source_untouched(source_dir, tmp_path, catalog):
    """The headline assertion. Never skipped, never marked xfail."""
    before = _fingerprint(source_dir)
    assert before, "la cartella sorgente di prova e' vuota: il test non proverebbe nulla"

    _run_full_cycle(source_dir, tmp_path / "export")

    after = _fingerprint(source_dir)
    assert after == before, (
        "la cartella sorgente e' cambiata: "
        f"{sorted(set(before) ^ set(after)) or 'stessi file, contenuto o mtime diversi'}"
    )


def test_the_export_refuses_the_source_folder(source_dir):
    """An export aimed at the source folder fails; it does not overwrite."""
    from conftest import as_decoded

    rng = np.random.default_rng(4)
    scene = rng.uniform(0.01, 0.9, size=(32, 48, 3)).astype(np.float32)
    image = render(as_decoded(scene), neutral_params())

    before = _fingerprint(source_dir)
    with pytest.raises(SourceWriteError):
        save_image(image, source_dir / "DSC00001.JPG", ExportFormat.JPEG, source=source_dir)
    assert _fingerprint(source_dir) == before


def test_the_source_folder_is_never_opened_for_writing(
    source_dir, tmp_path, monkeypatch, catalog
):
    """Belt and braces: intercept ``open`` itself and watch what modes it sees."""
    real_open = open
    offences: list[tuple[str, str]] = []

    def watching_open(file, mode="r", *args, **kwargs):
        try:
            resolved = Path(os.fspath(file)).resolve()
        except TypeError:  # a file descriptor, not a path
            return real_open(file, mode, *args, **kwargs)
        writes = any(flag in mode for flag in ("w", "a", "x", "+"))
        if writes and (resolved == source_dir or source_dir.resolve() in resolved.parents):
            offences.append((str(resolved), mode))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", watching_open)
    _run_full_cycle(source_dir, tmp_path / "export")
    assert not offences, f"aperture in scrittura dentro la sorgente: {offences}"


def test_sidecars_beside_the_raws_only_ever_add_xmp_files(source_dir, tmp_path):
    """Section 2.4: the one allowed write near an original creates, never changes.

    The user's own sidecar -- from darktable, say -- is left byte for byte as it
    was, and nothing but ``.xmp`` files named after a RAW can be created.
    """
    from ape.safety import create_beside_source

    raw = source_dir / "DSC00001.ARW"
    theirs = source_dir / "DSC00001.ARW.xmp"
    theirs.write_bytes(b"<x:xmpmeta>la modifica dell'utente in darktable</x:xmpmeta>")
    before = _fingerprint(source_dir)

    assert create_beside_source(raw, "DSC00001.ARW.xmp", b"nostro") is False
    assert create_beside_source(raw, "DSC00001.xmp", b"<adobe/>") is True
    for forbidden in ("DSC00001.jpg", "DSC00001.ARW", "altro.xmp", "../DSC00001.xmp"):
        with pytest.raises(SourceWriteError):
            create_beside_source(raw, forbidden, b"no")

    after = _fingerprint(source_dir)
    assert {k: v for k, v in after.items() if k in before} == before
    assert sorted(set(after) - set(before)) == ["DSC00001.xmp"]
