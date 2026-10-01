# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test 12 of docs/SPEC.md section 13: importing is idempotent.

"Re-importing the same folder creates no duplicates and preserves edits, culling
decisions, clusters and approvals; adding a file imports it on its own; removing
a file marks it ``missing`` without deleting the row or touching the disk. Two
files with identical content produce **one** Photo with two ``duplicate_paths``.
A JPEG with the same basename as an ARW does not become a Photo."

Each of those clauses is a test below, in that order. The one they all rest on
is the first: a photo is its content, so the catalogue can be rebuilt around a
folder that moved, was renamed, or grew, without the user losing a single edit.
"""

from __future__ import annotations

from pathlib import Path

from ape.db.models import EditVersion, EditVersionSource, Photo, Project
from ape.importer import import_folder, remap_source, scan_folder
from conftest_catalog import file_state, touch_jpeg, write_raw


def _project(session, folder: Path, name: str = "prova") -> Project:
    project = Project(name=name, source_dir=str(folder))
    session.add(project)
    session.flush()
    return project


def _photos(session, project_id: int) -> list[Photo]:
    return sorted(
        session.query(Photo).filter(Photo.project_id == project_id).all(),
        key=lambda p: p.filename,
    )


def test_a_folder_is_read_but_never_written(card: Path, catalog):
    """Section 2 first: the import must not change a single byte of the card."""
    before = file_state(card)
    with catalog() as session:
        project = _project(session, card)
        import_folder(session, project)
        session.commit()
    assert file_state(card) == before


def test_only_arw_files_become_photos(card: Path, catalog):
    with catalog() as session:
        project = _project(session, card)
        summary = import_folder(session, project)
        session.commit()
        photos = _photos(session, project.id)

    assert summary.imported == 6
    assert {p.filename for p in photos} == {f"DSC0{i:04d}.ARW" for i in range(1, 7)}
    # The subfolder is ignored but declared, because silence here turns into
    # photos the user cannot find.
    assert summary.subdirectories_ignored == 1
    # Another manufacturer's RAW is refused out loud, not skipped quietly.
    assert summary.rejected_formats == {".cr2": 1}


def test_a_sidecar_jpeg_is_attached_not_imported(card: Path, catalog):
    with catalog() as session:
        project = _project(session, card)
        import_folder(session, project)
        session.commit()
        photos = {p.filename: p for p in _photos(session, project.id)}

    assert "DSC00001.JPG" not in photos
    assert photos["DSC00001.ARW"].sidecar_jpeg_path == str(card / "DSC00001.JPG")
    assert photos["DSC00002.ARW"].sidecar_jpeg_path is None


def test_reimporting_changes_nothing_and_keeps_the_work(card: Path, catalog):
    """The headline case: a second import must cost the user nothing."""
    with catalog() as session:
        project = _project(session, card)
        import_folder(session, project)
        session.commit()
        photo = _photos(session, project.id)[0]
        # Everything a re-import must not touch.
        session.add(
            EditVersion(
                photo_id=photo.id,
                params={"params_version": 1, "exposure": {"ev": 0.75}},
                source=EditVersionSource.USER_EDITED,
                is_current=True,
            )
        )
        photo.culled = True
        photo.cull_decided_by = "user"
        photo.cluster_id = 7
        photo.confidence = 0.42
        photo.status = "approved"
        session.commit()
        photo_id, project_id = photo.id, project.id

    with catalog() as session:
        project = session.get(Project, project_id)
        summary = import_folder(session, project)
        session.commit()

        assert summary.imported == 0
        assert summary.already_present == 6
        assert len(_photos(session, project_id)) == 6

        kept = session.get(Photo, photo_id)
        assert kept.culled is True
        assert kept.cull_decided_by.value == "user"
        assert kept.cluster_id == 7
        assert kept.confidence == 0.42
        assert kept.status.value == "approved"
        versions = session.query(EditVersion).filter(EditVersion.photo_id == photo_id).all()
        assert len(versions) == 1
        assert versions[0].params["exposure"]["ev"] == 0.75


def test_a_new_file_is_imported_on_its_own(card: Path, catalog):
    with catalog() as session:
        project = _project(session, card)
        import_folder(session, project)
        session.commit()
        project_id = project.id

    write_raw(card, "DSC00007.ARW")

    with catalog() as session:
        project = session.get(Project, project_id)
        summary = import_folder(session, project)
        session.commit()
        assert summary.imported == 1
        assert summary.already_present == 6
        assert len(_photos(session, project_id)) == 7


def test_a_removed_file_is_marked_missing_and_the_row_survives(card: Path, catalog):
    with catalog() as session:
        project = _project(session, card)
        import_folder(session, project)
        session.commit()
        project_id = project.id
        gone = _photos(session, project_id)[0]
        gone_id, gone_name = gone.id, gone.filename

    (card / gone_name).unlink()

    with catalog() as session:
        project = session.get(Project, project_id)
        summary = import_folder(session, project)
        session.commit()

        assert summary.marked_missing == 1
        assert len(_photos(session, project_id)) == 6, "la riga non va cancellata"
        assert session.get(Photo, gone_id).missing is True

    # And nothing else on the card was touched by the discovery.
    assert not (card / gone_name).exists()
    assert len(list(card.glob("*.ARW"))) == 5


def test_a_file_that_comes_back_stops_being_missing(card: Path, catalog):
    with catalog() as session:
        project = _project(session, card)
        import_folder(session, project)
        session.commit()
        project_id = project.id
        target = _photos(session, project_id)[0]
        name, photo_id = target.filename, target.id

    content = (card / name).read_bytes()
    (card / name).unlink()
    with catalog() as session:
        import_folder(session, session.get(Project, project_id))
        session.commit()
        assert session.get(Photo, photo_id).missing is True

    (card / name).write_bytes(content)
    with catalog() as session:
        import_folder(session, session.get(Project, project_id))
        session.commit()
        assert session.get(Photo, photo_id).missing is False


def test_identical_files_are_one_photo_with_two_paths(tmp_path: Path, catalog):
    folder = tmp_path / "doppioni"
    first = write_raw(folder, "DSC00001.ARW")
    write_raw(folder, "COPIA.ARW", content=first.read_bytes())

    with catalog() as session:
        project = _project(session, folder)
        summary = import_folder(session, project)
        session.commit()
        photos = _photos(session, project.id)

    assert summary.imported == 1
    assert summary.duplicates == 1
    assert len(photos) == 1
    assert photos[0].duplicate_paths == [str(folder / "DSC00001.ARW")]
    assert photos[0].filename == "COPIA.ARW"


def test_files_of_the_same_size_are_not_duplicates(tmp_path: Path, catalog):
    """Same length, different bytes: two photos, as a photographer would expect."""
    folder = tmp_path / "coincidenze"
    write_raw(folder, "A.ARW", content=b"a" * 8192)
    write_raw(folder, "B.ARW", content=b"b" * 8192)

    with catalog() as session:
        project = _project(session, folder)
        summary = import_folder(session, project)
        session.commit()
        assert summary.imported == 2
        assert summary.duplicates == 0


def test_a_renamed_file_stays_the_same_photo(card: Path, catalog):
    """Identity is content: renaming on disk must not produce a second row."""
    with catalog() as session:
        project = _project(session, card)
        import_folder(session, project)
        session.commit()
        project_id = project.id
        original = _photos(session, project_id)[0]
        photo_id, old_name = original.id, original.filename

    (card / old_name).rename(card / "RINOMINATA.ARW")

    with catalog() as session:
        project = session.get(Project, project_id)
        summary = import_folder(session, project)
        session.commit()
        photos = _photos(session, project_id)

    assert summary.imported == 0
    assert summary.moved == 1
    assert len(photos) == 6
    # The same row, with everything that was ever done to it, now points at the
    # new name. A rename is not a new photograph.
    followed = next(p for p in photos if p.id == photo_id)
    assert followed.filename == "RINOMINATA.ARW"
    assert followed.missing is False


def test_remapping_a_moved_folder_reattaches_by_content(card: Path, tmp_path: Path, catalog):
    with catalog() as session:
        project = _project(session, card)
        import_folder(session, project)
        session.commit()
        project_id = project.id
        edited = _photos(session, project_id)[0]
        edited.cluster_id = 3
        session.commit()
        edited_id = edited.id

    moved = tmp_path / "disco-esterno" / "card"
    moved.parent.mkdir(parents=True, exist_ok=True)
    card.rename(moved)

    with catalog() as session:
        project = session.get(Project, project_id)
        summary = remap_source(session, project, moved)
        session.commit()

        assert summary.reattached == 6
        assert summary.still_missing == 0
        assert project.source_dir == str(moved)
        reattached = session.get(Photo, edited_id)
        assert reattached.cluster_id == 3, "il riaggancio non deve perdere il lavoro fatto"
        assert Path(reattached.path).parent == moved


def test_scanning_reports_without_importing(card: Path):
    scan = scan_folder(card)
    assert len(scan.raws) == 6
    assert set(scan.sidecars) == {"dsc00001"}
    assert [p.name for p in scan.subdirectories] == ["sottocartella"]
    assert scan.rejected == {".cr2": 1}
    assert scan.other_files == 1  # appunti.txt


def test_an_orphan_jpeg_is_not_a_sidecar(tmp_path: Path):
    folder = tmp_path / "solo-jpeg"
    folder.mkdir()
    touch_jpeg(folder, "VACANZA.JPG")
    scan = scan_folder(folder)
    assert scan.sidecars == {}
    assert scan.other_files == 1
