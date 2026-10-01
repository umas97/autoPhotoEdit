# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The camera's embedded preview, and the catalogue change that came with it.

Culling reads the JPEG inside the ARW instead of decoding the RAW (section
7.2). Three things about that JPEG must be right for everything downstream to
be: it is upright, the camera's focus point lands where the camera put it, and
a JPEG the camera wrote next to the RAW is preferred when there is one.

The last test here is the schema migration that phase 4 needed: a catalogue
written by the previous build opens, gains the new column, and loses nothing.
"""

from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest
from sqlalchemy import inspect, text

from ape.culling.features import _focus_point
from ape.raw.embedded import orient, preview_paths_for, read_embedded_preview, write_preview_cache


@pytest.mark.fixtures
def test_the_preview_is_upright_and_matches_what_the_camera_meant(raw_fixtures):
    """Orientation is applied from LibRaw's flag, not from the JPEG's own tag;
    the result must be what an EXIF-honouring decoder shows."""
    import rawpy

    for path in raw_fixtures:
        preview = read_embedded_preview(path)
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        honoured = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
        assert preview.source == "embedded"
        assert np.array_equal(preview.image, honoured), path.name


@pytest.mark.fixtures
def test_a_sidecar_jpeg_is_preferred(raw_fixtures, tmp_path):
    raw = raw_fixtures[0]
    card = tmp_path / "card"
    card.mkdir()
    copy = card / raw.name
    shutil.copy2(raw, copy)
    sidecar = card / f"{raw.stem}.JPG"
    image = read_embedded_preview(copy).image
    cv2.imwrite(str(sidecar), cv2.resize(image, None, fx=2, fy=2))
    preview = read_embedded_preview(copy, sidecar=sidecar)
    assert preview.source == "sidecar"


@pytest.mark.fixtures
def test_an_unreadable_sidecar_falls_back_to_the_embedded_preview(raw_fixtures, tmp_path):
    sidecar = tmp_path / "rotto.jpg"
    sidecar.write_bytes(b"\xff\xd8 non un jpeg")
    assert read_embedded_preview(raw_fixtures[0], sidecar=sidecar).source == "embedded"


def test_a_file_that_is_not_a_raw_is_a_clear_error(tmp_path):
    fake = tmp_path / "DSC00001.ARW"
    fake.write_bytes(b"non un raw" * 100)
    with pytest.raises(ValueError, match="illeggibile"):
        read_embedded_preview(fake)
    with pytest.raises(FileNotFoundError):
        read_embedded_preview(tmp_path / "assente.ARW")


def test_the_focus_point_turns_with_the_frame():
    """Sony records ``FocusLocation`` on the sensor, before rotation."""
    tags = {"Exif.Sony2.FocusLocation": "6000 4000 1500 1000"}  # a quarter across, down
    assert _focus_point(tags, 0) == (0.25, 0.25)
    assert _focus_point(tags, 3) == (0.75, 0.75)
    # A quarter turn clockwise: what was on the left is now at the top.
    assert _focus_point(tags, 6) == (0.75, 0.25)
    assert _focus_point(tags, 5) == (0.25, 0.75)
    assert _focus_point({}, 0) is None
    assert _focus_point({"Exif.Sony2.FocusLocation": "6000 4000 0 0"}, 0) is None


def test_the_focus_point_rotation_agrees_with_the_image_rotation():
    """The point and the pixels are turned by the same rule."""
    frame = np.zeros((40, 60, 3), np.uint8)
    frame[10, 15] = 255  # (x=15, y=10): a quarter across, a quarter down
    tags = {"Exif.Sony2.FocusLocation": "60 40 15 10"}
    for flip in (0, 3, 5, 6):
        turned = orient(frame, flip)
        y, x = np.argwhere(turned[..., 0] == 255)[0]
        u, v = _focus_point(tags, flip)
        assert abs(u - x / turned.shape[1]) < 0.05 and abs(v - y / turned.shape[0]) < 0.05, flip


def test_the_cached_copies_live_in_the_cache_and_are_named_by_content(xdg_home):
    from ape.config import get_settings
    from ape.raw.embedded import EmbeddedPreview

    preview = EmbeddedPreview(np.full((1080, 1616, 3), 90, np.uint8), "embedded")
    paths = write_preview_cache(preview, "q:123:abc")
    assert paths == preview_paths_for("q:123:abc")
    assert paths.exist()
    assert get_settings().cache_dir in paths.full.parents
    grid = cv2.imread(str(paths.grid))
    assert max(grid.shape[:2]) == 400
    assert not list(paths.full.parent.glob("*.tmp")), "nessun file temporaneo lasciato"


def test_a_catalogue_of_the_previous_schema_is_migrated(xdg_home):
    """Section 11: when the format changes, write a migration, never break a
    project. A version-1 catalogue has no ``culling_features`` column."""
    from ape.db.models import Project
    from ape.db.session import SCHEMA_VERSION, get_engine, init_db, read_setting, session_scope

    engine = get_engine()
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE photo DROP COLUMN culling_features"))
        connection.execute(text("UPDATE setting SET value = '1' WHERE key = 'schema_version'"))
        connection.execute(
            text("INSERT INTO project (name, source_dir, coherence_lambda, confidence_threshold,"
                 " culling_enabled, culling_mode, culling_aggressiveness, culling_weights,"
                 " culling_criteria, export_template, export_on_conflict, export_strip_gps,"
                 " source_missing, merge_detection_enabled, created_at, updated_at, status)"
                 " VALUES ('vecchio', '/x', 0.3, 0.6, 0, 'conservative', 0.5, '{}', '{}',"
                 " '{basename}.{ext}', 'ask', 0, 0, 1, '2026-01-01', '2026-01-01', 'new')")
        )
    assert "culling_features" not in {c["name"] for c in inspect(engine).get_columns("photo")}

    init_db(engine)
    assert "culling_features" in {c["name"] for c in inspect(engine).get_columns("photo")}
    with session_scope() as session:
        assert int(read_setting(session, "schema_version")) == SCHEMA_VERSION
        assert session.query(Project).one().name == "vecchio"
