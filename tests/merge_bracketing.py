# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A hand-held bracketing of three stand-in RAWs, for the merge tests.

The RAWs are stand-in files -- the importer reads bytes, not pixels -- and the
merge decodes them, through a patched decoder, into a synthetic bracketing of a
real A7 III frame. Everything downstream of the merge reads the intermediate
with the real decoder. Loaded as a plugin by conftest.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import select

from ape.db.models import Photo, Project
from ape.raw.decode import DecodedRaw, decode_linear
from ape.raw.intermediate import is_intermediate
from ape.raw.metadata import PhotoMetadata
from conftest_catalog import write_raw

__all__ = ["FRAMES", "T0", "bracketing", "project"]

#: Name -> (EV, shift and rotation) of the hand-held bracketing.
FRAMES = {
    "DSC00001.ARW": (0.0, None),
    "DSC00002.ARW": (-2.0, (4.0, -3.0, 0.2)),
    "DSC00003.ARW": (2.0, (-2.5, 3.5, -0.15)),
}
_SHUTTER = {"DSC00001.ARW": 1 / 250, "DSC00002.ARW": 1 / 1000, "DSC00003.ARW": 1 / 60}
T0 = datetime(2026, 2, 26, 14, 0, 0)


@pytest.fixture
def bracketing(raw_fixtures, monkeypatch):
    real = decode_linear(raw_fixtures[16], half_size=True)
    small = cv2.resize(real.rgb, (900, 600), interpolation=cv2.INTER_AREA)
    scene = small / np.float32(np.percentile(small.max(axis=-1), 85))

    def fake_decode(path, *, half_size=False, white_balance=None, **_kwargs):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        ev, motion = FRAMES[path.name]
        frame = np.clip(scene * np.float32(2.0**ev), None, 1.0)
        if motion is not None:
            matrix = cv2.getRotationMatrix2D((450, 300), motion[2], 1.0)
            matrix[:, 2] += motion[:2]
            frame = cv2.warpAffine(frame, matrix, (900, 600), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REFLECT)
        return DecodedRaw(rgb=frame.astype(np.float32), camera=real.camera,
                          baseline_exposure_ev=real.baseline_exposure_ev, source_path=path)

    import ape.raw.metadata as metadata

    original = metadata.read_metadata

    def fake_metadata(path):
        if is_intermediate(path):
            return original(path)
        name = Path(path).name
        return PhotoMetadata(
            camera_make="SONY", camera_model="ILCE-7M3", lens_model="E 28-75mm F2.8 A063",
            iso=100, aperture=8.0, shutter=_SHUTTER[name], focal_length=35.0,
            shot_at=T0 + timedelta(seconds=int(name[3:8])), width=6000, height=4000,
            raw_tags={"Exif.Image.Make": "SONY", "Exif.Image.Model": "ILCE-7M3",
                      "Exif.Photo.FNumber": "8/1", "Exif.Photo.ISOSpeedRatings": "100"},
        )

    monkeypatch.setattr("ape.merge.engine.decode_linear", fake_decode)
    monkeypatch.setattr("ape.raw.metadata.read_metadata", fake_metadata)


@pytest.fixture
def project(catalog, tmp_path):
    from ape.importer import import_folder

    source = tmp_path / "card"
    for name in FRAMES:
        write_raw(source, name)
    with catalog() as session:
        project = Project(name="hdr", source_dir=str(source.resolve()))
        session.add(project)
        session.flush()
        import_folder(session, project)
        for photo in session.scalars(select(Photo).where(Photo.project_id == project.id)):
            photo.shot_at = T0 + timedelta(seconds=int(photo.filename[3:8]))
            photo.shutter, photo.aperture, photo.iso = _SHUTTER[photo.filename], 8.0, 100
            photo.focal_length, photo.camera = 35.0, "ILCE-7M3"
        session.commit()
        return project.id, source
