# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Helpers for the export tests: a project of fake cards with a fake sensor.

The batch logic -- names, collisions, pause, retry, what is recorded -- does not
depend on what the pixels are, and a full-resolution export of a real ARW costs
seconds. So these helpers import stand-in files (bytes with the right name, the
importer never looks inside) and replace the decoder with a small synthetic
frame for the duration of a test. The real decode is exercised by the tests
marked ``fixtures`` and by the non-destructiveness test.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ape.raw.metadata import PhotoMetadata


def synthetic_decode(monkeypatch: pytest.MonkeyPatch, *, fill: float | None = None) -> None:
    """Make ``decode_linear`` return a 96x128 frame, and ``read_metadata`` a Sony's EXIF."""
    from conftest import as_decoded

    def fake_decode(path, **_kwargs):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        if Path(path).read_bytes()[:7] == b"CORRUPT":
            raise ValueError(f"file RAW illeggibile: {Path(path).name} (LibRaw: data corrupted)")
        if fill is not None:
            scene = np.full((96, 128, 3), fill, dtype=np.float32)
        else:
            seed = sum(Path(path).name.encode())
            rng = np.random.default_rng(seed)
            scene = rng.uniform(0.01, 0.9, size=(96, 128, 3)).astype(np.float32)
        decoded = as_decoded(scene)
        decoded.source_path = Path(path)
        return decoded

    def fake_metadata(path):
        meta = PhotoMetadata(camera_make="SONY", camera_model="ILCE-7M3")
        meta.raw_tags = {
            "Exif.Image.Make": "SONY",
            "Exif.Image.Model": "ILCE-7M3",
            "Exif.Photo.ExposureTime": "1/125",
            "Exif.Photo.FNumber": "28/10",
            "Exif.Photo.ISOSpeedRatings": "250",
            "Exif.Photo.FocalLength": "750/10",
            "Exif.Photo.LensModel": "E 28-75mm F2.8 A063",
            "Exif.Photo.DateTimeOriginal": "2026:02:26 14:37:59",
            "Exif.Photo.BodySerialNumber": "1234567",
            "Exif.GPSInfo.GPSLatitudeRef": "N",
            "Exif.GPSInfo.GPSLatitude": "43/1 4/1 0/1",
            "Exif.GPSInfo.GPSLongitudeRef": "E",
            "Exif.GPSInfo.GPSLongitude": "12/1 37/1 0/1",
        }
        return meta

    monkeypatch.setattr("ape.raw.decode.decode_linear", fake_decode)
    monkeypatch.setattr("ape.raw.metadata.read_metadata", fake_metadata)


def make_project(catalog, source: Path, count: int = 3, *, name: str = "export") -> int:
    """Import ``count`` stand-in RAWs and give them shooting times. Returns the id."""
    from ape.db.models import Photo, Project
    from ape.importer import import_folder
    from conftest_catalog import write_raw

    for index in range(count):
        write_raw(source, f"DSC{index + 1:05d}.ARW")
    with catalog() as session:
        project = Project(name=name, source_dir=str(source.resolve()))
        session.add(project)
        session.flush()
        import_folder(session, project)
        start = datetime(2026, 2, 26, 14, 0, 0)
        for offset, photo in enumerate(
            sorted(session.query(Photo).filter_by(project_id=project.id), key=lambda p: p.filename)
        ):
            photo.shot_at = start + timedelta(seconds=10 * offset)
            photo.camera = "ILCE-7M3"
        session.commit()
        return project.id


def run_queue() -> None:
    """Run every queued job in this process, then return."""
    from ape.config import get_settings
    from ape.jobs.worker import worker_loop

    class _Never:
        @staticmethod
        def is_set() -> bool:
            return False

    worker_loop(get_settings().db_path, _Never(), idle_exit=True)


def start_batch(catalog, project_id: int, **changes):
    """Start a batch with ``changes`` on top of the project's settings."""
    from ape.db.models import Project
    from ape.export import service
    from ape.export.settings import ExportSettings, load_settings

    with catalog() as session:
        project = session.get(Project, project_id)
        settings = ExportSettings.model_validate(
            load_settings(project).model_dump(mode="json") | changes
        )
        batch = service.start(session, project, settings)
        session.commit()
        return batch.id
