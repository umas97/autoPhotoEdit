# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Put analysed photos straight into a catalogue, for the culling tests.

The culling analysis of a photo needs its embedded JPEG, and the importer's
test files are bytes with an ARW extension and no JPEG in them. So the tests of
everything *after* the analysis -- grouping, selection, the API -- insert rows
that look exactly like the ones ``jobs/handlers.run_cull`` writes, with the
features of a real analysis of a synthetic preview. The handler itself is
exercised on real files by the tests marked ``fixtures``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ape.culling.features import analyse_preview
from ape.db.models import Photo, PhotoKind, PhotoStatus, Project
from ape.raw.embedded import EmbeddedPreview

__all__ = ["AnalysedShot", "insert_analysed", "make_project"]


@dataclass
class AnalysedShot:
    image: np.ndarray
    shot_at: datetime
    shutter: float = 1 / 250
    aperture: float = 4.0
    iso: int = 100
    focal_length: float = 35.0
    exposure_bias: float | None = 0.0
    release_mode: int | None = None
    camera: str = "ILCE-7M3"


def make_project(session, folder: Path, name: str = "Cernita") -> Project:
    folder.mkdir(parents=True, exist_ok=True)
    project = Project(name=name, source_dir=str(folder), culling_enabled=True)
    session.add(project)
    session.flush()
    return project


def insert_analysed(session, project: Project, shots: list[AnalysedShot]) -> list[int]:
    """Rows as the culling job leaves them. Returns their ids, in order."""
    ids = []
    start = len(project.photos)
    for offset, shot in enumerate(shots):
        index = start + offset
        analysis = analyse_preview(EmbeddedPreview(shot.image, "embedded"))
        features = analysis.features
        features["camera"] = {
            "exposure_bias": shot.exposure_bias,
            "release_mode": shot.release_mode,
            "shot_number": index,
            "focus_point": None,
        }
        scores = analysis.scores
        photo = Photo(
            project_id=project.id,
            path=str(Path(project.source_dir) / f"DSC{index:05d}.ARW"),
            filename=f"DSC{index:05d}.ARW",
            hash=f"q:test:{uuid.uuid4().hex}",
            kind=PhotoKind.RAW,
            camera=shot.camera,
            shot_at=shot.shot_at,
            shutter=shot.shutter,
            aperture=shot.aperture,
            iso=shot.iso,
            focal_length=shot.focal_length,
            sharpness=scores.focus,
            motion_blur=None if scores.motion is None else 1.0 - scores.motion,
            exposure_score=scores.exposure,
            culling_features=features,
            status=PhotoStatus.CULLED,
        )
        session.add(photo)
        session.flush()
        ids.append(photo.id)
    return ids
