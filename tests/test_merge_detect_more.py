# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focus stacks and panoramas are found; bursts and different scenes are not.

Section 25.2. The sequences are built from the camera previews of the fixtures:
a focus stack sweeps a plane of focus across a real picture (blur growing with
the distance from it), a burst jitters, recompresses and shakes one frame of
it, a panorama is overlapping crops of it. The thresholds of
``merge/detect_stack.py`` were chosen on the same construction over all 24
previews; here a third of them keep it honest.

And the part of detection that is about the catalogue: in a project that went
straight to editing, the proposals appear without culling ever running, and
the panorama search is left to a worker.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import select

from ape.db.models import Job, JobKind, MergeGroup, MergeKind, Photo, PhotoKind, Project
from ape.merge.detect_pano import PanoShot, detect_panoramas, measure_overlap
from ape.merge.detect_stack import StackShot, detect_focus_stacks
from ape.merge.features import grouping_features
from ape.merge.pano_select import select_frames
from ape.merge.service import ensure_detected
from ape.raw.embedded import read_embedded_preview

T0 = datetime(2026, 2, 26, 12, 0, 0)


@pytest.fixture(scope="module")
def previews(raw_fixtures) -> list[np.ndarray]:
    return [read_embedded_preview(path).image for path in raw_fixtures[::3]]


def _jpeg(image: np.ndarray) -> np.ndarray:
    _, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def focus_sweep(image: np.ndarray, frames: int, axis: int = 0) -> list[np.ndarray]:
    """A stack: frame k sharp at depth k/frames, blur growing away from it."""
    levels = [image.astype(np.float32)] + [
        cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma) for sigma in (1, 2, 3, 4)
    ]
    length = image.shape[axis]
    position = np.linspace(0.0, 1.0, length)
    out = []
    for k in range(frames):
        sigma = np.clip(np.abs(position - (k + 0.5) / frames) * 8.0, 0.0, 4.0)
        low = np.floor(sigma).astype(int)
        high = np.minimum(low + 1, 4)
        t = (sigma - low).astype(np.float32)
        frame = np.empty_like(levels[0])
        for index in range(length):
            cut = slice(index, index + 1)
            line = (cut,) if axis == 0 else (slice(None), cut)
            a, b = levels[low[index]][line], levels[high[index]][line]
            frame[line] = a * (1 - t[index]) + b * t[index]
        out.append(_jpeg(np.clip(frame, 0, 255).astype(np.uint8)))
    return out


def burst(image: np.ndarray, frames: int, seed: int = 0) -> list[np.ndarray]:
    """A burst of a still subject: jitter, noise, recompression, one shaken frame."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(frames):
        shift = np.float32([[1, 0, rng.integers(-2, 3)], [0, 1, rng.integers(-2, 3)]])
        frame = cv2.warpAffine(image, shift, image.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
        frame = frame.astype(np.float32)
        if k == frames // 2:
            frame = cv2.GaussianBlur(frame, (0, 0), 1.2)
        frame += rng.normal(0, 2, frame.shape).astype(np.float32)
        out.append(_jpeg(np.clip(frame, 0, 255).astype(np.uint8)))
    return out


def _stack_shots(frames: list[np.ndarray], positions=None, start: int = 0) -> list[StackShot]:
    shots = []
    for k, frame in enumerate(frames):
        features = grouping_features(frame, None)
        shots.append(
            StackShot(
                id=start + k, shot_at=T0 + timedelta(seconds=2 * (start + k)), order=(start + k,),
                focal_length=90.0, aperture=5.6, iso=100, shutter=1 / 125,
                focus_position=None if positions is None else positions[k],
                signature=features["signature"], sharp_map=features["sharp_map"],
            )
        )
    return shots


def test_a_moving_plane_of_focus_is_a_stack_and_a_burst_is_not(previews):
    for image in previews:
        for frames, axis in ((5, 0), (5, 1), (4, 0)):
            found = detect_focus_stacks(_stack_shots(focus_sweep(image, frames, axis)))
            assert [len(f.members) for f in found] == [frames]
            assert found[0].kind == "focus_stack" and found[0].reasons["focus_moves"]
        assert detect_focus_stacks(_stack_shots(burst(image, 5))) == []


def test_the_focus_position_alone_is_evidence(previews):
    frames = burst(previews[0], 4)  # the pixels say nothing
    assert detect_focus_stacks(_stack_shots(frames, positions=[255] * 4)) == []
    found = detect_focus_stacks(_stack_shots(frames, positions=[40, 52, 67, 80]))
    assert len(found) == 1 and found[0].reasons["focus_positions"]


def test_a_stack_needs_fixed_exposure_and_framing(previews):
    frames = focus_sweep(previews[1], 4)
    shots = _stack_shots(frames)
    changed = [shots[0], shots[1], StackShot(**{**shots[2].__dict__, "shutter": 1 / 60}), shots[3]]
    assert detect_focus_stacks(changed) == []
    other = _stack_shots(focus_sweep(previews[2], 2), start=2)
    assert detect_focus_stacks(shots[:2] + other) == []


def _crops(image: np.ndarray, width: int, step: int) -> list[np.ndarray]:
    return [image[:, x : x + width] for x in range(0, image.shape[1] - width + 1, step)]


def test_overlapping_frames_are_a_panorama_and_different_scenes_are_not(previews):
    # The first landscape preview is left out: its right third is bokeh, with
    # nothing for ORB to hold on to -- a panorama of it would not stitch either.
    wide = [p for p in previews if p.shape[1] > p.shape[0]][1:]
    assert len(wide) >= 3
    for image in wide:
        frames = _crops(image, 700, 420)  # 40% overlap
        shots = [
            PanoShot(id=k, shot_at=T0 + timedelta(seconds=2 * k), order=(k,), camera="ILCE-7M3",
                     focal_length=28.0, brightness=0.0)
            for k in range(len(frames))
        ]
        found = detect_panoramas(shots, lambda i, frames=frames: frames[i])
        assert [f.members for f in found] == [tuple(range(len(frames)))]
        assert found[0].kind == "panorama" and found[0].reasons["axis"] == "orizzontale"

    # Consecutive, same settings, but different pictures: not a panorama.
    scenes = [s for s in previews]
    shots = [
        PanoShot(id=k, shot_at=T0 + timedelta(seconds=2 * k), order=(k,), camera="ILCE-7M3",
                 focal_length=28.0, brightness=0.0)
        for k in range(len(scenes))
    ]
    assert detect_panoramas(shots, lambda i: scenes[i]) == []
    # The same picture twice is a burst, not a panorama: the overlap is too large.
    assert measure_overlap(image, image).overlap > 0.9
    assert detect_panoramas(shots[:2], lambda i: image) == []


def _pano_shots(count: int, gap_s: float = 0.25) -> list[PanoShot]:
    return [
        PanoShot(id=k, shot_at=T0 + timedelta(seconds=gap_s * k), order=(k,), camera="ILCE-7M3",
                 focal_length=28.0, brightness=0.0)
        for k in range(count)
    ]


def test_a_panorama_shot_as_a_burst_is_one_panorama(previews):
    """The user's panoramas were bursts while turning: 17 and 19 frames, 85-95%
    overlap between neighbours (``tests/fixtures/fase11``). Pair by pair the
    same picture twice; as a sequence, a panorama."""
    wide = [p for p in previews if p.shape[1] > p.shape[0]][1:]
    for image in wide:
        frames = _crops(image, 1000, 60)  # 94% overlap, 60% of a frame in all
        found = detect_panoramas(_pano_shots(len(frames)), lambda i, f=frames: f[i])
        assert [f.members for f in found] == [tuple(range(len(frames)))]
        assert found[0].reasons["sweep"] is True

        # The same burst that travels a fifth of a frame is not one: its first
        # and last frame overlap by 80%.
        short = frames[:4]
        assert detect_panoramas(_pano_shots(4), lambda i, f=short: f[i]) == []


def test_a_sweep_keeps_the_frames_that_cover_it(previews):
    image = [p for p in previews if p.shape[1] > p.shape[0]][1]
    sweep = _crops(image, 1000, 60)  # 11 frames, 94% overlap
    assert select_frames(len(sweep), 5, lambda i: sweep[i]) == [0, 5, 10]
    # A panorama shot the ordinary way loses nothing, ...
    ordinary = _crops(image, 700, 420)
    assert select_frames(len(ordinary), 1, lambda i: ordinary[i]) == list(range(len(ordinary)))
    # ... nor frames that do not move, nor frames without a preview.
    assert select_frames(4, 1, lambda i: image) == [0, 1, 2, 3]
    assert select_frames(4, 1, lambda i: None) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# A project without culling.


def _insert(session, project, image, seconds, shutter, index):
    photo = Photo(
        project_id=project.id, path=str(Path(project.source_dir) / f"DSC{index:05d}.ARW"),
        filename=f"DSC{index:05d}.ARW", hash=f"q:{index}", kind=PhotoKind.RAW,
        camera="ILCE-7M3", shot_at=T0 + timedelta(seconds=seconds), shutter=shutter,
        aperture=8.0, iso=100, focal_length=35.0,
        culling_features=grouping_features(image, None),
    )
    session.add(photo)
    return photo


def test_a_project_without_culling_gets_its_proposals(catalog, previews, tmp_path):
    with catalog() as session:
        project = Project(name="diretto", source_dir=str(tmp_path), culling_enabled=False)
        session.add(project)
        session.flush()
        base = previews[3]
        for index, (ev, shutter) in enumerate(((0, 1 / 250), (-2, 1 / 1000), (2, 1 / 60))):
            frame = np.clip(base.astype(np.float32) * 2.0 ** (ev / 2.2), 0, 255).astype(np.uint8)
            _insert(session, project, frame, index, shutter, index)
        _insert(session, project, previews[4], 30, 1 / 250, 3)
        session.flush()

        assert ensure_detected(session, project)
        groups = session.scalars(select(MergeGroup)).all()
        assert [(g.kind, len(g.members)) for g in groups] == [(MergeKind.HDR, 3)]
        jobs = session.scalars(select(Job).where(Job.kind == JobKind.DETECT_MERGES)).all()
        assert len(jobs) == 1  # the panoramas are a worker's business
        session.commit()

    with catalog() as session:
        project = session.get(Project, project.id)
        # Nothing new was measured, and the panorama search is still queued.
        assert not ensure_detected(session, project)
