# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The pure parts of phase 5: crop proposal, scene features, clustering, downloads.

No catalogue and no RAW here -- ``test_analysis_api.py`` has those. What is
checked is the behaviour each module promises in its docstring, on inputs whose
right answer is known.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from ape.analysis import cluster, crop, embed, scene
from ape.downloads import Download, DownloadError, DownloadState


def _subject_frame(cx: float, cy: float, width: int = 600, height: int = 400) -> np.ndarray:
    """A muted, blurred background with one sharp, saturated subject."""
    rng = np.random.default_rng(0)
    frame = np.full((height, width, 3), (110, 120, 105), dtype=np.uint8)
    frame = cv2.GaussianBlur(
        (frame + rng.normal(0, 6, frame.shape)).clip(0, 255).astype(np.uint8), (0, 0), 6
    )
    centre = (int(cx * width), int(cy * height))
    cv2.circle(frame, centre, int(0.07 * width), (200, 40, 30), -1)
    for i in range(12):  # texture, so the subject is also the sharp part
        cv2.line(frame, (centre[0] - 30, centre[1] - 30 + 5 * i),
                 (centre[0] + 30, centre[1] - 30 + 5 * i), (250, 230, 220), 1)
    return frame


# --- crop -------------------------------------------------------------------


def test_an_off_centre_subject_gets_a_crop_that_keeps_it_on_a_third():
    frame = _subject_frame(0.78, 0.5)
    proposal = crop.propose_crop(frame)
    assert proposal is not None
    x0, y0 = proposal.x * 600, proposal.y * 400
    x1, y1 = x0 + proposal.width * 600, y0 + proposal.height * 400
    radius = 0.07 * 600
    # The whole subject is inside: a proposal never cuts through it.
    assert x0 <= 0.78 * 600 - radius and x1 >= 0.78 * 600 + radius
    assert y0 <= 200 - radius and y1 >= 200 + radius
    # And it sits near a power point of the crop.
    u = (0.78 * 600 - x0) / (x1 - x0)
    v = (200 - y0) / (y1 - y0)
    assert min(abs(u - 1 / 3), abs(u - 2 / 3)) < 0.12
    assert min(abs(v - 1 / 3), abs(v - 2 / 3), abs(v - 0.5)) < 0.2
    assert proposal.score > proposal.baseline


def test_a_frame_with_nothing_to_crop_for_gets_no_proposal():
    rng = np.random.default_rng(4)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (400, 600, 3), dtype=np.uint8), (0, 0), 2)
    assert crop.propose_crop(texture) is None


def test_proposals_stay_inside_the_frame_and_respect_the_aspect():
    frame = _subject_frame(0.2, 0.3)
    proposal = crop.propose_crop(frame, aspects=("1:1",))
    if proposal is None:
        pytest.skip("nessuna proposta 1:1 migliore dell'originale per questa scena")
    assert proposal.x >= 0 and proposal.x + proposal.width <= 1 + 1e-9
    assert proposal.y >= 0 and proposal.y + proposal.height <= 1 + 1e-9
    assert proposal.width * 600 == pytest.approx(proposal.height * 400, abs=2)


# --- scene features ---------------------------------------------------------


def test_scene_features_have_the_declared_layout_and_round_trip():
    frame = _subject_frame(0.5, 0.5)
    vector = scene.scene_features(
        frame, scene.SceneInputs(iso=100, aperture=8.0, shutter=1 / 125, focal_length=35,
                                 as_shot_temperature_k=5500, as_shot_tint=0)
    )
    assert vector.shape == (len(scene.FEATURE_NAMES),)
    named = dict(zip(scene.FEATURE_NAMES, vector.tolist(), strict=True))
    # Sunny 16's cousin: f/8 at 1/125 and ISO 100 is EV 13.
    assert named["ev100"] == pytest.approx(math.log2(64 * 125), abs=1e-4)
    assert named["cct_mired"] == pytest.approx(1e6 / 5500, rel=1e-4)
    assert sum(named[f"hue_{i}"] for i in range(8)) == pytest.approx(1.0, abs=1e-4)
    assert named["outdoor"] > 0.8
    np.testing.assert_array_equal(scene.decode_features(scene.encode_features(vector)), vector)


def test_unknown_exif_is_nan_not_zero():
    vector = scene.scene_features(_subject_frame(0.5, 0.5), scene.SceneInputs())
    named = dict(zip(scene.FEATURE_NAMES, vector.tolist(), strict=True))
    assert math.isnan(named["ev100"]) and math.isnan(named["cct_mired"])


def test_features_of_another_version_are_not_reinterpreted():
    blob = scene.encode_features(np.zeros(len(scene.FEATURE_NAMES), dtype=np.float32))
    stale = blob[:4] + bytes([scene.SCENE_FEATURES_VERSION + 1]) + blob[5:]
    assert scene.decode_features(stale) is None
    assert scene.decode_features(None) is None


# --- clustering -------------------------------------------------------------


def _items_from(groups: list[list[np.ndarray]], start: datetime, gap_s: float = 5.0):
    items, n = [], 0
    for g, members in enumerate(groups):
        for vector in members:
            when = start + timedelta(minutes=10 * g, seconds=gap_s * n)
            items.append(cluster.ClusterItem(n, None, vector, when))
            n += 1
    return items


def test_features_cluster_into_their_scenes_with_the_medoid_first():
    rng = np.random.default_rng(1)
    size = len(scene.FEATURE_NAMES)
    centres = [rng.normal(0, 3, size) for _ in range(3)]
    groups = [[c + rng.normal(0, 0.05, size) for _ in range(5)] for c in centres]
    items = _items_from(groups, datetime(2026, 2, 26, 14, 0))
    result = cluster.cluster_photos(items)
    assert result.basis == "features"
    labels = [result.labels[i.photo_id] for i in items]
    assert labels == [1] * 5 + [2] * 5 + [3] * 5  # numbered in shooting order
    for label in (1, 2, 3):
        assert sorted(r for pid, r in result.ranks.items() if result.labels[pid] == label) == list(
            range(5)
        )


def test_embeddings_are_used_when_every_photo_has_one():
    rng = np.random.default_rng(2)
    centres = [rng.normal(0, 1, 512) for _ in range(2)]
    items = []
    for n in range(10):
        vector = centres[n // 5] + rng.normal(0, 0.05, 512)
        items.append(cluster.ClusterItem(n, vector / np.linalg.norm(vector), None, None))
    result = cluster.cluster_photos(items)
    assert result.basis == "embedding"
    assert len(set(result.labels.values())) == 2


def test_time_splits_two_visits_to_the_same_place():
    """Same content, an hour apart: the review corrects lights, not places."""
    rng = np.random.default_rng(3)
    size = len(scene.FEATURE_NAMES)
    centre = rng.normal(0, 3, size)
    start = datetime(2026, 2, 26, 9, 0)
    items = [
        cluster.ClusterItem(n, None, centre + rng.normal(0, 0.001, size),
                            start + timedelta(hours=2 * (n // 4), seconds=n))
        for n in range(8)
    ]
    near = cluster.THRESHOLD_FEATURES
    try:
        # Tighten the cut to what the content alone does not cross.
        cluster.THRESHOLD_FEATURES = cluster.TIME_WEIGHT * 0.9
        result = cluster.cluster_photos(items)
    finally:
        cluster.THRESHOLD_FEATURES = near
    assert len(set(result.labels.values())) == 2


def test_one_light_one_place_is_one_scene():
    """The noise of features that do not vary must not become distance.

    Regression: with the spread taken from the project alone, eight frames of
    one scene with 0.01 of jitter came out as eight scenes.
    """
    rng = np.random.default_rng(5)
    size = len(scene.FEATURE_NAMES)
    centre = rng.normal(0, 3, size)
    start = datetime(2026, 2, 26, 9, 0)
    items = [
        cluster.ClusterItem(n, None, centre + rng.normal(0, 0.01, size),
                            start + timedelta(seconds=20 * n))
        for n in range(8)
    ]
    assert len(set(cluster.cluster_photos(items).labels.values())) == 1


def test_nothing_to_cluster_is_not_an_error():
    assert cluster.cluster_photos([]).labels == {}
    assert cluster.cluster_photos([cluster.ClusterItem(7, None, None, None)]).labels == {}


# --- embedding and downloads (section 17, test 18) --------------------------


def test_the_embedding_is_simply_absent_without_the_model(xdg_home):
    assert embed.available() is False
    assert embed.embed(_subject_frame(0.5, 0.5)) is None


def test_a_corrupt_model_on_disk_is_refused(xdg_home):
    from ape.models_registry import Unavailable, entry_for, feature_status, model_path

    entry = entry_for("embedding")
    path = model_path(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"non sono un modello")
    ok, reason, _ = feature_status("embedding")
    assert ok is False and reason is Unavailable.CHECKSUM_MISMATCH
    assert embed.embed(_subject_frame(0.5, 0.5)) is None


def _served(tmp_path: Path, content: bytes) -> str:
    source = tmp_path / "remoto.bin"
    source.write_bytes(content)
    return source.as_uri()


def test_a_download_is_kept_only_if_its_checksum_matches(xdg_home, tmp_path):
    content = b"x" * 200_000
    good = hashlib.sha256(content).hexdigest()
    target = tmp_path / "modelli" / "m.onnx"
    download = Download(_served(tmp_path, content), target, sha256=good)
    assert download.run() == target
    assert target.read_bytes() == content
    assert download.state is DownloadState.DONE and download.fraction == 1.0


def test_a_tampered_download_leaves_nothing_behind(xdg_home, tmp_path):
    target = tmp_path / "modelli" / "m.onnx"
    download = Download(_served(tmp_path, b"falso"), target, sha256="0" * 64)
    with pytest.raises(DownloadError, match="checksum"):
        download.run()
    assert download.state is DownloadState.FAILED
    assert not target.exists()
    assert not target.with_name("m.onnx.part").exists()


def test_no_network_is_a_readable_failure(xdg_home, tmp_path):
    download = Download("http://127.0.0.1:9/nulla", tmp_path / "m.onnx", sha256="0" * 64)
    with pytest.raises(DownloadError, match="rete"):
        download.run()
    assert not (tmp_path / "m.onnx").exists()


def test_a_download_refuses_to_write_into_a_source_folder(xdg_home, tmp_path):
    from ape.safety import SourceWriteError, register_protected_root, unregister_protected_root

    source = tmp_path / "scheda"
    source.mkdir()
    register_protected_root(source)
    try:
        with pytest.raises(SourceWriteError):
            Download(_served(tmp_path, b"x"), source / "m.onnx").run()
    finally:
        unregister_protected_root(source)


def test_an_unpinned_model_cannot_be_downloaded(xdg_home):
    from ape.models_registry import start_download

    with pytest.raises(ValueError, match="checksum"):
        start_download("faces")  # YuNet has no pinned hash yet
