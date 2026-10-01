# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Detection on the user's own bracketings and panoramas.

``tests/fixtures/fase11`` holds what the user shot for phase 11 with an A7 III
and the Tamron 28-75: two five-frame bracketings a third of a stop apart
(07309-07313 by ISO, 07314-07318 by shutter) and two panoramas shot as bursts
while turning, 17 portrait frames across a garden (07319-07335) and 19 frames
tilting up from the ground at the photographer's feet (07336-07354). Every
threshold that was chosen on synthetic sequences missed at least one of them
the first time; this keeps them found. The merges themselves were checked by
eye: they are too slow for the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ape.culling.burst import visual_signature
from ape.culling.features import camera_details
from ape.merge.detect import BracketShot, detect_brackets, exposure_brightness
from ape.merge.detect_pano import PanoShot, detect_panoramas
from ape.merge.pano_select import select_frames
from ape.raw.embedded import read_embedded_preview
from ape.raw.metadata import read_metadata

pytestmark = pytest.mark.fixtures

FOLDER = Path(__file__).parent / "fixtures" / "fase11"


@pytest.fixture(scope="module")
def shots():
    files = sorted(FOLDER.glob("*.ARW"))
    if len(files) != 46:
        pytest.skip("i file della Fase 11 non sono in tests/fixtures/fase11")
    names, brackets, panos, previews = [], [], [], []
    for index, path in enumerate(files):
        meta = read_metadata(path)
        preview = read_embedded_preview(path).image
        camera = camera_details(meta)
        bracket = BracketShot(
            id=index, shot_at=meta.shot_at, order=(index,), focal_length=meta.focal_length,
            aperture=meta.aperture, iso=meta.iso, shutter=meta.shutter,
            exposure_bias=meta.exposure_bias, release_mode=camera["release_mode"],
            signature=visual_signature(preview), camera_bracket=camera["camera_bracket"],
        )
        names.append(path.stem[3:])
        brackets.append(bracket)
        panos.append(PanoShot(id=index, shot_at=meta.shot_at, order=(index,),
                              camera=meta.camera_model, focal_length=meta.focal_length,
                              brightness=exposure_brightness(bracket)))
        previews.append(preview)
    return names, brackets, panos, previews


def test_the_users_merges_are_all_found(shots):
    names, brackets, panos, previews = shots
    found = detect_brackets(brackets)
    assert [[names[i] for i in g.members] for g in found] == [
        [f"0{n}" for n in range(7309, 7314)], [f"0{n}" for n in range(7314, 7319)],
    ]
    assert [names[g.reference] for g in found] == ["07309", "07314"]  # the metered frames
    taken = {i for g in found for i in g.members}

    panoramas = detect_panoramas(panos, lambda i: previews[i], excluded=taken)
    assert [(names[g.members[0]], names[g.members[-1]], len(g.members)) for g in panoramas] == [
        ("07319", "07335", 17), ("07336", "07354", 19),
    ]
    assert [g.reasons["axis"] for g in panoramas] == ["orizzontale", "verticale"]


def test_a_sweep_is_stitched_from_a_few_of_its_frames(shots):
    names, _, _, previews = shots
    for first, last, reference, expected in (
        (10, 26, 18, ["07319", "07321", "07327", "07335"]),
        (27, 45, 36, ["07336", "07339", "07345", "07349", "07354"]),
    ):
        frames = previews[first : last + 1]
        kept = select_frames(len(frames), reference - first, lambda i, f=frames: f[i])
        assert [names[first + i] for i in kept] == expected
