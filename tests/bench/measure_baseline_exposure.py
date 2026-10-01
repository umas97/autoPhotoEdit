# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure where Sony puts middle grey, in stops below sensor saturation.

``raw/decode.py`` anchors the exposure with ``BASELINE_EXPOSURE_EV``: the number
of stops between the level LibRaw normalises to 1.0 (sensor saturation) and the
scene-linear middle grey our tone mapping is built around. The default is
derived from ISO 12232, which puts the metered grey at 10/78 of the saturation
exposure. That is a standard, not a measurement, and every manufacturer keeps a
different margin above white.

This script measures the real number on real files, by asking the camera. Each
ARW carries the JPEG the camera itself would have produced; the tone curve
inside that JPEG is Sony's own statement about which sensor level is middle
grey. So: find the pixels the camera renders at middle grey, look at what the
linear decode has there, and take the ratio.

Usage::

    uv run python tests/bench/measure_baseline_exposure.py [files or dir]

Defaults to ``tests/fixtures/``. Prints one line per file and a summary; it
writes nothing anywhere and opens every file read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from ape.pipeline.colorspace import MIDDLE_GREY, display_decode, luminance  # noqa: E402
from ape.raw.decode import BASELINE_EXPOSURE_EV, decode_linear  # noqa: E402
from ape.raw.metadata import read_metadata  # noqa: E402
from ape.safety import register_protected_root  # noqa: E402

#: Half-width of the window around middle grey, in display-linear units. Wide
#: enough to catch a useful number of pixels in any scene, narrow enough that
#: the camera's curve is straight across it.
_WINDOW = 0.02

#: Long edge both images are brought to before they are compared. The embedded
#: JPEG is a reduced rendering anyway, and matching detail is not the point --
#: matching tone is.
_COMPARE_EDGE = 512

#: Below this many matching pixels the median means nothing: the scene simply
#: has no midtones, and the file cannot answer the question.
_MIN_PIXELS = 2000


def _embedded_jpeg(path: Path) -> np.ndarray | None:
    """The camera's own rendering, as float32 sRGB in [0, 1]."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            return None
        if thumb.format is not rawpy.ThumbFormat.JPEG:
            return None
        buffer = np.frombuffer(thumb.data, dtype=np.uint8)

    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255.0)


def _resized(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    scale = _COMPARE_EDGE / max(height, width)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def measure(path: Path) -> tuple[float, int] | None:
    """Stops between saturation and middle grey for one file, and the sample count."""
    camera_jpeg = _embedded_jpeg(path)
    if camera_jpeg is None:
        return None

    decoded = decode_linear(path, half_size=True)
    scene = _resized(decoded.rgb)
    rendered = _resized(camera_jpeg)
    if scene.shape[:2] != rendered.shape[:2]:
        # The embedded JPEG is sometimes the full 3:2 frame and sometimes a
        # 16:9 crop; a shape mismatch means we would be comparing different
        # parts of the picture.
        rendered = cv2.resize(
            rendered, (scene.shape[1], scene.shape[0]), interpolation=cv2.INTER_AREA
        )

    display = luminance(display_decode(rendered))
    linear = luminance(np.clip(scene, 0.0, None))

    at_grey = np.abs(display - np.float32(MIDDLE_GREY)) < _WINDOW
    # Clipped or nearly black pixels say nothing about the camera's curve.
    usable = at_grey & (linear > 1e-4) & (linear < 0.9)
    count = int(np.count_nonzero(usable))
    if count < _MIN_PIXELS:
        return None

    grey_level = float(np.median(linear[usable]))
    return float(np.log2(MIDDLE_GREY / grey_level)), count


def main(argv: list[str]) -> int:
    targets = [Path(a).expanduser() for a in argv[1:]] or [
        Path(__file__).resolve().parents[1] / "fixtures"
    ]
    files: list[Path] = []
    for target in targets:
        if target.is_dir():
            files.extend(sorted(p for p in target.glob("*") if p.suffix.lower() == ".arw"))
        elif target.is_file():
            files.append(target)
    if not files:
        print("nessun file ARW da misurare", file=sys.stderr)
        return 1

    for parent in {f.parent for f in files}:
        register_protected_root(parent)

    print(f"{'file':<16}{'fotocamera':<18}{'ISO':>6}{'EV baseline':>13}{'pixel':>9}")
    results: list[tuple[str, float]] = []
    for path in files:
        meta = read_metadata(path)
        outcome = measure(path)
        camera = (meta.camera or "?").replace("SONY ", "")
        if outcome is None:
            print(f"{path.name:<16}{camera:<18}{meta.iso or 0:>6}{'--':>13}{'-':>9}")
            continue
        ev, count = outcome
        print(f"{path.name:<16}{camera:<18}{meta.iso or 0:>6}{ev:>13.3f}{count:>9}")
        results.append((camera, ev))

    if not results:
        print("\nnessun file utilizzabile: nessuno ha abbastanza mezzitoni")
        return 1

    values = np.array([ev for _, ev in results])
    print(
        f"\n{len(values)} file: mediana {np.median(values):+.3f} EV, "
        f"media {values.mean():+.3f} EV, deviazione {values.std():.3f}, "
        f"intervallo {values.min():+.3f}..{values.max():+.3f}"
    )
    for camera in sorted({c for c, _ in results}):
        per_body = np.array([ev for c, ev in results if c == camera])
        print(f"  {camera:<16} {len(per_body):>3} file, mediana {np.median(per_body):+.3f} EV")
    print(f"\nvalore attualmente in uso (ISO 12232): {BASELINE_EXPOSURE_EV:+.3f} EV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
