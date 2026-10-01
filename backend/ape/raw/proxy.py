# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Proxies: the reduced images the interface actually works on.

The user never waits for a 24 MP render. Everything the UI shows comes from a
2048 px proxy, and there are two of them because they answer two questions.

**The browsing proxy** is a JPEG on disk, one per photo, written once and
reused. It is what the grid, the culling screen and the filmstrip display. A
few hundred kilobytes each: a thousand photos cost a fraction of a gigabyte,
which fits the cache budget of section 20.3 with room to spare.

**The editing proxy** is the decoded frame itself -- linear, scene-referred,
2048 px -- handed to a ``StageRenderer`` when the user opens one photo. It is
not written to disk: at 16 MB per photo a catalogue's worth would swamp the
cache, and regenerating it costs one half-size decode, which section 12 budgets
at 1.2 s. It lives in memory for as long as that photo is open, and the stage
cache underneath it is what keeps a slider under the 150 ms of section 10.

Decoding at ``half_size`` is what makes both cheap: one pixel per sensor site,
no demosaicing, a quarter of the work and a quarter of the memory. At 24 MP that
is 3000x2000, still half again more than the 2048 px the proxy needs, so the
detail thrown away is detail the proxy could not show anyway.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..pipeline.filters import resize_long_edge
from ..pipeline.params import EditParams, neutral_params
from ..pipeline.render import RenderOptions, render
from .decode import DecodedRaw, decode_linear

__all__ = [
    "PROXY_QUALITY",
    "PROXY_VERSION",
    "is_current",
    "ProxyResult",
    "build_proxy",
    "editing_proxy",
    "proxy_path_for",
]

#: Bumped when what a proxy shows changes, so that the old files are
#: recognised as stale by their name instead of being served. Version 2 is the
#: lens-corrected proxy of phase 5: the straightening detector reads it, and it
#: must show the straight lines the lens bent.
PROXY_VERSION = 2

#: Quality of the browsing JPEG. Ninety is where the artefacts stop being
#: visible on a photograph at 100%; the proxy is never the exported file, so
#: spending more on it buys nothing.
PROXY_QUALITY = 90


def proxy_path_for(identity: str, *, long_edge: int | None = None) -> Path:
    """Where the browsing proxy of a photo lives.

    ``identity`` is the photo's content hash: a proxy belongs to the *content*,
    so a re-import under a new name reuses it instead of recomputing it, and
    two identical files share one.

    The path is sharded by the first two characters of the digest. A thousand
    photos in one directory is fine; a hundred thousand is not, and the cost of
    being right from the start is one line.
    """
    settings = get_settings()
    edge = long_edge or settings.proxy_long_edge
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return settings.proxy_dir / digest[:2] / f"{digest}-{edge}-v{PROXY_VERSION}.jpg"


def is_current(path: str | Path | None) -> bool:
    """Whether a proxy on record was written by this version of the pipeline."""
    return bool(path) and str(path).endswith(f"-v{PROXY_VERSION}.jpg") and Path(path).is_file()


@dataclass(frozen=True)
class ProxyResult:
    path: Path
    width: int
    height: int
    #: False when the file was already there and up to date.
    generated: bool
    #: The camera's white balance, known only when the RAW was decoded.
    as_shot: dict[str, float] | None = None


def build_proxy(
    raw_path: str | Path,
    identity: str,
    *,
    params: EditParams | None = None,
    long_edge: int | None = None,
    force: bool = False,
    lens_override: tuple[str, str] | None = None,
) -> ProxyResult:
    """Render the browsing JPEG for one RAW, or report the existing one.

    Args:
        raw_path: the source file. Opened read-only.
        identity: the photo's content hash, which names the proxy.
        params: development to bake in. Defaults to neutral, which is what the
            grid shows before a style has been predicted.
        long_edge: proxy size, defaulting to ``settings.proxy_long_edge``.
        force: regenerate even if the file is already there.
        lens_override: the lens profile the user associated by hand, if any.

    Returns:
        A :class:`ProxyResult` pointing at a JPEG under the cache directory.

    Raises:
        FileNotFoundError: if the RAW is not where the catalogue says it is.
        ValueError: if LibRaw cannot read it.
    """
    from ..export.image import ExportFormat, save_image

    settings = get_settings()
    edge = long_edge or settings.proxy_long_edge
    destination = proxy_path_for(identity, long_edge=edge)

    if destination.is_file() and not force:
        from PIL import Image

        with Image.open(destination) as existing:
            return ProxyResult(destination, existing.width, existing.height, generated=False)

    decoded = editing_proxy(raw_path, long_edge=edge, lens_override=lens_override)
    image = render(decoded, params or neutral_params(), RenderOptions(long_edge=edge))
    save_image(image, destination, ExportFormat.JPEG, quality=PROXY_QUALITY)
    height, width = image.shape[:2]
    as_shot = {
        "temperature_k": round(float(decoded.camera.as_shot_temperature_k), 1),
        "tint": round(float(decoded.camera.as_shot_tint), 2),
    }
    return ProxyResult(destination, width, height, generated=True, as_shot=as_shot)


def editing_proxy(
    raw_path: str | Path,
    *,
    long_edge: int | None = None,
    lens_override: tuple[str, str] | None = None,
) -> DecodedRaw:
    """Decode a RAW straight to proxy size, linear and scene-referred.

    The result is the input of a ``StageRenderer``: same colour space, same
    camera context and same baseline exposure as the full-resolution decode, so
    a preview computed from it and an export computed from the file agree --
    which is test 1 of section 13, and the promise the UI makes.
    """
    settings = get_settings()
    edge = long_edge or settings.proxy_long_edge
    decoded = decode_linear(raw_path, half_size=True, lens_override=lens_override)
    reduced = resize_long_edge(decoded.rgb, edge)
    if reduced is decoded.rgb:
        return decoded
    decoded.rgb = np.ascontiguousarray(reduced)
    return decoded
