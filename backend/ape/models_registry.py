# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The ONNX models: where they come from, what they are for, whether they are here.

Section 17 makes this a declaration rather than code: every model the program
may ever use is listed with its origin, its licence and the feature that needs
it. No model ships with the program and none is downloaded at startup -- the
user who never switches on faces, aesthetics or segmentation never downloads a
byte.

**A model is available only when its file is on disk and matches its pinned
SHA-256.** An entry without a pinned hash cannot be verified, so it is not
available, whatever is on disk: section 17 makes the check mandatory, and a
model that cannot be checked is a model that could be anything.

A pin is the SHA-256 the *publisher* declares for the file at a fixed revision
-- Hugging Face exposes it as the LFS object id -- checked against a download
of that revision, never a hash computed from whatever a mirror happened to
serve. The download itself (:func:`start_download`) happens only when the user
asks for the feature, with progress and cancellation, and a file that does not
match is deleted before anything can load it.

Nothing here imports ``onnxruntime``: asking whether a feature is available
must not load a runtime of several hundred megabytes (section 26).
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from .config import get_settings

if TYPE_CHECKING:
    from .downloads import Download

__all__ = [
    "MODELS",
    "ModelEntry",
    "Unavailable",
    "active_download",
    "entry_for",
    "feature_status",
    "model_path",
    "start_download",
]


class Unavailable(StrEnum):
    """Why a feature cannot run. The interface translates these."""

    #: ``onnxruntime`` is not installed (the ``ml`` extra of ``pyproject.toml``).
    RUNTIME_MISSING = "runtime_missing"
    #: No verified checksum for the model yet, so it cannot be trusted.
    NOT_PINNED = "not_pinned"
    #: Pinned, but not downloaded.
    MODEL_MISSING = "model_missing"
    #: On disk, and not the file that was pinned.
    CHECKSUM_MISMATCH = "checksum_mismatch"


@dataclass(frozen=True)
class ModelEntry:
    name: str
    #: The feature that needs it, as named in ``Project.culling_criteria`` and
    #: elsewhere: ``faces``, ``aesthetic``, ``embedding``, ``segment_person``,
    #: ``segment_sky``.
    feature: str
    filename: str
    url: str
    licence: str
    size_mb: float
    #: ``None`` until the publisher's hash of a fixed revision has been checked.
    sha256: str | None = None
    #: The file itself, at a fixed revision: a moving ``main`` would make the
    #: pin fail the day upstream re-exports the model.
    download_url: str | None = None
    size_bytes: int | None = None
    #: Shown next to the toggle. The NIMA weights are the reason this exists.
    notice: str | None = None


#: Section 17's table. The URLs are the upstream sources the licences refer
#: to; ``sha256`` is filled in when an entry is verified and pinned.
MODELS: tuple[ModelEntry, ...] = (
    # The visual tower of OpenAI's CLIP ViT-B/32 (MIT), exported to ONNX and
    # quantised to int8 by Xenova. 88.6 MB, inside the 120 MB per model of
    # section 26, and 30 ms per image on one core of the target machine.
    ModelEntry(
        name="clip-vit-b32-visual-int8",
        feature="embedding",
        filename="clip-vit-b32-visual-int8.onnx",
        url="https://huggingface.co/Xenova/clip-vit-base-patch32",
        licence="MIT",
        size_mb=88.6,
        sha256="0ab0c1b3ace708e539633af1744d5a95247fe4e14d3e08ff197ef82a6cb9bd93",
        download_url=(
            "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/"
            "d15189d7028b43f1d3e65039190477f6af591c2a/onnx/vision_model_int8.onnx"
        ),
        size_bytes=88_648_877,
    ),
    ModelEntry(
        name="yunet",
        feature="faces",
        filename="face_detection_yunet_2023mar.onnx",
        url="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        licence="MIT",
        size_mb=0.3,
    ),
    ModelEntry(
        name="nima-mobilenet",
        feature="aesthetic",
        filename="nima-mobilenet.onnx",
        url="https://github.com/idealo/image-quality-assessment",
        licence="Apache-2.0 (code); weights derived from the AVA dataset, research only",
        size_mb=14.0,
        notice="ava_research_only",
    ),
    # Segmentation on request (section 6.3), two models because none of the
    # salient-object ones of section 17 (U^2-Netp, BiRefNet) tells a sky, a
    # person or skin apart.
    #
    # People and skin: MediaPipe's selfie multiclass segmenter (Apache-2.0),
    # converted to ONNX. 16.5 MB, 27 ms on one core; six classes at 256 px.
    ModelEntry(
        name="mediapipe-selfie-multiclass-256",
        feature="segment_person",
        filename="selfie-multiclass-256.onnx",
        url="https://huggingface.co/senty-au/selfie_multiclass_256x256-ONNX",
        licence="Apache-2.0",
        size_mb=16.5,
        sha256="35ec1ecd9ee7f85073c99c00020b7f6751b69506eeacf683bc8665f6117f85b0",
        download_url=(
            "https://huggingface.co/senty-au/selfie_multiclass_256x256-ONNX/resolve/"
            "6db8421a7150ac20558f2c24675078eb3a1a04d0/onnx/model.onnx"
        ),
        size_bytes=16_454_560,
    ),
    # Sky: a U^2-Net trained for sky segmentation, FP16 (MIT as published, with
    # a doubt: the training data are not declared and the conversion is a third
    # party's -- models/LICENSES.md). 88 MB, 231 ms on one core, 320 px.
    ModelEntry(
        name="skyseg-u2net-fp16",
        feature="segment_sky",
        filename="skyseg-u2net-fp16.onnx",
        url="https://huggingface.co/voyagerfromeast/skyseg",
        licence="MIT (dati di addestramento non dichiarati)",
        size_mb=88.1,
        sha256="74d87f4a69378a610a6be662f859c38cfbdfdd75ff74bbfc54842965ed6fc9f7",
        download_url=(
            "https://huggingface.co/voyagerfromeast/skyseg/resolve/"
            "d76f97cf68f049654f087d78d071dfd7505b2561/skyseg_fp16.onnx"
        ),
        size_bytes=88_084_505,
        notice="skyseg_provenance",
    ),
    # The magic eraser's optional engine (docs/SPEC_rimozione.md 5):
    # LaMa (Apache-2.0), the OpenCV Zoo's int8 export of Carve/LaMa-ONNX.
    # 92.6 MB, 2 s at 512 px on four threads; trained on Places2, whose
    # images are for non-commercial research -- said next to the download.
    ModelEntry(
        name="lama-inpainting-int8",
        feature="inpaint",
        filename="inpainting-lama-2025jan.onnx",
        url="https://huggingface.co/opencv/inpainting_lama",
        licence="Apache-2.0 (addestrato su Places2)",
        size_mb=92.6,
        sha256="7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2",
        download_url=(
            "https://huggingface.co/opencv/inpainting_lama/resolve/"
            "aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f/inpainting_lama_2025jan.onnx"
        ),
        size_bytes=92_591_623,
        notice="places2_terms",
    ),
)


def model_path(entry: ModelEntry) -> Path:
    return get_settings().models_dir / entry.filename


@lru_cache(maxsize=16)
def _sha256_of(path: Path, size: int, mtime_ns: int) -> str:
    """Hash of a file, remembered for as long as its size and mtime hold.

    Keyed on both so that a model replaced on disk is hashed again, and a model
    that did not change is not re-read on every request.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_of(path, stat.st_size, stat.st_mtime_ns)


def feature_status(feature: str) -> tuple[bool, Unavailable | None, ModelEntry | None]:
    """Whether a feature can run here, and if not, the first reason why.

    Returns ``(available, reason, entry)``. Cheap enough to call on every
    request: the checksum is computed once per version of the file on disk.
    """
    entry = entry_for(feature)
    if entry is None:
        return False, Unavailable.NOT_PINNED, None
    if importlib.util.find_spec("onnxruntime") is None:
        return False, Unavailable.RUNTIME_MISSING, entry
    if entry.sha256 is None:
        return False, Unavailable.NOT_PINNED, entry
    path = model_path(entry)
    if not path.is_file():
        return False, Unavailable.MODEL_MISSING, entry
    if _sha256(path) != entry.sha256:
        return False, Unavailable.CHECKSUM_MISMATCH, entry
    return True, None, entry


def entry_for(feature: str) -> ModelEntry | None:
    return next((m for m in MODELS if m.feature == feature), None)


#: One download per feature at a time, kept after it ends so that the interface
#: can read how it went. Only the server process downloads.
_DOWNLOADS: dict[str, Download] = {}


def active_download(feature: str) -> Download | None:
    return _DOWNLOADS.get(feature)


def start_download(feature: str, on_done=None) -> Download:
    """Fetch the model of one feature in the background, on the user's request.

    Raises:
        LookupError: no model serves this feature.
        ValueError: the entry is not pinned, so there is nothing it could be
            verified against -- which section 17 forbids downloading at all.
    """
    from .downloads import Download, DownloadState

    entry = entry_for(feature)
    if entry is None:
        raise LookupError(f"nessun modello per la funzione {feature!r}")
    if entry.sha256 is None or entry.download_url is None:
        raise ValueError(f"il modello {entry.name} non ha un checksum verificato")
    current = _DOWNLOADS.get(feature)
    if current is not None and current.state is DownloadState.RUNNING:
        return current
    download = Download(
        url=entry.download_url,
        destination=model_path(entry),
        sha256=entry.sha256,
        expected_size=entry.size_bytes,
    )
    _DOWNLOADS[feature] = download
    download.start(on_done)
    return download
