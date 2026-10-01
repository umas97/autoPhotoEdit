# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The CLIP scene embedding of section 8.3, on the CPU through ONNX Runtime.

Optional by construction (section 17): until the user asks for it the model is
not on disk, :func:`available` says so, and every caller carries on with the
engineered features alone. When it is there, it is loaded on the first photo a
worker analyses -- never at start-up -- and dropped again after five minutes
without use (section 26), which ``jobs/worker.py`` checks each time the queue
comes up empty.

One thread per session: the parallelism is fourteen workers, one photo each
(section 12). Measured on the target machine: 29 ms per photo on one core, and
135 MB of resident memory per process holding the session -- 188 MB with ONNX
Runtime's memory arena, which is switched off because it buys nothing at batch
size one. Fourteen workers holding it cost 1.9 GB, which is why it is released
when idle.

Output: 512 float32, L2-normalised, so the cosine distance the clustering uses
is ``1 - dot``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from ..models_registry import feature_status, model_path

__all__ = [
    "EMBEDDING_DIM",
    "available",
    "decode_embedding",
    "embed",
    "encode_embedding",
    "release_if_idle",
]

_log = logging.getLogger(__name__)

EMBEDDING_DIM = 512

#: The preprocessing CLIP was trained with (``preprocessor_config.json`` of the
#: pinned revision): shortest side to 224, bicubic, centre crop, these means.
_SIZE = 224
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

#: Section 26: a model nobody used for five minutes is memory taken from the
#: next decode.
IDLE_UNLOAD_S = 300.0

_lock = threading.Lock()
_session: Any = None
_last_used = 0.0


def available() -> bool:
    ok, _reason, _entry = feature_status("embedding")
    return ok


def _load() -> Any:
    global _session
    if _session is None:
        import onnxruntime as ort

        _ok, _reason, entry = feature_status("embedding")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        # The arena pre-reserves for batches this code never sends: 53 MB more
        # per process, measured, for the same 29 ms.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        _session = ort.InferenceSession(
            str(model_path(entry)), options, providers=["CPUExecutionProvider"]
        )
        _log.info("modello di scena caricato")
    return _session


def release_if_idle(now: float | None = None) -> bool:
    """Drop the session if it has been idle long enough. True if it was dropped."""
    global _session
    with _lock:
        if _session is None:
            return False
        if (now if now is not None else time.monotonic()) - _last_used < IDLE_UNLOAD_S:
            return False
        _session = None
    _log.info("modello di scena scaricato dalla memoria dopo %d s di inattività", IDLE_UNLOAD_S)
    return True


def _preprocess(image: np.ndarray) -> np.ndarray:
    """CLIP's own preprocessing, with Pillow's bicubic as its processor uses.

    Pillow's filters widen with the reduction, so a 2048 px proxy is
    antialiased on the way to 224; OpenCV's bicubic is not, and aliases. The
    difference is not cosmetic: measured on the user's frames, embeddings from
    OpenCV's bicubic agree with Pillow's to a cosine of only 0.93 median, 0.83
    worst.
    """
    from PIL import Image

    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    scale = _SIZE / min(height, width)
    size = (max(_SIZE, round(width * scale)), max(_SIZE, round(height * scale)))
    resized = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BICUBIC))
    top = (resized.shape[0] - _SIZE) // 2
    left = (resized.shape[1] - _SIZE) // 2
    crop = resized[top : top + _SIZE, left : left + _SIZE].astype(np.float32) / 255.0
    normalised = (crop - _MEAN) / _STD
    return np.ascontiguousarray(normalised.transpose(2, 0, 1)[None])


def embed(image: np.ndarray) -> np.ndarray | None:
    """The embedding of an upright sRGB image, or ``None`` if the model is absent."""
    global _last_used
    if not available():
        return None
    batch = _preprocess(image)
    with _lock:
        session = _load()
        _last_used = time.monotonic()
        output = session.run(None, {session.get_inputs()[0].name: batch})[0][0]
    vector = np.asarray(output, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def encode_embedding(vector: np.ndarray) -> bytes:
    return vector.astype("<f4").tobytes()


def decode_embedding(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    vector = np.frombuffer(blob, dtype="<f4")
    return vector.copy() if len(vector) == EMBEDDING_DIM else None
