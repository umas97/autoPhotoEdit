# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Segmentation on request: the sky, the people, the skin (docs/SPEC.md section 6.3).

Two models, pinned in ``models_registry``:

``segment_person``
    MediaPipe's selfie multiclass segmenter: NHWC ``[1, 256, 256, 3]``, RGB in
    0..1, the frame stretched to the square; six logits per pixel -- 0
    background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other. A person
    is everything but the background; skin is the two skin classes.
``segment_sky``
    A U^2-Net trained on skies: NCHW ``[1, 3, 320, 320]``, ImageNet
    normalisation; the first of its seven outputs is the fused map, already
    through its sigmoid.

Both run on the proxy -- the frame itself, lens-corrected and upright, at 2048
px -- and answer at their own few hundred pixels. The probability is brought
back to the proxy's size and **refined by a guided filter on the picture's own
luminance**, so that the edge follows the skyline or the shoulder instead of
the model's coarse grid. The result is stored as a mask raster
(``masks_store``), which the render resamples like any painted mask.

Two rules about the scene clean up what the models get wrong on the user's
frames (measured in phase 9):

* **a sky touches the top of the frame.** The sky model takes a frozen pond,
  flat and pale, for sky (DSC05634); a region that does not reach the top edge
  of an upright frame is not the sky, and is dropped -- reflections included.
* **a person shows some skin.** The people model, trained on portraits, reads
  bark as clothes (DSC05618) and a shadow as hair (DSC05625) where nobody is
  in the picture; a region with no skin in it at all is dropped. A figure seen
  from behind with gloves and a hood is lost with them: the brush is there.

Like the scene model (``analysis/embed.py``): never loaded at start-up, and
dropped after five idle minutes (section 26). Unlike it, four threads: this is
a click the user waits on, not batch work -- the sky model takes 935 ms on one
core of the target machine, 277 ms on four.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

from ..models_registry import Unavailable, feature_status, model_path
from ..pipeline.filters import guided_filter

__all__ = [
    "SEGMENT_VERSION",
    "SUBJECT_FEATURE",
    "release_if_idle",
    "segment",
    "status",
]

_log = logging.getLogger(__name__)

#: Bumped when the output of :func:`segment` changes for the same photo.
SEGMENT_VERSION = 1

SUBJECT_FEATURE = {"sky": "segment_sky", "person": "segment_person", "skin": "segment_person"}

_PEOPLE_SIZE = 256
_SKY_SIZE = 320
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: The refinement: a neighbourhood of 0.6 % of the long edge (12 px on the
#: proxy, about the model's own pixel at 2048 / 320) and edges stronger than
#: about 0.03 in luminance. Wider, and a thin branch against the sky is filled
#: in; narrower, and the model's grid shows through.
_REFINE_RADIUS = 0.006
_REFINE_EPS = 0.03**2

#: A region counts from half probability; the rules above look at regions.
_REGION = 0.5
#: "Touches the top": reaches the first 2 % of the rows.
_TOP_ROWS = 0.02
#: "Some skin": this many pixels of skin probability above one half, as a
#: fraction of the frame -- a face at the far side of a group is 0.05 %.
_MIN_SKIN = 0.0002
#: Softening of the kept regions, so their feathered edge survives the cut.
_KEEP_SIGMA = 0.004

THREADS = 4

IDLE_UNLOAD_S = 300.0

_lock = threading.Lock()
_sessions: dict[str, Any] = {}
_last_used: dict[str, float] = {}


def status(subject: str) -> tuple[bool, Unavailable | None]:
    ok, reason, _entry = feature_status(SUBJECT_FEATURE[subject])
    return ok, reason


def _session(feature: str) -> Any:
    session = _sessions.get(feature)
    if session is None:
        import onnxruntime as ort

        _ok, _reason, entry = feature_status(feature)
        options = ort.SessionOptions()
        options.intra_op_num_threads = THREADS
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.enable_cpu_mem_arena = False
        session = ort.InferenceSession(
            str(model_path(entry)), options, providers=["CPUExecutionProvider"]
        )
        _sessions[feature] = session
        _log.info("modello di segmentazione %s caricato", entry.name)
    _last_used[feature] = time.monotonic()
    return session


def release_if_idle(now: float | None = None) -> bool:
    """Drop the sessions idle for five minutes. True if any was dropped."""
    now = time.monotonic() if now is None else now
    with _lock:
        idle = [f for f in _sessions if now - _last_used.get(f, 0.0) >= IDLE_UNLOAD_S]
        for feature in idle:
            del _sessions[feature]
    if idle:
        _log.info("modelli di segmentazione scaricati dalla memoria: %s", ", ".join(idle))
    return bool(idle)


def _people(image: np.ndarray) -> np.ndarray:
    """Softmax probabilities, ``(256, 256, 6)``."""
    small = cv2.resize(image, (_PEOPLE_SIZE, _PEOPLE_SIZE), interpolation=cv2.INTER_AREA)
    batch = (small.astype(np.float32) / 255.0)[None]
    with _lock:
        session = _session("segment_person")
        logits = session.run(None, {session.get_inputs()[0].name: batch})[0][0]
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def _sky(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (_SKY_SIZE, _SKY_SIZE), interpolation=cv2.INTER_AREA)
    normalised = (small.astype(np.float32) / 255.0 - _MEAN) / _STD
    batch = np.ascontiguousarray(normalised.transpose(2, 0, 1)[None])
    with _lock:
        session = _session("segment_sky")
        out = session.run(None, {session.get_inputs()[0].name: batch})[0][0, 0]
    return np.asarray(out, dtype=np.float32)


def segment(image: np.ndarray, subject: str) -> np.ndarray:
    """How much of each pixel is ``subject``, ``(H, W)`` float32 in [0, 1].

    Args:
        image: the proxy, ``(H, W, 3)`` uint8 RGB -- the frame masks live in.
        subject: ``sky``, ``person`` or ``skin``.

    Raises:
        RuntimeError: the model is not here (not downloaded, or not the file
            that was pinned). The caller checks :func:`status` first.
    """
    ok, reason = status(subject)
    if not ok:
        raise RuntimeError(f"modello di segmentazione non disponibile ({reason})")
    height, width = image.shape[:2]

    def full(small: np.ndarray) -> np.ndarray:
        return cv2.resize(small.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    if subject == "sky":
        up = full(_sky(image))
        keep = _regions(up, lambda labels: np.unique(labels[: max(1, round(height * _TOP_ROWS))]))
    else:
        classes = _people(image)
        skin = full(classes[..., 2] + classes[..., 3])
        person = full(1.0 - classes[..., 0])
        minimum = max(1, round(_MIN_SKIN * height * width))

        def with_skin(labels: np.ndarray) -> np.ndarray:
            counts = np.bincount(labels[skin > _REGION], minlength=labels.max() + 1)
            return np.flatnonzero(counts >= minimum)

        # Skin too only inside a person that passed: a patch of warm bark is
        # not a face.
        keep = _regions(person, with_skin)
        up = person if subject == "person" else skin
    if keep is not None:
        up *= keep
    guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    refined = guided_filter(guide, up, _REFINE_RADIUS, _REFINE_EPS)
    return np.clip(refined, 0.0, 1.0, out=refined)


def _regions(probability: np.ndarray, kept) -> np.ndarray:
    """A soft 0..1 map of the regions ``kept(labels)`` names, 1 inside them.

    Regions are the connected components of ``probability > 0.5``; ``kept``
    receives their label image (0 = no region) and returns the labels to keep.
    """
    binary = (probability > _REGION).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(probability)
    wanted = np.asarray(kept(labels), dtype=np.int64)
    wanted = wanted[wanted > 0]
    lookup = np.zeros(count, dtype=np.float32)
    lookup[wanted] = 1.0
    mask = lookup[labels]
    sigma = _KEEP_SIGMA * max(probability.shape)
    # Grown a little before softening: the feathered rim of a kept region lies
    # just outside its half-probability outline.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=max(1, round(sigma)))
    return cv2.GaussianBlur(mask, (0, 0), sigma)
