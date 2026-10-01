# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Memory of the workers: how many heavy jobs at once, and giving memory back.

The pool has fourteen workers on the machine of section 3 because most jobs are
small: a proxy, an analysis, a prediction. A full-resolution export is not --
the whole 24 MP frame in float32 and the render's working buffers -- and
fourteen of them at once would need more memory than the machine has. The pool
keeps its size and caps the exports instead; since the render is bound by
memory bandwidth (measured in phase 5), fewer exports at once lose little
throughput.
"""

from __future__ import annotations

import sys

__all__ = [
    "EXPORT_PEAK_GB",
    "default_export_limit",
    "release_idle_models",
    "trim_heap",
]


#: Peak memory of one full-resolution export of a 24 MP frame, in GB, measured
#: on the user's files in phase 8. What the export cap is computed from.
EXPORT_PEAK_GB = 1.3


def _available_memory_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def default_export_limit(workers: int) -> int:
    """How many full-resolution exports may run at once without swapping.

    Three quarters of the memory available when the pool starts, divided by the
    peak of one export: the rest is the server, the browser and the other
    workers' own resident size. Never more than the pool, never fewer than one.
    """
    available = _available_memory_gb()
    if available is None:
        return max(1, workers // 2)
    return max(1, min(workers, int(available * 0.75 / EXPORT_PEAK_GB)))


def trim_heap() -> None:
    """Give freed memory back to the system.

    A decode frees its buffers, but glibc keeps the pages for the next one:
    measured, a worker that has analysed a few photos sits at 345 MB, and 264
    after a trim -- 175 once the scene model is dropped too. Fourteen workers
    make that the difference between 4.8 and 2.5 GB held by an idle program
    (section 26). Linux only; elsewhere a no-op.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def release_idle_models() -> bool:
    """Drop ONNX sessions nobody used for five minutes (section 26).

    Checked from here, when the queue is empty, rather than by a timer: the
    loop is already awake, and a worker that never loaded a model never
    imports the module that would hold one. Returns whether it dropped one.
    """
    released = False
    for name in ("ape.analysis.embed", "ape.analysis.segment", "ape.retouch.ml"):
        module = sys.modules.get(name)
        if module is not None and module.release_if_idle():
            released = True
    return released
