# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Thread pinning, done before anything is in a position to ignore it.

docs/SPEC.md section 12 says each worker pins BLAS and OpenCV to one thread. The
subtlety is *when*. OpenMP reads ``OMP_NUM_THREADS`` once, when its runtime
initialises, which happens the first time something links against it -- LibRaw,
inside ``rawpy``, before a single job has been claimed. Setting the variable in
the worker loop after that point changes nothing at all, and the pool ends up
with fourteen processes each running sixteen OpenMP threads.

The cost of getting this wrong is not theoretical. Measured on the machine of
section 3, importing 96 photos with fourteen workers:

* pinned only after ``rawpy`` had loaded: 397 ms per photo;
* pinned before: 246 ms per photo.

Same code, same files, 1.6x the throughput, purely from not oversubscribing
twelve cores by a factor of sixteen.

So this module does two things in this order and nothing else: set the
variables, then import the heavy libraries. It is what the forkserver preloads,
which means every worker forked from it starts with the libraries already in
memory *and* already pinned.
"""

from __future__ import annotations

import os

#: Every environment variable that makes a numerical library start a thread pool.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

for _variable in THREAD_VARIABLES:
    os.environ[_variable] = "1"

# Order matters: the variables above are read by these imports, not after them.
import cv2  # noqa: E402
import numpy  # noqa: E402, F401
import rawpy  # noqa: E402, F401

cv2.setNumThreads(1)

# The handlers, so that a forked worker does not import them one at a time.
from . import handlers, handlers_analysis, worker  # noqa: E402, F401

__all__ = ["THREAD_VARIABLES"]
