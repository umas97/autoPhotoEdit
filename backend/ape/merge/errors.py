# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The one way a merge says no."""

from __future__ import annotations

__all__ = ["MergeFailure"]


class MergeFailure(Exception):
    """A merge that cannot be made, with a sentence the user can act on.

    Raised for what is wrong with the *shots* -- frames that do not align, a
    panorama without enough overlap, one too large for the memory -- as opposed
    to a bug. The group goes to ``failed`` with this message and no merged photo
    is created (section 25.5.6): failing well is a requirement.
    """
