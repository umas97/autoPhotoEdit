# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Disk use and the cache quota for the Settings screen (section 20.3).

``GET /api/storage`` says how much each category takes and what "Svuota cache"
would free and keep -- *before* it runs, as the section asks; ``POST
/api/storage/clear`` runs it. The quota itself is the ``cache_max_gb``
setting, written through ``PUT /api/settings/cache_max_gb``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import cache
from .deps import get_session

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["storage"])


@router.get("/storage")
def read_storage(session: Session = Depends(get_session)) -> dict:
    from ..config import get_settings

    limit = cache.limit_bytes(session)
    return {
        "limit_bytes": limit,
        "limit_gb": limit / cache.GB,
        "masks_dir": str(get_settings().masks_dir),
        "retouch_dir": str(get_settings().retouch_dir),
        "clear": cache.clear_plan(),
        **cache.usage(),
    }


@router.post("/storage/clear")
def clear_cache() -> dict:
    return cache.clear()
