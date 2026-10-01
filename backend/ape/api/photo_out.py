# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""How a photo is described to the grid: its proxy, its revision, its badge.

Shared by the photo listing, the review screen and the merges screen, which
all show thumbnails and must agree on which URL names which picture.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import JobKind, MergeGroup, MergeMember, Photo
from .schemas import MergeBadge, PhotoOut

__all__ = ["merge_badges", "photo_out", "proxy_on_disk", "proxy_rev", "restore_proxies"]


def proxy_rev(path: str | None) -> str | None:
    """Changes whenever the proxy file does, for the URL the browser caches.

    The proxy is served ``immutable``, so a regenerated one -- a new lens
    profile, the lens-corrected proxies of phase 5 -- needs a new URL, or the
    grid would show the old picture for a year.
    """
    import os

    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    token = f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode()
    return hashlib.blake2b(token, digest_size=6).hexdigest()


def proxy_on_disk(path: str | None) -> bool:
    """A proxy on record *and* in the cache: the quota may have removed it (section 20.3)."""
    return bool(path) and Path(path).is_file()


def restore_proxies(session: Session, photos: list[Photo]) -> int:
    """Queue the proxies the cache lost; the photos keep their analysis."""
    from ..jobs.queue import enqueue

    queued = 0
    for photo in photos:
        if photo.proxy_path and not photo.missing and not proxy_on_disk(photo.proxy_path):
            job = enqueue(
                session,
                JobKind.PROXY,
                {"photo_id": photo.id, "restore": True},
                project_id=photo.project_id,
                dedupe_key=f"proxy:{photo.id}",
            )
            queued += job is not None
    return queued


def merge_badges(session: Session, photos: list[Photo]) -> dict[int, MergeBadge]:
    """Kind and number of sources of merged photos, in one query."""
    groups = {p.merge_group_id: p.id for p in photos if p.merge_group_id is not None}
    if not groups:
        return {}
    rows = session.execute(
        select(MergeGroup.id, MergeGroup.kind, func.count(MergeMember.id))
        .join(MergeMember, MergeMember.group_id == MergeGroup.id)
        .where(MergeGroup.id.in_(list(groups)))
        .group_by(MergeGroup.id)
    ).all()
    return {groups[gid]: MergeBadge(kind=kind.value, sources=count) for gid, kind, count in rows}


def photo_out(photo: Photo, badge: MergeBadge | None = None) -> PhotoOut:
    data = PhotoOut.model_validate(photo).model_dump()
    data["has_proxy"] = proxy_on_disk(photo.proxy_path)
    data["proxy_rev"] = proxy_rev(photo.proxy_path)
    data["has_thumb"] = bool(photo.path) and not photo.missing
    data["merge"] = badge
    return PhotoOut(**data)
