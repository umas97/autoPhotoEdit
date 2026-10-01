# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Which lensfun profile a photo's lens has, if any (docs/SPEC.md section 6.4).

Two databases can answer. The one bundled with the ``lensfunpy`` wheel is always
there and is what a fresh install uses -- no system package needed. It is also
whatever lensfun had when the wheel was built, and recent lenses arrive later:
the user's Tamron 28-75 G2 (A063) is in the lensfun repository and not in the
wheel. So the Settings offer "Aggiorna dati lensfun", an explicit download of
the current data (one of the two network calls section 19 allows), which is
kept under ``$XDG_DATA_HOME/autophotoedit/lensfun/`` and, once present, is used
*instead of* the bundled copy: it is a superset of it.

The wheel's library reads database format 1, so that is the format fetched.
Format 2 files make it refuse the whole database.

**A wrong profile is worse than none.** lensfun's matcher is fuzzy by design, so
a match is accepted only if the focal range written in the EXIF name agrees
with the profile's: "28-75mm" must not correct as "24-70mm" because the words
around it look alike. A lens without a profile is recorded as such, corrected
not at all, flagged in the interface and in the confidence score (section 9.1),
and the user can associate one by hand -- remembered for that lens in every
project (``LensProfileOverride``).
"""

from __future__ import annotations

import logging
import re
import shutil
import tarfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import get_settings
from .safety import assert_outside_source

__all__ = [
    "LENSFUN_DATA_URL",
    "LensIdentity",
    "LensProfile",
    "data_status",
    "find_lens",
    "install_update",
    "lensfun_available",
    "resolve",
    "search_lenses",
]

_log = logging.getLogger(__name__)

#: What ``lensfun-update-data`` fetches for a version-1 library.
LENSFUN_DATA_URL = "https://lensfun.github.io/db/version_1.tar.bz2"

#: lensfun scores a perfect textual match at 85 and drops for every token that
#: had to be skipped. The A063 matches its own profile at 71 -- "Tamron" and
#: "Di III VXD G2" are in the profile, not in the EXIF -- while unrelated
#: lenses that happen to share a focal range stay far lower. The numeric check
#: below is the real guard; this only drops matches lensfun itself doubts.
_MIN_SCORE = 50

_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")
_FOCAL = re.compile(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*mm", re.IGNORECASE)


@dataclass(frozen=True)
class LensIdentity:
    """What the EXIF says about the optics of one shot."""

    camera_maker: str | None
    camera_model: str | None
    lens_model: str | None
    focal_length: float | None
    aperture: float | None
    #: Focus distance in metres. Sony writes none that lensfun could use, and
    #: its only effect is on the vignetting interpolation, so infinity it is.
    distance: float = 1000.0
    #: ``(maker, model)`` of a profile the user associated by hand.
    override: tuple[str, str] | None = None


@dataclass(frozen=True)
class LensProfile:
    """A lensfun profile, by name. The lensfun objects themselves do not pickle."""

    maker: str
    model: str
    #: ``auto`` found by matching the EXIF name; ``override`` chosen by the user.
    source: str
    crop_factor: float

    def as_json(self) -> dict[str, Any]:
        return {"maker": self.maker, "model": self.model, "source": self.source}


def lensfun_available() -> bool:
    try:
        import lensfunpy  # noqa: F401
    except Exception:  # pragma: no cover - depends on the installed extras
        return False
    return True


def _updated_dir() -> Path:
    return get_settings().data_dir / "lensfun" / "version_1"


def _stamp() -> str:
    """Changes whenever the database on disk changes, to key the caches on."""
    marker = _updated_dir() / "timestamp.txt"
    try:
        return marker.read_text().strip() or "updated"
    except OSError:
        return "bundled"


@lru_cache(maxsize=2)
def _database(stamp: str) -> Any:
    import lensfunpy

    if stamp != "bundled":
        files = sorted(str(p) for p in _updated_dir().glob("*.xml"))
        if files:
            try:
                return lensfunpy.Database(paths=files, load_common=False, load_bundled=False)
            except Exception as exc:  # a broken update must not break correction
                _log.warning("dati lensfun aggiornati illeggibili, uso quelli inclusi: %s", exc)
    return lensfunpy.Database()


def database() -> Any:
    return _database(_stamp())


def data_status() -> dict[str, Any]:
    """Which database is in use, for the Settings screen."""
    if not lensfun_available():
        return {"available": False, "source": None, "updated_at": None, "lenses": 0}
    stamp = _stamp()
    db = database()
    updated_at = None
    if stamp != "bundled" and stamp.isdigit():
        from datetime import UTC, datetime

        updated_at = datetime.fromtimestamp(int(stamp), UTC).isoformat()
    return {
        "available": True,
        "source": "bundled" if stamp == "bundled" else "updated",
        "updated_at": updated_at,
        "lenses": len(db.lenses),
    }


def _focal_range(text: str | None) -> tuple[float, float] | None:
    """``"FE 28-70mm F3.5-5.6 OSS"`` -> ``(28, 70)``."""
    if not text:
        return None
    match = _FOCAL.search(text)
    if match is None:
        return None
    low = float(match.group(1))
    high = float(match.group(2)) if match.group(2) else low
    return low, high


def _plausible(identity: LensIdentity, lens: Any) -> bool:
    """The numeric guard against lensfun's fuzzy matching (see the module doc)."""
    stated = _focal_range(identity.lens_model)
    if stated is not None:
        low, high = stated
        if abs(low - lens.min_focal) > 0.6 or abs(high - lens.max_focal) > 0.6:
            return False
    if identity.focal_length is not None and lens.min_focal and lens.max_focal:
        f = identity.focal_length
        if not lens.min_focal * 0.98 <= f <= lens.max_focal * 1.02:
            return False
    return True


def _camera(db: Any, identity: LensIdentity) -> Any | None:
    if not identity.camera_model:
        return None
    cameras = db.find_cameras(identity.camera_maker or None, identity.camera_model)
    return cameras[0] if cameras else None


def find_lens(identity: LensIdentity) -> tuple[Any | None, Any | None]:
    """``(camera, lens)`` lensfun objects for one shot, either possibly ``None``."""
    if not lensfun_available():
        return None, None
    db = database()
    camera = _camera(db, identity)
    if identity.override is not None:
        maker, model = identity.override
        for lens in db.lenses:
            if lens.maker == maker and lens.model == model:
                return camera, lens
        return camera, None
    if not identity.lens_model:
        return camera, None
    try:
        candidates = db.find_lenses(camera, None, identity.lens_model)
    except Exception as exc:  # lensfun raises on odd characters in names
        _log.debug("ricerca obiettivo %r fallita: %s", identity.lens_model, exc)
        return camera, None
    for lens in candidates:
        if lens.score >= _MIN_SCORE and _plausible(identity, lens):
            return camera, lens
    return camera, None


@lru_cache(maxsize=256)
def _resolve_cached(identity: LensIdentity, stamp: str) -> LensProfile | None:
    camera, lens = find_lens(identity)
    if lens is None:
        return None
    crop = float(camera.crop_factor) if camera is not None else float(lens.crop_factor)
    source = "override" if identity.override is not None else "auto"
    return LensProfile(maker=lens.maker, model=lens.model, source=source, crop_factor=crop)


def resolve(identity: LensIdentity) -> LensProfile | None:
    """The profile that corrects this shot, or ``None`` for "senza profilo"."""
    if not lensfun_available():
        return None
    return _resolve_cached(identity, _stamp())


def search_lenses(text: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """Candidates for a manual association, best first.

    Matches every word of ``text`` against maker and model, and ranks lenses
    whose focal range is also in the text first: a user typing "28-75" wants
    the 28-75s, whatever their brand.
    """
    if not lensfun_available():
        return []
    words = [w.lower() for w in re.split(r"\s+", text.strip()) if w]
    stated = _focal_range(text)
    numbers = {float(n) for n in _NUMBER.findall(text)}
    scored: list[tuple[float, Any]] = []
    for lens in database().lenses:
        haystack = f"{lens.maker} {lens.model}".lower()
        if words and not all(w in haystack for w in words if not _NUMBER.fullmatch(w)):
            continue
        score = 0.0
        same_range = stated and (
            abs(lens.min_focal - stated[0]) < 0.6 and abs(lens.max_focal - stated[1]) < 0.6
        )
        if same_range:
            score += 10
        score += sum(1 for n in numbers if n in (lens.min_focal, lens.max_focal))
        score += sum(1 for w in words if w in haystack)
        if score > 0 or not words:
            scored.append((score, lens))
    scored.sort(key=lambda pair: (-pair[0], pair[1].maker, pair[1].model))
    return [
        {
            "maker": lens.maker,
            "model": lens.model,
            "min_focal": float(lens.min_focal),
            "max_focal": float(lens.max_focal),
        }
        for _score, lens in scored[:limit]
    ]


def install_update(archive: Path) -> dict[str, Any]:
    """Unpack a downloaded ``version_1.tar.bz2`` and switch to it.

    The archive is extracted beside the live copy, loaded once to prove the
    library accepts it, and only then swapped in. A bad archive leaves the
    database that was working in place.

    Raises:
        ValueError: the archive is not a lensfun database this library reads.
    """
    import lensfunpy

    target = _updated_dir()
    staging = target.with_name(target.name + ".new")
    assert_outside_source(staging)
    assert_outside_source(target)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            members = [
                m for m in tar.getmembers()
                if m.isfile() and "/" not in m.name.strip("./") and (
                    m.name.endswith(".xml") or m.name.endswith("timestamp.txt")
                )
            ]
            tar.extractall(staging, members=members, filter="data")
        files = sorted(str(p) for p in staging.glob("*.xml"))
        if not files:
            raise ValueError("l'archivio non contiene dati lensfun")
        db = lensfunpy.Database(paths=files, load_common=False, load_bundled=False)
        count = len(db.lenses)
        # lensfun skips what it cannot parse instead of failing: an archive cut
        # short loads as an empty database, which would switch correction off
        # for every lens. Emptiness is the failure to look for.
        if count == 0 or not db.cameras:
            raise ValueError("l'archivio non contiene obiettivi o fotocamere")
        if not (staging / "timestamp.txt").is_file():
            # The archive carries no date of its own; when it was installed is
            # what the Settings screen reports, and it keys the caches.
            (staging / "timestamp.txt").write_text(str(int(time.time())))
    except (tarfile.TarError, lensfunpy.XMLFormatError, OSError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"dati lensfun non validi: {exc}") from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    previous = target.with_name(target.name + ".old")
    shutil.rmtree(previous, ignore_errors=True)
    if target.exists():
        target.rename(previous)
    staging.rename(target)
    shutil.rmtree(previous, ignore_errors=True)
    _database.cache_clear()
    _resolve_cached.cache_clear()
    _log.info("dati lensfun aggiornati: %d obiettivi", count)
    return data_status()
