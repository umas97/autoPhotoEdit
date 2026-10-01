# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Application settings and XDG directory layout.

Everything the program produces lives under the XDG data/state/runtime
directories or under a user-chosen export folder -- never next to the RAW files
(docs/SPEC.md section 2.4).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME = "autophotoedit"


def _xdg(env_var: str, default: str) -> Path:
    """Resolve an XDG base directory, honouring the environment variable."""
    raw = os.environ.get(env_var)
    base = Path(raw).expanduser() if raw else Path.home() / default
    return base / APP_NAME


class Settings(BaseSettings):
    """Process-wide settings. Overridable through ``APE_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="APE_", extra="ignore")

    host: str = "127.0.0.1"  # docs/SPEC.md section 21.1: never bind anywhere else.
    port: int = 8787
    log_level: str = "INFO"

    # Cache budget, in gigabytes (docs/SPEC.md section 20.3).
    cache_max_gb: float = Field(default=20.0, gt=0)

    # Long edge of the UI proxy images.
    proxy_long_edge: int = 2048

    # Full-resolution exports allowed at once; 0 computes it from the memory
    # available when the pool starts (``jobs/limits.py``).
    export_concurrency: int = Field(default=0, ge=0)

    data_dir: Path = Field(default_factory=lambda: _xdg("XDG_DATA_HOME", ".local/share"))
    state_dir: Path = Field(default_factory=lambda: _xdg("XDG_STATE_HOME", ".local/state"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "catalog.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def proxy_dir(self) -> Path:
        return self.cache_dir / "proxies"

    @property
    def stage_cache_dir(self) -> Path:
        return self.cache_dir / "stages"

    @property
    def intermediate_dir(self) -> Path:
        """Linear scene-referred buffers of merged photos (docs/SPEC.md section 25.1)."""
        return self.cache_dir / "intermediates"

    @property
    def merge_preview_dir(self) -> Path:
        """The 1024 px previews of merge groups (section 25.6). Cache."""
        return self.cache_dir / "merges"

    @property
    def masks_dir(self) -> Path:
        """Hand-painted masks. Not cache: these cannot be regenerated (section 20.3)."""
        return self.data_dir / "masks"

    @property
    def retouch_dir(self) -> Path:
        """The fills of the magic eraser. Not cache either: a fill made by a
        model that is no longer here cannot be made again."""
        return self.data_dir / "retouch"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def runtime_dir(self) -> Path:
        raw = os.environ.get("XDG_RUNTIME_DIR")
        return Path(raw) if raw else self.state_dir

    @property
    def lock_path(self) -> Path:
        return self.runtime_dir / f"{APP_NAME}.lock"

    def ensure_dirs(self) -> None:
        """Create the directories the program writes into. Never touches sources."""
        for path in (
            self.data_dir,
            self.cache_dir,
            self.proxy_dir,
            self.stage_cache_dir,
            self.intermediate_dir,
            self.merge_preview_dir,
            self.masks_dir,
            self.retouch_dir,
            self.models_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
