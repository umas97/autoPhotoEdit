# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Serving the built interface from the server that owns the data (section 4).

In development the interface is served by ``vite dev``, which proxies ``/api``
and ``/ws`` here. In production there is no second process: ``npm run build``
writes into ``backend/ape/static`` and uvicorn serves it, so the program is one
process listening on one port.

Three rules make the difference between a static mount and a PWA that behaves:

* **the router owns the URLs.** A reload on ``/progetti/3`` must give the same
  document as ``/``, because there is no file at that path and the browser has
  no address bar to fix it with (section 21.2). Anything that is not a file and
  not an API call falls back to ``index.html``;
* **the shell is never cached by the browser, the assets always are.** Vite puts
  a content hash in every asset name, so those are immutable; ``index.html``,
  the service worker and the manifest are not hashed, and a stale copy of any of
  them pins the user to an old build;
* **``/api`` and ``/ws`` are never reachable through here.** They are mounted
  first, and the fallback refuses those prefixes outright, so a routing mistake
  cannot turn an API 404 into a 200 with an HTML body -- which is exactly the
  failure test 19 forbids in the service worker, and it would be no better here.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

__all__ = ["STATIC_DIR", "build_present", "mount_frontend"]

#: Where ``vite build`` writes. Packaged inside the wheel (section 18).
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

#: Never cached: the browser must see a new build the moment there is one.
_NO_STORE = {"Cache-Control": "no-cache, no-store, must-revalidate"}

#: Files at the root of the build that carry no content hash in their name.
_UNHASHED = frozenset(
    {
        "index.html",
        "sw.js",
        "registerSW.js",
        "manifest.webmanifest",
        "offline.html",
        "workbox-window.prod.es5.mjs",
    }
)


def build_present(root: Path | None = None) -> bool:
    """Is there an interface to serve? False in a source checkout before a build."""
    return ((root or STATIC_DIR) / "index.html").is_file()


class _Shell(StaticFiles):
    """Static files with the cache headers each kind of file deserves."""

    def file_response(self, *args, **kwargs) -> Response:  # type: ignore[override]
        response = super().file_response(*args, **kwargs)
        name = Path(getattr(response, "path", "")).name
        if name in _UNHASHED:
            response.headers.update(_NO_STORE)
        else:
            # Hashed by Vite: the name changes whenever the bytes do.
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def mount_frontend(app: FastAPI, root: Path | None = None) -> bool:
    """Serve the built interface, if it has been built.

    Args:
        app: the application. Its API routes must already be registered: the
            fallback is a catch-all and would otherwise shadow them.
        root: override the build directory. Tests use it.

    Returns:
        True when a build was found and mounted.
    """
    target = root or STATIC_DIR
    if not build_present(target):
        return False

    # Not in the default map on every distribution, and a manifest served as
    # ``text/plain`` is a manifest the browser ignores -- with it, no install.
    mimetypes.add_type("application/manifest+json", ".webmanifest")

    assets = target / "assets"
    if assets.is_dir():
        app.mount("/assets", _Shell(directory=assets), name="assets")

    index = target / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str, request: Request) -> Response:
        """A file if there is one, the shell otherwise, never an API route."""
        if path.startswith(("api/", "ws", "docs", "redoc", "openapi.json")):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "risorsa inesistente")

        candidate = (target / path).resolve() if path else index
        if path and target.resolve() in candidate.parents and candidate.is_file():
            headers = dict(_NO_STORE) if candidate.name in _UNHASHED else {
                "Cache-Control": "public, max-age=31536000, immutable"
            }
            return FileResponse(candidate, headers=headers)

        # Unknown path: the router inside the document will make sense of it.
        return FileResponse(index, headers=_NO_STORE)

    _ = spa  # registered on the app; the name is not used again
    return True
