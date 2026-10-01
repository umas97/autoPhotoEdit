# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Only the program's own window may talk to the server.

Binding ``127.0.0.1`` keeps the network out, but not the other pages open in
the user's browser. Two ways in are left, and each header closes one:

- **DNS rebinding.** A page on ``evil.example`` has its name re-resolved to
  ``127.0.0.1`` and then reads our answers as same-origin. The browser still
  sends ``Host: evil.example``, so a request whose ``Host`` is not a loopback
  name is refused.
- **Cross-site requests.** A page can POST to ``http://127.0.0.1:8787`` without
  reading the answer ("Esci dall'applicazione" needs nothing more), and can open
  ``ws://127.0.0.1:8787/ws`` and read it, since WebSockets have no same-origin
  rule. The ``Host`` is right there; the ``Origin`` is not. Browsers send it on
  every such request, so one that is not a loopback origin is refused too.

A request without ``Origin`` passes: that is a navigation, a same-origin GET, or
a program that is not a browser (the launcher asking a running instance for a
window, ``curl``) -- none of which a web page can forge.

The port is not compared. The attacks above name another host, never another
port of this one, and in development the window comes from ``vite`` on 5173.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

__all__ = ["LOOPBACK_NAMES", "LocalOnlyMiddleware"]

#: The names the window reaches the server by. ``[::1]`` is not among them
#: because the server never binds it (section 21.1).
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost"})

_BAD_HOST = "richiesta rifiutata: host non locale"
_BAD_ORIGIN = "richiesta rifiutata: proviene da una pagina che non è autoPhotoEdit"


def _hostname(netloc: str) -> str | None:
    """The host part of ``host[:port]``, lower-cased; ``None`` if unparsable."""
    try:
        return urlsplit(f"//{netloc}").hostname
    except ValueError:
        return None


def _refusal(headers: Headers) -> str | None:
    """Why this request is refused, or ``None`` if it may pass."""
    if _hostname(headers.get("host", "")) not in LOOPBACK_NAMES:
        return _BAD_HOST
    origin = headers.get("origin")
    if origin is None:
        return None
    # ``null`` is a sandboxed frame or a ``file://`` page: not our window.
    parts = urlsplit(origin) if origin != "null" else None
    if parts is None or parts.scheme not in ("http", "https"):
        return _BAD_ORIGIN
    if _hostname(parts.netloc) not in LOOPBACK_NAMES:
        return _BAD_ORIGIN
    return None


class LocalOnlyMiddleware:
    """Refuse HTTP and WebSocket requests that do not come from a local page."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        reason = _refusal(Headers(scope=scope))
        if reason is None:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # Closing before accepting refuses the handshake with a 403.
            await send({"type": "websocket.close", "code": 1008, "reason": reason})
            return
        status = 400 if reason is _BAD_HOST else 403
        body = json.dumps({"detail": reason}, ensure_ascii=False).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
