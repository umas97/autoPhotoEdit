#!/usr/bin/env python3
# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Draw the application icons (docs/SPEC.md sections 18 and 21.2).

Three PNGs and one SVG, all from the same geometry: on a dark plate, the four
corners of a crop and, inside them, a tone curve -- the S that every RAW goes
through on its way to a photograph -- with its one control point at the middle.
The two gestures of the program, framing and toning, and nothing else.

Grey rather than coloured, for the reason section 10 gives: the icon sits in a
dock next to the window, and the program's whole business is that the user's
eye is not being lied to about colour.

The geometry is written once, in a 512 px square, and both the SVG and the
Pillow drawing read it, so the two can never disagree.

The maskable variant is the same drawing at 80% scale on a full-bleed plate:
Android and some desktops crop an icon to their own shape, and anything outside
the safe circle is lost.

Run from ``frontend/``::

    ../.venv/bin/python scripts/make_icons.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

PLATE = "#111111"  # --color-ink-900
FRAME = "#6b6b6b"  # the crop corners, receding
CURVE = "#ededed"  # the curve and its point, the brightest thing there

OUT = Path(__file__).resolve().parent.parent / "public" / "icons"
#: The hicolor sizes of the desktop entry (section 18), installed by install.sh.
DESKTOP = Path(__file__).resolve().parents[2] / "packaging" / "icons"

# --- the geometry, in a 512 px square ----------------------------------------

ROUND = 112  # corner radius of the plate
#: The crop: inset, length of each arm, stroke.
CROP_IN, CROP_ARM, CROP_W = 104, 72, 20
#: The curve runs corner to corner of the crop, as a tone curve runs from black
#: to white; the control handles make it a gentle S. Symmetric about the
#: centre, so the control point sits at (256, 256).
CURVE_FROM, CURVE_TO = (150, 362), (362, 150)
HANDLE_A, HANDLE_B = (250, 366), (262, 146)
CURVE_W = 24
POINT_R, POINT_RING = 30, 14


def _crop_paths() -> list[list[tuple[float, float]]]:
    """Four L-shaped corners, each as a polyline: arm, corner, arm."""
    a, b, arm = CROP_IN, 512 - CROP_IN, CROP_ARM
    return [
        [(a, a + arm), (a, a), (a + arm, a)],
        [(b - arm, a), (b, a), (b, a + arm)],
        [(b, b - arm), (b, b), (b - arm, b)],
        [(a + arm, b), (a, b), (a, b - arm)],
    ]


def _bezier(t: float) -> tuple[float, float]:
    p0, p1, p2, p3 = CURVE_FROM, HANDLE_A, HANDLE_B, CURVE_TO
    u = 1 - t
    return (
        u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
    )


def _curve_points(steps: int = 240) -> list[tuple[float, float]]:
    return [_bezier(i / steps) for i in range(steps + 1)]


# --- the PNGs ------------------------------------------------------------------


def _polyline(draw: ImageDraw.ImageDraw, points, width: float, fill: str) -> None:
    """A thick line with round caps and joins: Pillow's own has neither."""
    draw.line(points, fill=fill, width=round(width))
    r = width / 2
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def _stroke(draw: ImageDraw.ImageDraw, points, width: float, fill: str) -> None:
    """A thick curve, stamped as discs along it. A wide ``draw.line`` over many
    short segments frays at every one of their ends."""
    r = width / 2
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def draw_icon(size: int, *, maskable: bool = False) -> Image.Image:
    # Supersampled and downscaled: Pillow has no antialiasing of its own.
    scale = 4
    edge = size * scale
    canvas = Image.new("RGBA", (edge, edge), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    if maskable:
        draw.rectangle([0, 0, edge, edge], fill=PLATE)
        zoom = 0.8
    else:
        draw.rounded_rectangle([0, 0, edge - 1, edge - 1], radius=edge * ROUND / 512, fill=PLATE)
        zoom = 1.0
    k = edge / 512 * zoom
    offset = edge * (1 - zoom) / 2

    def at(point: tuple[float, float]) -> tuple[float, float]:
        return (offset + point[0] * k, offset + point[1] * k)

    for corner in _crop_paths():
        _polyline(draw, [at(p) for p in corner], CROP_W * k, FRAME)
    # Densely enough that the discs overlap into a smooth edge at any size.
    _stroke(draw, [at(p) for p in _curve_points(steps=6 * edge)], CURVE_W * k, CURVE)

    cx, cy = at(_bezier(0.5))
    outer, inner = POINT_R * k, (POINT_R - POINT_RING) * k
    draw.ellipse([cx - outer, cy - outer, cx + outer, cy + outer], fill=CURVE)
    draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=PLATE)
    return canvas.resize((size, size), Image.LANCZOS)


# --- the SVG -------------------------------------------------------------------


def _pts(points) -> str:
    return " ".join(f"{x:g},{y:g}" for x, y in points)


def svg_document() -> str:
    (x0, y0), (x3, y3) = CURVE_FROM, CURVE_TO
    (x1, y1), (x2, y2) = HANDLE_A, HANDLE_B
    path = f"M{x0} {y0} C{x1} {y1} {x2} {y2} {x3} {y3}"
    cx, cy = _bezier(0.5)
    corners = "\n".join(f'    <polyline points="{_pts(c)}"/>' for c in _crop_paths())
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img"
     aria-label="autoPhotoEdit">
  <rect width="512" height="512" rx="{ROUND}" fill="{PLATE}"/>
  <g fill="none" stroke="{FRAME}" stroke-width="{CROP_W}" stroke-linecap="round"
     stroke-linejoin="round">
{corners}
  </g>
  <path d="{path}" fill="none" stroke="{CURVE}" stroke-width="{CURVE_W}"
        stroke-linecap="round"/>
  <circle cx="{cx:g}" cy="{cy:g}" r="{POINT_R - POINT_RING / 2:g}" fill="{PLATE}"
          stroke="{CURVE}" stroke-width="{POINT_RING}"/>
</svg>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_icon(192).save(OUT / "icon-192.png")
    draw_icon(512).save(OUT / "icon-512.png")
    draw_icon(512, maskable=True).save(OUT / "icon-512-maskable.png")
    (OUT / "icon.svg").write_text(svg_document(), encoding="utf-8")
    DESKTOP.mkdir(parents=True, exist_ok=True)
    for size in (48, 128, 256):
        draw_icon(size).save(DESKTOP / f"autophotoedit-{size}.png")
    (DESKTOP / "autophotoedit.svg").write_text(svg_document(), encoding="utf-8")
    print(f"icone scritte in {OUT} e {DESKTOP}")


if __name__ == "__main__":
    main()
