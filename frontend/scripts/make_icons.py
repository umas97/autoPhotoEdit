#!/usr/bin/env python3
# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Draw the application icons (docs/SPEC.md sections 18 and 21.2).

Three PNGs and one SVG, all from the same geometry: a neutral grey plate with an
aperture on it. Grey rather than coloured, for the reason section 10 gives -- the
icon sits in a dock next to the window, and the program's whole business is that
the user's eye is not being lied to about colour.

The maskable variant is the same drawing at 62% scale on a full-bleed plate:
Android and some desktops crop an icon to their own shape, and anything outside
the safe circle is lost.

Run from ``frontend/``::

    ../.venv/bin/python scripts/make_icons.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

PLATE = (17, 17, 17, 255)  # --color-ink-900
BLADE = (212, 212, 212, 255)  # --color-ink-100
RING = (138, 138, 138, 255)  # --color-ink-300

OUT = Path(__file__).resolve().parent.parent / "public" / "icons"
#: The hicolor sizes of the desktop entry (section 18), installed by install.sh.
DESKTOP = Path(__file__).resolve().parents[2] / "packaging" / "icons"


def _aperture(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float) -> None:
    """A six-blade iris: a light disc with six wedges cut out of it."""
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=BLADE,
        outline=RING,
        width=max(2, int(radius * 0.06)),
    )
    gap = radius * 0.17
    for index in range(6):
        angle = math.radians(60 * index - 20)
        # From a point on the inner hexagon, outwards past the rim: the cut that
        # makes one blade end and the next begin.
        inner = (cx + math.cos(angle) * radius * 0.30, cy + math.sin(angle) * radius * 0.30)
        outer_angle = angle + math.radians(38)
        outer = (
            cx + math.cos(outer_angle) * radius * 1.15,
            cy + math.sin(outer_angle) * radius * 1.15,
        )
        draw.line([inner, outer], fill=PLATE, width=int(gap))

    hole = radius * 0.22
    draw.ellipse([cx - hole, cy - hole, cx + hole, cy + hole], fill=PLATE)


def draw_icon(size: int, *, maskable: bool = False) -> Image.Image:
    # Supersampled and downscaled: the blades are thin lines, and Pillow has no
    # antialiasing of its own.
    scale = 4
    canvas = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    edge = size * scale

    if maskable:
        draw.rectangle([0, 0, edge, edge], fill=PLATE)
        radius = edge * 0.31
    else:
        draw.rounded_rectangle([0, 0, edge - 1, edge - 1], radius=edge * 0.22, fill=PLATE)
        radius = edge * 0.34

    _aperture(draw, edge / 2, edge / 2, radius)
    return canvas.resize((size, size), Image.LANCZOS)


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" role="img"
     aria-label="autoPhotoEdit">
  <rect width="512" height="512" rx="112" fill="#111111"/>
  <g transform="translate(256 256)">
    <circle r="174" fill="#d4d4d4" stroke="#8a8a8a" stroke-width="10"/>
    <g stroke="#111111" stroke-width="30" stroke-linecap="butt">
{cuts}
    </g>
    <circle r="38" fill="#111111"/>
  </g>
</svg>
"""


def svg_document() -> str:
    lines = []
    for index in range(6):
        angle = math.radians(60 * index - 20)
        x1, y1 = math.cos(angle) * 52, math.sin(angle) * 52
        outer = angle + math.radians(38)
        x2, y2 = math.cos(outer) * 200, math.sin(outer) * 200
        lines.append(f'      <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    return SVG.format(cuts="\n".join(lines))


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
