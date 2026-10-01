# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A standalone XMP packet, written as text.

The sidecars are XML and nothing more, so they are written with the standard
library rather than through exiv2: one less reason for the GPL dependency to
spread outside ``raw/metadata.py`` and ``export/`` (section 24), and full
control over what goes in. Both darktable and Lightroom read packets laid out
this way -- one ``rdf:Description`` with simple properties as attributes and
the structured ones (sequences, language alternatives) as child elements.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr

__all__ = ["Packet", "format_number"]

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def format_number(value: float, decimals: int = 0, *, sign: bool = False) -> str:
    """A number the way Camera Raw writes it: ``+0.35``, ``-12``, ``0``."""
    if decimals == 0:
        number = int(round(value))
        text = str(number)
        return f"+{text}" if sign and number > 0 else text
    rounded = round(float(value), decimals)
    if rounded == 0:
        rounded = 0.0  # no "-0.00"
    text = f"{rounded:.{decimals}f}"
    return f"+{text}" if sign and rounded > 0 else text


@dataclass
class Packet:
    """One XMP packet: namespaces, simple properties, and structured children."""

    namespaces: dict[str, str] = field(default_factory=dict)
    attributes: list[tuple[str, str]] = field(default_factory=list)
    #: ``(qualified name, xml fragment)`` of each child element.
    children: list[tuple[str, str]] = field(default_factory=list)

    def set(self, name: str, value: object) -> None:
        self.attributes.append((name, str(value)))

    def update(self, values: Mapping[str, object]) -> None:
        for name, value in values.items():
            self.set(name, value)

    def seq(self, name: str, items: Iterable[str]) -> None:
        body = "".join(f"<rdf:li>{escape(str(item))}</rdf:li>" for item in items)
        self.children.append((name, f"<{name}><rdf:Seq>{body}</rdf:Seq></{name}>"))

    def bag(self, name: str, items: Iterable[str]) -> None:
        body = "".join(f"<rdf:li>{escape(str(item))}</rdf:li>" for item in items)
        self.children.append((name, f"<{name}><rdf:Bag>{body}</rdf:Bag></{name}>"))

    def alt(self, name: str, text: str) -> None:
        body = f'<rdf:li xml:lang="x-default">{escape(text)}</rdf:li>'
        self.children.append((name, f"<{name}><rdf:Alt>{body}</rdf:Alt></{name}>"))

    def seq_of_structs(self, name: str, structs: Sequence[Mapping[str, object]]) -> None:
        """A sequence of structures written as attribute-only ``rdf:li`` elements."""
        items = []
        for struct in structs:
            attrs = " ".join(f"{key}={quoteattr(str(value))}" for key, value in struct.items())
            items.append(f"<rdf:li {attrs}/>")
        self.children.append((name, f"<{name}><rdf:Seq>{''.join(items)}</rdf:Seq></{name}>"))

    def to_bytes(self, toolkit: str) -> bytes:
        namespaces = "".join(
            f"\n    xmlns:{prefix}={quoteattr(uri)}" for prefix, uri in self.namespaces.items()
        )
        attributes = "".join(
            f"\n   {name}={quoteattr(value)}" for name, value in self.attributes
        )
        children = "".join(f"\n   {fragment}" for _name, fragment in self.children)
        text = (
            '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            f'<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk={quoteattr(toolkit)}>\n'
            f' <rdf:RDF xmlns:rdf="{_RDF}">\n'
            f'  <rdf:Description rdf:about=""{namespaces}{attributes}>'
            f"{children}\n"
            "  </rdf:Description>\n"
            " </rdf:RDF>\n"
            "</x:xmpmeta>\n"
            '<?xpacket end="w"?>\n'
        )
        return text.encode("utf-8")
