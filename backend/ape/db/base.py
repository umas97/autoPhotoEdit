# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The declarative base and the helpers every table here uses.

Split out of ``models.py`` so that the table modules can import it without
importing each other (docs/SPEC.md section 26 caps a source file at about four
hundred lines, and the catalogue of section 11 is larger than that).
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import DateTime, Dialect, TypeDecorator
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase

__all__ = ["Base", "UTCDateTime", "enum_column", "sql_timestamp", "utcnow"]


def utcnow() -> datetime:
    """Timezone-aware now. Every timestamp in the catalogue is UTC."""
    return datetime.now(UTC)



class Base(DeclarativeBase):
    pass



def enum_column(enum_type: type[enum.Enum], name: str) -> SAEnum:
    """A string-valued enum column with a CHECK constraint behind it.

    ``native_enum=False`` because SQLite has no enum type; what it gives us is a
    ``VARCHAR`` plus a ``CHECK``, which is exactly the right thing: readable in
    a shell, and still impossible to put a typo into.
    """
    return SAEnum(
        enum_type,
        name=name,
        native_enum=False,
        length=32,
        values_callable=lambda e: [member.value for member in e],
        validate_strings=True,
    )




class UTCDateTime(TypeDecorator):
    """A timestamp that is UTC on both sides of the database.

    SQLite has no date type: SQLAlchemy stores a datetime as a string and hands
    it back naive, ``timezone=True`` or not. Left alone, that turns every
    comparison between a stored timestamp and ``utcnow()`` into a TypeError, and
    -- worse -- lets a timestamp written in one timezone be read as if it were
    in another. This normalises to UTC going in and re-attaches UTC coming out,
    so the rest of the program never sees a naive datetime.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            # A naive value can only sensibly mean UTC here: nothing in this
            # program constructs a local timestamp.
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        return None if value is None else value.replace(tzinfo=UTC)


def sql_timestamp(value: datetime | None = None) -> str:
    """A timestamp formatted the way :class:`UTCDateTime` stores one.

    For the handful of statements written in raw SQL -- the atomic job claim --
    which bypass the type system and would otherwise write a different format
    from the ORM's.
    """
    moment = value or utcnow()
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC).replace(tzinfo=None)
    return moment.isoformat(sep=" ")
