# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Schema 7: the tint changes sign in every JSON the catalogue holds.

Parameters version 2 turned the tint round to Adobe's convention
(``raw/whitepoint.py``). Blobs carry a version and flip when they load (scene
features, style models); the JSON columns do not, so they are rewritten here,
once, in the same transaction as the schema bump:

* ``edit_version.params`` and ``style_sample.params`` through the parameters'
  own migration, so there is one definition of what changed;
* style vectors and as-shot contexts (``style_sample``, ``photo.prediction``);
* the camera's as-shot tint and the automatic white balance of
  ``photo.analysis``;
* the rules of a user copy of a built-in, if its base sets a tint shift.

Every render is identical afterwards, to the bit (``tests/test_tint_sign.py``).
The one derived value is the signature of the style last written by the
program, a digest of the parameters: it is recomputed where it matched, or
every styled photo would look edited by hand and stop following its profile.

Plain SQL on purpose: the ORM models describe the newest schema, a migration
must keep working against the one it was written for.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

__all__ = ["flip_tint"]

_log = logging.getLogger(__name__)


def _load(value: Any) -> Any:
    if value is None:
        return None
    return json.loads(value) if isinstance(value, str | bytes) else value


def _dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


def _signature(payload: dict) -> str | None:
    from ..pipeline.params import EditParams
    from ..style.apply import style_signature

    try:
        return style_signature(EditParams.model_validate(payload))
    except ValueError:
        return None


def _flip_analysis(analysis: dict | None) -> dict | None:
    from ..style.tint_sign import negate

    if not analysis:
        return analysis
    out = dict(analysis)
    shot = out.get("as_shot")
    if isinstance(shot, dict) and "tint" in shot:
        out["as_shot"] = {**shot, "tint": negate(shot["tint"])}
    measured = out.get("auto")
    if isinstance(measured, dict) and "wb_tint_shift" in measured:
        out["auto"] = {**measured, "wb_tint_shift": negate(measured["wb_tint_shift"])}
    return out


def flip_tint(session: Session) -> None:
    """Rewrite the catalogue's tints in the new sign.

    Runs once, in the step from schema 6 to 7: only the parameters say their
    own version, so running it twice would flip everything else back.
    """
    from ..pipeline.params import migrate
    from ..style.tint_sign import flip_context, flip_prediction, flip_rules, flip_vector

    connection = session.connection()

    # Versions first: the signature fix below needs each photo's current one.
    current: dict[int, tuple[str | None, str | None]] = {}
    rows = connection.execute(text("SELECT id, photo_id, params, is_current FROM edit_version"))
    for version_id, photo_id, raw, is_current in rows.all():
        payload = _load(raw) or {}
        if int(payload.get("params_version", 1)) >= 2:
            continue
        migrated = migrate(dict(payload))
        if is_current:
            current[photo_id] = (_signature(payload), _signature(migrated))
        connection.execute(
            text("UPDATE edit_version SET params = :p, params_version = :v WHERE id = :id"),
            {"p": _dump(migrated), "v": migrated["params_version"], "id": version_id},
        )

    rows = connection.execute(text("SELECT id, analysis, prediction FROM photo"))
    for photo_id, raw_analysis, raw_prediction in rows.all():
        analysis = _flip_analysis(_load(raw_analysis))
        prediction = flip_prediction(_load(raw_prediction))
        if prediction and photo_id in current:
            old, new = current[photo_id]
            if old is not None and prediction.get("applied_signature") == old:
                prediction["applied_signature"] = new
        connection.execute(
            text("UPDATE photo SET analysis = :a, prediction = :p WHERE id = :id"),
            {"a": _dump(analysis), "p": _dump(prediction), "id": photo_id},
        )

    rows = connection.execute(text("SELECT id, params, vector, context FROM style_sample"))
    for sample_id, raw_params, raw_vector, raw_context in rows.all():
        params = _load(raw_params)
        connection.execute(
            text("UPDATE style_sample SET params = :p, vector = :v, context = :c WHERE id = :id"),
            {
                "p": _dump(migrate(dict(params))) if params else raw_params,
                "v": _dump(flip_vector(_load(raw_vector))),
                "c": _dump(flip_context(_load(raw_context))),
                "id": sample_id,
            },
        )

    rows = connection.execute(text("SELECT id, builtin_rules FROM style_profile"))
    for profile_id, raw_rules in rows.all():
        rules = _load(raw_rules)
        flipped = flip_rules(rules)
        if flipped is not rules:
            connection.execute(
                text("UPDATE style_profile SET builtin_rules = :r WHERE id = :id"),
                {"r": _dump(flipped), "id": profile_id},
            )
    _log.info("tinta convertita alla convenzione Adobe in tutto il catalogo")
