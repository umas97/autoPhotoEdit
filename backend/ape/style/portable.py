# autoPhotoEdit -- automatic post-production for Sony RAW files.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``.apestyle``: a profile in one file, to move between machines (section 20.1).

A zip with:

``format.json``
    ``{"format": "apestyle", "format_version": 2}`` -- read first, so a file
    from a newer build is refused with a sentence rather than half-imported.
    Version 1 had the tint the other way round and imports converted
    (``style/tint_sign.py``);
``profile.json``
    name, notes, rules, normalisation statistics, and one entry per sample
    (original paths, parameters, style vector, context, ΔE, exclusion);
``model.npz``
    the trained model exactly as the catalogue stores it;
``samples.npz``
    the samples' scene features and embeddings;
``thumbnails/<n>.jpg``
    the 512 px references, which cover the "Sample di riferimento" panel of the
    review when the RAWs are not on this machine (section 20.1).

Importing never needs the RAWs, and a model imported is byte-for-byte the
model exported but for the sample ids it refers to, which are renumbered to
the new rows: the predictions are identical (test 15).

Everything is plain JSON and ``allow_pickle=False`` arrays: opening a profile
someone sent you must not be able to run code. The archive is read member by
member with a size cap, and only the names above are ever looked at.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..analysis.embed import EMBEDDING_DIM, decode_embedding, encode_embedding
from ..analysis.scene import FEATURE_NAMES, decode_features, encode_features
from ..db.models import StyleProfile, StyleSample, StyleSampleStatus
from .model import StyleModel

__all__ = ["FORMAT_VERSION", "PortableError", "export_profile", "import_profile"]

#: 2: the tint changed sign (``raw/whitepoint.py``).
FORMAT_VERSION = 2
#: A profile of 200 pairs is a few MB; anything past this is not one.
MAX_UNCOMPRESSED = 256 * 1024 * 1024


class PortableError(ValueError):
    """The file is not a profile this build can import."""



def _without_retouch(params: dict | None) -> dict:
    """A sample's parameters without the removals of the photo it came from."""
    if not params or "retouch" not in params:
        return dict(params or {})
    return {key: value for key, value in params.items() if key != "retouch"}

def export_profile(session: Session, profile: StyleProfile) -> bytes:
    samples = list(
        session.scalars(
            select(StyleSample)
            .where(
                StyleSample.profile_id == profile.id,
                StyleSample.status != StyleSampleStatus.PROPOSED,
            )
            .order_by(StyleSample.id)
        )
    )
    entries = []
    features = np.full((len(samples), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    embeddings = np.full((len(samples), EMBEDDING_DIM), np.nan, dtype=np.float32)
    for index, sample in enumerate(samples):
        vector = decode_features(sample.scene_features)
        if vector is not None:
            features[index] = vector
        embedding = decode_embedding(sample.embedding)
        if embedding is not None:
            embeddings[index] = embedding
        entries.append(
            {
                "id": sample.id,
                "raw_path": sample.raw_path,
                "reference_path": sample.reference_path,
                # A removal is its photo's own (docs/SPEC_rimozione.md R6): it
                # never travels with a style.
                "params": _without_retouch(sample.params),
                "vector": sample.vector,
                "context": sample.context,
                "residual_loss": sample.residual_loss,
                "delta_e": sample.delta_e,
                "excluded": bool(sample.excluded),
                "status": sample.status.value,
                "error": sample.error,
                "pairing": sample.pairing,
                "has_features": vector is not None,
                "has_embedding": embedding is not None,
                "has_thumbnail": sample.thumbnail is not None,
            }
        )
    document = {
        "name": profile.name,
        "notes": profile.notes,
        "builtin_rules": profile.builtin_rules,
        "norm_stats": profile.norm_stats,
        "n_pairs": profile.n_pairs,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "trained_at": profile.trained_at.isoformat() if profile.trained_at else None,
        "from_builtin": bool(profile.builtin),
        "samples": entries,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "format.json", json.dumps({"format": "apestyle", "format_version": FORMAT_VERSION})
        )
        archive.writestr("profile.json", json.dumps(document, ensure_ascii=False, indent=1))
        if profile.model:
            archive.writestr("model.npz", profile.model)
        arrays = io.BytesIO()
        np.savez_compressed(arrays, features=features, embeddings=embeddings)
        archive.writestr("samples.npz", arrays.getvalue())
        for index, sample in enumerate(samples):
            if sample.thumbnail:
                archive.writestr(f"thumbnails/{index}.jpg", sample.thumbnail)
    return buffer.getvalue()


def _read(archive: zipfile.ZipFile, name: str, budget: list[int]) -> bytes | None:
    try:
        info = archive.getinfo(name)
    except KeyError:
        return None
    budget[0] -= info.file_size
    if budget[0] < 0:
        raise PortableError("file troppo grande per essere un profilo di stile")
    return archive.read(info)


def _timestamp(value: str | None):
    from datetime import UTC, datetime

    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def _unique_name(session: Session, wanted: str) -> str:
    name, n = wanted, 1
    while session.scalars(select(StyleProfile.id).where(StyleProfile.name == name)).first():
        n += 1
        name = f"{wanted} ({n})"
    return name


def _flip_tint(
    document: dict, entries: list[dict], features: np.ndarray | None
) -> tuple[dict, list[dict], np.ndarray | None]:
    """A version 1 profile in the new tint sign. The model flips when it loads."""
    from ..pipeline.params import migrate
    from .tint_sign import flip_context, flip_rules, flip_vector

    document = {**document, "builtin_rules": flip_rules(document.get("builtin_rules"))}
    entries = [
        {
            **entry,
            "params": migrate(dict(entry["params"])) if entry.get("params") else {},
            "vector": flip_vector(entry.get("vector")),
            "context": flip_context(entry.get("context")),
        }
        for entry in entries
    ]
    if features is not None:
        features = np.array(features, dtype=np.float32)
        column = FEATURE_NAMES.index("tint")
        features[:, column] = np.float32(0.0) - features[:, column]
    return document, entries, features


def import_profile(session: Session, data: bytes, *, name: str | None = None) -> StyleProfile:
    """Create a new profile from an ``.apestyle``. Never overwrites an existing one.

    Raises:
        PortableError: not an ``.apestyle``, a newer format, or damaged.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise PortableError("il file non è un profilo di stile (.apestyle)") from exc
    budget = [MAX_UNCOMPRESSED]
    with archive:
        header = _read(archive, "format.json", budget)
        try:
            fmt = json.loads(header or b"{}")
        except json.JSONDecodeError as exc:
            raise PortableError("profilo danneggiato") from exc
        if fmt.get("format") != "apestyle":
            raise PortableError("il file non è un profilo di stile (.apestyle)")
        version = int(fmt.get("format_version", 0))
        if version > FORMAT_VERSION:
            raise PortableError(
                "il profilo è stato esportato da una versione più recente: aggiorna autoPhotoEdit"
            )
        try:
            document = json.loads(_read(archive, "profile.json", budget) or b"")
        except json.JSONDecodeError as exc:
            raise PortableError("profilo danneggiato") from exc
        model_blob = _read(archive, "model.npz", budget)
        arrays_blob = _read(archive, "samples.npz", budget)
        entries = document.get("samples") or []
        features = embeddings = None
        if arrays_blob:
            with np.load(io.BytesIO(arrays_blob), allow_pickle=False) as arrays:
                features, embeddings = arrays["features"], arrays["embeddings"]
        if version < 2:
            document, entries, features = _flip_tint(document, entries, features)
        thumbnails = [
            _read(archive, f"thumbnails/{i}.jpg", budget) if e.get("has_thumbnail") else None
            for i, e in enumerate(entries)
        ]

    model = None
    if model_blob:
        try:
            model = StyleModel.from_bytes(model_blob)
        except (ValueError, KeyError, OSError) as exc:
            raise PortableError(f"modello del profilo non utilizzabile: {exc}") from exc

    profile = StyleProfile(
        name=_unique_name(session, name or document.get("name") or "Profilo importato"),
        notes=document.get("notes"),
        # An exported built-in comes back as a user copy: built-ins are the
        # program's, and exist already on every machine.
        builtin=False,
        builtin_rules=document.get("builtin_rules"),
        norm_stats=document.get("norm_stats"),
        n_pairs=int(document.get("n_pairs") or 0),
        trained_at=_timestamp(document.get("trained_at")),
    )
    session.add(profile)
    session.flush()

    renumber: dict[int, int] = {}
    for index, entry in enumerate(entries):
        status = entry.get("status") or StyleSampleStatus.READY.value
        row = StyleSample(
            profile_id=profile.id,
            raw_path=entry.get("raw_path"),
            reference_path=entry.get("reference_path"),
            params=_without_retouch(entry.get("params") or {}),
            vector=entry.get("vector"),
            context=entry.get("context"),
            residual_loss=entry.get("residual_loss"),
            delta_e=entry.get("delta_e"),
            excluded=bool(entry.get("excluded")),
            status=StyleSampleStatus(status),
            error=entry.get("error"),
            pairing=entry.get("pairing"),
            thumbnail=thumbnails[index],
        )
        if features is not None and entry.get("has_features"):
            row.scene_features = encode_features(features[index])
        if embeddings is not None and entry.get("has_embedding"):
            row.embedding = encode_embedding(embeddings[index])
        session.add(row)
        session.flush()
        renumber[int(entry["id"])] = row.id

    if model is not None:
        model.sample_ids = np.array(
            [renumber.get(int(i), -1) for i in model.sample_ids], dtype=np.int64
        )
        profile.model = model.to_bytes()
        from .profile import retrain_centroid

        retrain_centroid(session, profile)
    session.flush()
    return profile
