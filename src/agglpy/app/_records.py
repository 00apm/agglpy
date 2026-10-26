"""Particle records: what each particle file was made from.

A record sits next to its particle file in ``agglpy/<name>/`` and
stores the image (hash, size, time, pixel size), the params or the
imported source, the versions and the warnings of the run. Records are
read strictly: anything agglpy can't read exactly as it writes it is
treated as missing, with an ``UnreadableRecordWarning``.
"""

import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from agglpy import _yaml
from agglpy.app._config import ImageParams, plain, to_params
from agglpy.app._files import replace_file
from agglpy.errors import FileFormatError, UnreadableRecordWarning
from agglpy.export import _dependency_versions, _version
from agglpy.io import read_image_metadata

__all__ = [
    "RECORD_FORMAT",
    "CorrectionRecord",
    "DetectionRecord",
    "ImageInfo",
    "ParticlesInfo",
    "SourceInfo",
    "StartedFrom",
    "image_info",
    "missing_image",
    "nanometres",
    "now",
    "read_record",
    "record_text",
    "same_image",
    "sha256",
    "versions",
    "write_record",
]

RECORD_FORMAT = 1


@dataclass(frozen=True)
class ImageInfo:
    """The image a particle set was made from.

    Every field but ``file`` is None when the image was missing.
    """

    file: str | None
    sha256: str | None
    size: int | None
    modified: int | None  # st_mtime_ns
    width: int | None
    height: int | None
    pixel_size_nm: float | None


@dataclass(frozen=True)
class ParticlesInfo:
    """The particle file a record belongs to."""

    file: str
    count: int
    sha256: str


@dataclass(frozen=True)
class SourceInfo:
    """The file a correction was imported from."""

    path: str
    format: str
    sha256: str


@dataclass(frozen=True)
class StartedFrom:
    """The detection present when a correction was imported."""

    sha256: str
    detected: str | None  # ``created`` of its record
    kept: int
    detection_count: int


@dataclass(frozen=True)
class DetectionRecord:
    """The record of ``particles_hct.csv``."""

    created: str
    agglpy: str | None
    dependencies: Mapping[str, str | None]
    image: ImageInfo
    params: ImageParams
    particles: ParticlesInfo
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CorrectionRecord:
    """The record of ``particles_corrected.csv``."""

    created: str
    agglpy: str | None
    dependencies: Mapping[str, str | None]
    image: ImageInfo
    particles: ParticlesInfo
    warnings: tuple[str, ...]
    source: SourceInfo
    started_from: StartedFrom | None


Record = DetectionRecord | CorrectionRecord
R = TypeVar("R", DetectionRecord, CorrectionRecord)

_KINDS: dict[type, str] = {
    DetectionRecord: "detection",
    CorrectionRecord: "correction",
}
_MADE_BY = {
    DetectionRecord: "detecting particles",
    CorrectionRecord: "importing corrected particles",
}


# ---------------------------------------------------------------------
# Facts and helpers
# ---------------------------------------------------------------------


def sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes, as hex."""
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def image_info(stored: str, path: Path) -> ImageInfo:
    """Hash, size, time and metadata of the image at ``path``.

    ``stored`` is the path as the sheet holds it.

    Raises:
        FileNotFoundError: If the image doesn't exist.
        FileFormatError: If its header or pixels can't be read.
    """
    meta = read_image_metadata(path)
    stat = path.stat()
    return ImageInfo(
        file=stored,
        sha256=sha256(path),
        size=stat.st_size,
        modified=stat.st_mtime_ns,
        width=meta.width,
        height=meta.height,
        pixel_size_nm=nanometres(meta.pixel_size),
    )


def nanometres(metres: float | None) -> float | None:
    """A pixel size in m as nm; None stays None.

    Rounded to 12 digits: drops the float noise of the conversion, far
    below any calibration.
    """
    return None if metres is None else float(f"{metres * 1e9:.12g}")


def missing_image(stored: str) -> ImageInfo:
    """The image facts of an image that was missing: only the path."""
    return ImageInfo(stored, None, None, None, None, None, None)


def same_image(info: ImageInfo, path: Path) -> bool:
    """Whether the file at ``path`` is the image ``info`` describes.

    Equal size and time are trusted without hashing; otherwise (a copy
    or a clone sets new times) the bytes are hashed again.
    """
    stat = path.stat()
    if stat.st_size == info.size and stat.st_mtime_ns == info.modified:
        return True
    return info.sha256 is not None and sha256(path) == info.sha256


def now() -> str:
    """The local date and time, ISO 8601 to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def versions() -> tuple[str | None, dict[str, str | None]]:
    """agglpy's version and those of its main dependencies."""
    return _version(("agglpy",)), _dependency_versions()


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------


def record_text(record: Record) -> str:
    """The record as YAML text, with its header comment."""
    data = _yaml.new_mapping()
    data["kind"] = _KINDS[type(record)]
    data["format"] = RECORD_FORMAT
    data["created"] = record.created
    data["agglpy"] = record.agglpy
    data["dependencies"] = dict(record.dependencies)
    data["image"] = plain(record.image, "image")
    if isinstance(record, DetectionRecord):
        data["params"] = plain(record.params, "params")
    data["particles"] = plain(record.particles, "particles")
    if isinstance(record, CorrectionRecord):
        data["source"] = plain(record.source, "source")
        data["started_from"] = plain(record.started_from, "started_from")
    data["warnings"] = list(record.warnings)
    version = f" {record.agglpy}" if record.agglpy else ""
    _yaml.set_top_comment(
        data,
        f"Written by agglpy{version} after {_MADE_BY[type(record)]}. "
        "Do not edit: agglpy compares\n"
        f"it with {record.particles.file} and the image to tell whether "
        "they changed.",
    )
    return _yaml.dumps(data)


def write_record(path: Path, record: Record) -> None:
    """Write ``record`` to ``path``, replacing it safely."""
    text = record_text(record)
    replace_file(path, lambda tmp: tmp.write_bytes(text.encode("utf-8")))


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------


class _UnreadableError(Exception):
    """A record that doesn't read back as agglpy writes it."""


def read_record(path: Path, kind: type[R]) -> R | None:
    """Read the record at ``path``; None when missing or unreadable.

    A missing file gives None quietly; any other problem gives None
    and an ``UnreadableRecordWarning`` naming the file and the reason.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
        data = _yaml.loads(text, path.name)
        return cast(R, _parse(data, kind))
    except FileNotFoundError:
        return None
    except (FileFormatError, _UnreadableError) as exc:
        reason = str(exc)
    # another encoding, a lock, a folder of that name
    except (OSError, UnicodeDecodeError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
    warnings.warn(
        f"{path.name} in {path.parent.name}/ unreadable: {reason}; "
        "treated as missing",
        UnreadableRecordWarning,
        stacklevel=2,
    )
    return None


def _parse(data: Any, kind: type[Record]) -> Record:
    fields = _mapping(data, "the record")
    expected = _KINDS[kind]
    if fields.get("kind") != expected:
        raise _UnreadableError(
            f"kind is {fields.get('kind')!r}, expected {expected!r}"
        )
    if _int(fields, "format") != RECORD_FORMAT:
        raise _UnreadableError(
            f"format {fields['format']} is not {RECORD_FORMAT} "
            "(written by a newer agglpy?)"
        )
    common = {
        "created": _str(fields, "created"),
        "agglpy": _str(fields, "agglpy", optional=True),
        "dependencies": _dependencies(fields),
        "image": _image(_section(fields, "image")),
        "particles": _particles(_section(fields, "particles")),
        "warnings": _warnings(fields),
    }
    if kind is DetectionRecord:
        _known(fields, "", (*_COMMON_KEYS, "params"))
        return DetectionRecord(params=_params(fields), **common)
    _known(fields, "", (*_COMMON_KEYS, "source", "started_from"))
    started = fields.get("started_from", ...)
    if started is ...:
        raise _UnreadableError("started_from is missing")
    return CorrectionRecord(
        source=_source(_section(fields, "source")),
        started_from=None
        if started is None
        else _started(_mapping(started, "started_from")),
        **common,
    )


_COMMON_KEYS = (
    "kind",
    "format",
    "created",
    "agglpy",
    "dependencies",
    "image",
    "particles",
    "warnings",
)


def _image(data: Mapping[str, Any]) -> ImageInfo:
    _known(data, "image", ImageInfo.__dataclass_fields__)
    return ImageInfo(
        file=_str(data, "file", "image", optional=True),
        sha256=_str(data, "sha256", "image", optional=True),
        size=_int(data, "size", "image", optional=True),
        modified=_int(data, "modified", "image", optional=True),
        width=_int(data, "width", "image", optional=True),
        height=_int(data, "height", "image", optional=True),
        pixel_size_nm=_float(data, "pixel_size_nm", "image"),
    )


def _particles(data: Mapping[str, Any]) -> ParticlesInfo:
    _known(data, "particles", ParticlesInfo.__dataclass_fields__)
    return ParticlesInfo(
        file=_str(data, "file", "particles"),
        count=_int(data, "count", "particles"),
        sha256=_str(data, "sha256", "particles"),
    )


def _source(data: Mapping[str, Any]) -> SourceInfo:
    _known(data, "source", SourceInfo.__dataclass_fields__)
    return SourceInfo(
        path=_str(data, "path", "source"),
        format=_str(data, "format", "source"),
        sha256=_str(data, "sha256", "source"),
    )


def _started(data: Mapping[str, Any]) -> StartedFrom:
    _known(data, "started_from", StartedFrom.__dataclass_fields__)
    return StartedFrom(
        sha256=_str(data, "sha256", "started_from"),
        detected=_str(data, "detected", "started_from", optional=True),
        kept=_int(data, "kept", "started_from"),
        detection_count=_int(data, "detection_count", "started_from"),
    )


def _params(fields: Mapping[str, Any]) -> ImageParams:
    params, problems = to_params(_section(fields, "params"))
    if problems:
        path, message = problems[0]
        raise _UnreadableError(f"params.{path}: {message}")
    assert params is not None
    if params.hct is None:
        raise _UnreadableError("params.hct has no ranges")
    return params


def _dependencies(fields: Mapping[str, Any]) -> dict[str, str | None]:
    data = _section(fields, "dependencies")
    return {
        str(key): _str(data, key, "dependencies", optional=True)
        for key in data
    }


def _warnings(fields: Mapping[str, Any]) -> tuple[str, ...]:
    items = fields.get("warnings")
    if not isinstance(items, list) or not all(
        isinstance(item, str) for item in items
    ):
        raise _UnreadableError("warnings must be a list of text")
    return tuple(items)


# ---------------------------------------------------------------------
# Typed field access: each raises _UnreadableError naming the key
# ---------------------------------------------------------------------


def _where(key: str, parent: str) -> str:
    return f"{parent}.{key}" if parent else key


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _UnreadableError(f"{what} is not a mapping")
    return value


def _section(
    data: Mapping[str, Any], key: str, parent: str = ""
) -> Mapping[str, Any]:
    if key not in data:
        raise _UnreadableError(f"{_where(key, parent)} is missing")
    return _mapping(data[key], _where(key, parent))


def _known(
    data: Mapping[str, Any], parent: str, keys: Mapping[str, Any] | tuple
) -> None:
    unknown = [key for key in data if key not in keys]
    if unknown:
        raise _UnreadableError(
            f"unknown key {_where(str(unknown[0]), parent)}"
        )


def _value(data: Mapping[str, Any], key: str, parent: str) -> Any:
    if key not in data:
        raise _UnreadableError(f"{_where(key, parent)} is missing")
    return data[key]


def _str(
    data: Mapping[str, Any], key: str, parent: str = "", optional: bool = False
) -> Any:
    value = _value(data, key, parent)
    if isinstance(value, str) or (optional and value is None):
        return str(value) if value is not None else None
    raise _UnreadableError(f"{_where(key, parent)} is not text: {value!r}")


def _int(
    data: Mapping[str, Any], key: str, parent: str = "", optional: bool = False
) -> Any:
    value = _value(data, key, parent)
    if optional and value is None:
        return None
    # bool is an int in Python; true is no count
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    raise _UnreadableError(
        f"{_where(key, parent)} is not a whole number: {value!r}"
    )


def _float(data: Mapping[str, Any], key: str, parent: str) -> float | None:
    value = _value(data, key, parent)
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise _UnreadableError(f"{_where(key, parent)} is not a number: {value!r}")
