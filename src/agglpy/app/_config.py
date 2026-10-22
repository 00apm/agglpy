"""Per-image params from mappings shaped like ``agglpy.yml``.

The yml holds defaults and per-image differences. ``merge`` lays an
image's differences over the defaults; ``to_params`` turns the result
into frozen params, reporting every problem with its key path. The
params classes keep all value rules; this module checks only the shape.
"""

import dataclasses
import numbers
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from agglpy import _yaml
from agglpy.errors import ParamsError
from agglpy.params import (
    HCTParams,
    PreprocessParams,
    RegionParams,
)

__all__ = [
    "COMMENTS",
    "NEUTRAL",
    "SECTIONS",
    "ImageParams",
    "Problem",
    "merge",
    "plain",
    "template",
    "to_params",
]


@dataclass(frozen=True)
class ImageParams:
    """The params of one image.

    Attributes:
        region: The analysed part of the image.
        preprocess: Image preprocessing before detection.
        hct: The HCT ranges; None while ``hct.ranges`` is empty
            (detection needs at least one range).
    """

    region: RegionParams
    preprocess: PreprocessParams
    hct: HCTParams | None


SECTIONS: tuple[str, ...] = ("region", "preprocess", "hct")

NEUTRAL = ImageParams(RegionParams(), PreprocessParams(), None)

_SECTION_CLASSES: dict[str, type] = {
    "region": RegionParams,
    "preprocess": PreprocessParams,
    "hct": HCTParams,
}


class Problem(NamedTuple):
    """One problem in a params mapping.

    Attributes:
        path: Where, like ``"hct.ranges[0]"``; ``""`` for the root.
        message: What is wrong.
    """

    path: str
    message: str


def merge(
    default: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Lay ``override`` over ``default``; the inputs stay unchanged.

    A mapping is merged key by key (recursively), any other value,
    lists and None included, replaces the default. The result holds
    plain dicts and lists.
    """
    result: dict[str, Any] = _copy(default)
    for key, value in override.items():
        old = result.get(key)
        if isinstance(value, Mapping) and isinstance(old, Mapping):
            result[key] = merge(old, value)
        else:
            result[key] = _copy(value)
    return result


def _copy(value: Any) -> Any:
    """A deep copy as plain dicts and lists."""
    if isinstance(value, Mapping):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_copy(item) for item in value]
    return value


def plain(value: object, path: str) -> Any:
    """``value`` as plain YAML-ready data: dicts, lists, scalars.

    A params object becomes its full mapping, a tuple a list and a
    NumPy scalar an int, float or bool. Raises ParamsError naming the
    path for anything else.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(dataclasses.asdict(value), path)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ParamsError(f"{path}: key {key!r} is not text")
            result[key] = plain(item, f"{path}.{key}")
        return result
    if isinstance(value, list | tuple):
        return [plain(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    raise ParamsError(
        f"{path}: {type(value).__name__} values are not allowed "
        f"(use text, numbers, true/false, null, lists or mappings)"
    )


def to_params(
    data: Mapping[str, Any],
) -> tuple[ImageParams | None, list[Problem]]:
    """Convert a merged mapping into ``ImageParams``.

    Returns the params and no problems, or None and every problem
    found. ``hct.ranges`` empty or absent gives ``hct = None``.
    """
    problems: list[Problem] = _unknown_keys(data, SECTIONS, "")
    sections: dict[str, Any] = {}
    for name in ("region", "preprocess"):
        value = data.get(name, {})
        if not isinstance(value, Mapping):
            problems.append(
                Problem(name, f"must be a mapping, got {_show(value)}")
            )
            continue
        sections[name] = _build(_SECTION_CLASSES[name], value, name, problems)
    sections["hct"] = _hct(data, problems)
    if problems:
        return None, problems
    return ImageParams(**sections), []


def _hct(data: Mapping[str, Any], problems: list[Problem]) -> Any:
    if "hct" not in data:
        return None
    value = data["hct"]
    if value is None:
        problems.append(
            Problem(
                "hct",
                "hct can't be null; write `ranges: []` to leave "
                "detection unset",
            )
        )
        return None
    if not isinstance(value, Mapping):
        problems.append(
            Problem("hct", f"must be a mapping, got {_show(value)}")
        )
        return None
    if not value.get("ranges"):
        problems.extend(_unknown_keys(value, ("ranges",), "hct"))
        return None
    return _build(HCTParams, value, "hct", problems)


def _build(
    cls: type, data: Mapping[str, Any], path: str, problems: list[Problem]
) -> Any:
    """Construct ``cls`` from ``data``; None if anything is wrong.

    Problems are appended to ``problems``: the dataclass's own first,
    then its children's.
    """
    start = len(problems)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    problems.extend(_unknown_keys(data, tuple(fields), path))
    missing = [
        name
        for name, f in fields.items()
        if name not in data
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    ]
    for name in missing:
        problems.append(Problem(path, f"missing key {name!r}"))

    kwargs: dict[str, Any] = {}
    child_failed = False
    skip_build = bool(missing)
    for name in fields:
        if name not in data:
            continue
        value, ok = _field(hints[name], data[name], f"{path}.{name}", problems)
        kwargs[name] = value
        if not ok:
            child_failed = True
            # A failed optional child is None, so the dataclass is still
            # built to report its own problems; without a required one
            # it would only repeat the child's problem.
            skip_build = skip_build or _is_required(hints[name])
    if skip_build:
        return None
    try:
        obj = cls(**kwargs)
    except ParamsError as e:
        # Reported before the children's problems: a dataclass's own
        # problem comes first.
        problems.insert(start, Problem(path, str(e)))
        return None
    return None if child_failed else obj


def _field(
    hint: Any, value: Any, path: str, problems: list[Problem]
) -> tuple[Any, bool]:
    """Convert one field's value; ``(value, False)`` if it failed.

    A failed optional child gives None, so the parent can still be
    built and its own problems reported.
    """
    args = typing.get_args(hint)
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        inner = next(a for a in args if a is not type(None))
        if not _is_params_class(inner):
            return value, True
        if value is None:
            return None, True
        if not isinstance(value, Mapping):
            problems.append(
                Problem(path, f"must be a mapping or null, got {_show(value)}")
            )
            return None, False
        obj = _build(inner, value, path, problems)
        return obj, obj is not None
    if (
        origin is tuple
        and len(args) == 2
        and args[1] is Ellipsis
        and _is_params_class(args[0])
    ):
        if not isinstance(value, list | tuple):
            problems.append(
                Problem(path, f"must be a list, got {_show(value)}")
            )
            return value, False
        items = []
        ok = True
        for i, item in enumerate(value):
            where = f"{path}[{i}]"
            if not isinstance(item, Mapping):
                problems.append(
                    Problem(where, f"must be a mapping, got {_show(item)}")
                )
                ok = False
                continue
            obj = _build(args[0], item, where, problems)
            ok = ok and obj is not None
            items.append(obj)
        return items, ok
    return value, True


def _is_params_class(hint: Any) -> typing.TypeGuard[type]:
    return isinstance(hint, type) and dataclasses.is_dataclass(hint)


def _is_required(hint: Any) -> bool:
    return type(None) not in typing.get_args(hint)


def _unknown_keys(
    data: Mapping[str, Any], valid: tuple[str, ...], path: str
) -> list[Problem]:
    prefix = f"{path}." if path else ""
    return [
        Problem(f"{prefix}{key}", f"unknown key (valid: {', '.join(valid)})")
        for key in data
        if key not in valid
    ]


def _show(value: Any) -> str:
    return repr(_copy(value))


# One line per template key; a test checks that every key has one.
COMMENTS: dict[str, str] = {
    "region.bottom_fraction": (
        "fraction of the height cut off at the bottom (info bar)"
    ),
    "preprocess.median_blur": "odd kernel size in px, e.g. 3; null = off",
    "preprocess.rolling_ball": "e.g. {radius: 50}; null = off",
    "preprocess.clahe": "e.g. {clip_limit: 2.0, tile_grid: [8, 8]}; null = off",
    "hct.ranges": "required before detection, e.g.\n- {d_min: 8, d_max: 30}",
}

_TEMPLATE_TOP = (
    "agglpy parameters. Lengths in px.\n"
    '"defaults" apply to every image; "images" lists per-image '
    "differences only.\n"
    "A list given for one image replaces the default list."
)
_COMMENT_COLUMN = 32


def template() -> str:
    """The text of a new ``agglpy.yml``: every key at its neutral value.

    Built from the params fields, so it cannot drift from them.
    """
    sections = {
        "region": dataclasses.asdict(NEUTRAL.region),
        "preprocess": dataclasses.asdict(NEUTRAL.preprocess),
        "hct": {"ranges": []},
    }
    defaults = _yaml.new_mapping()
    for name, fields in sections.items():
        section = _yaml.new_mapping()
        for key, value in fields.items():
            section[key] = value
            _yaml.set_comment(
                section, key, COMMENTS[f"{name}.{key}"], _COMMENT_COLUMN
            )
        defaults[name] = section
    doc = _yaml.new_mapping()
    doc["defaults"] = defaults
    doc["images"] = _yaml.new_mapping()
    _yaml.set_top_comment(doc, _TEMPLATE_TOP)
    return _yaml.dumps(doc)
