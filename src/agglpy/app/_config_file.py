"""``agglpy.yml``: the params of a collection, defaults and per image.

The file holds ``defaults:`` and ``images:`` (per-image differences
only). ``ConfigFile`` keeps the loaded document, comments included, and
gives each image's params; every problem is reported once, where it was
written.
"""

import re
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from agglpy import _yaml
from agglpy.app._config import (
    NEUTRAL,
    SECTIONS,
    ImageParams,
    Problem,
    check_key,
    merge,
    plain,
    to_params,
)
from agglpy.errors import ParamsError

__all__ = ["CONFIG_NAME", "ConfigFile"]

CONFIG_NAME = "agglpy.yml"

_TOP_KEYS = ("defaults", "images")


class ConfigFile:
    """The loaded ``agglpy.yml``.

    Attributes:
        path: The file it was read from.
    """

    def __init__(self, path: Path, doc: Any) -> None:
        self.path = path
        self._doc = doc

    @classmethod
    def read(cls, path: Path) -> "ConfigFile":
        """Read and shape-check the file.

        Raises FileFormatError on a YAML syntax error and ParamsError
        listing every shape problem (not a mapping, unknown keys,
        image names YAML did not read as text).
        """
        path = Path(path)
        doc = _yaml.load(path)
        if doc is None:
            doc = _yaml.new_mapping()
        config = cls(path, doc)
        _raise(path, _shape_problems(doc))
        return config

    def text(self) -> str:
        """The document as YAML text, comments included."""
        return _yaml.dumps(self._doc)

    def params(self, image: str | None) -> ImageParams:
        """The params of ``image``; ``None`` gives the defaults.

        An image without an entry gets the defaults. Raises ParamsError
        listing the problems of the defaults and of that entry.
        """
        names = [] if image is None else [image]
        defaults, problems = self._problems(names)
        _raise(self.path, problems)
        if image is None:
            return defaults
        params, _ = to_params(merge(self._defaults(), self._entry(image)))
        assert params is not None
        return params

    def check(self, images: Sequence[str]) -> None:
        """Check the defaults and every entry against the sheet's images.

        Raises ParamsError listing every problem: defaults first, then
        the images in the order given, then entries naming an image
        that is not in ``images``.
        """
        _, problems = self._problems(images)
        known = set(images)
        by_folded = {name.casefold(): name for name in images}
        for name in self._entries():
            if name in known:
                continue
            message = "no image of that name in the sheet"
            similar = by_folded.get(name.casefold())
            if similar is not None:
                message += f" (did you mean {similar!r}?)"
            problems.append(Problem(f"images.{name}", message))
        _raise(self.path, problems)

    def set(self, image: str | None, sections: Mapping[str, object]) -> None:
        """Write params for ``image`` (None: the defaults), in memory.

        ``sections`` maps section names to mappings shaped like the
        yml or to params objects; mappings are merged into what is
        there, other values replace it. For an image, each key the
        call set is removed again when the image's params are the same
        without it, so the entry holds differences only. Nothing is
        checked or written here: the caller checks, then writes.
        """
        data: dict[str, Any] = {}
        for name, value in sections.items():
            if name not in SECTIONS:
                raise ParamsError(
                    f"unknown section {name!r} (valid: {', '.join(SECTIONS)})"
                )
            data[name] = plain(value, name)
        if image is None:
            _merge_into(self._section("defaults"), data)
            return
        images = self._section("images")
        if not isinstance(images.get(image), Mapping):
            _yaml.replace(images, image, {})
        _merge_into(images[image], data)
        self._prune(image, _leaf_paths(data))

    def reset(self, image: str, key: str | None = None) -> None:
        """Remove ``key`` (dotted, e.g. ``"preprocess.rolling_ball"``)
        from the entry of ``image``, or the whole entry, in memory.

        Raises ParamsError when ``key`` is not a params key; a key the
        entry does not set changes nothing.
        """
        images = self._entries()
        if key is not None:
            check_key(key)
        if image not in images:
            return
        if key is None:
            _yaml.remove(self._section("images"), image)
            return
        parts = key.split(".")
        self._remove_path(image, parts)
        self._remove_empty(image, parts[:-1])

    def _section(self, name: str) -> Any:
        """The top-level mapping ``name``, created when missing."""
        if not isinstance(self._doc.get(name), Mapping):
            _yaml.replace(self._doc, name, {})
        return self._doc[name]

    def _prune(self, image: str, paths: list[list[str]]) -> None:
        """Remove the touched keys that make no difference.

        Compared on converted params, so 50 equals 50.0. Deepest paths
        first; then emptied mappings on those paths, then an emptied
        entry.
        """
        defaults = self._defaults()
        for parts in sorted(paths, key=len, reverse=True):
            entry = merge({}, self._entry(image))
            parent = _walk(entry, parts[:-1])
            if not isinstance(parent, dict) or parts[-1] not in parent:
                continue
            before, _ = to_params(merge(defaults, entry))
            del parent[parts[-1]]
            after, _ = to_params(merge(defaults, entry))
            if before is None or after != before:
                continue
            self._remove_path(image, parts)
            self._remove_empty(image, parts[:-1])

    def _remove_path(self, image: str, parts: list[str]) -> None:
        parent = _walk(self._entry(image), parts[:-1])
        if isinstance(parent, MutableMapping) and parts[-1] in parent:
            _yaml.remove(parent, parts[-1])

    def _remove_empty(self, image: str, parts: list[str]) -> None:
        """Remove empty mappings along ``parts``, then an empty entry."""
        for depth in range(len(parts), 0, -1):
            parent = _walk(self._entry(image), parts[: depth - 1])
            value = parent.get(parts[depth - 1]) if parent else None
            if isinstance(value, Mapping) and not value:
                _yaml.remove(parent, parts[depth - 1])
        images = self._section("images")
        entry = images.get(image)
        if isinstance(entry, Mapping) and not entry:
            _yaml.remove(images, image)

    def _defaults(self) -> Mapping[str, Any]:
        return self._doc.get("defaults") or {}

    def _entries(self) -> Mapping[str, Any]:
        return self._doc.get("images") or {}

    def _entry(self, image: str) -> Mapping[str, Any]:
        return self._entries().get(image) or {}

    def _problems(
        self, images: Sequence[str]
    ) -> tuple[ImageParams, list[Problem]]:
        """The defaults' params and the problems of defaults + images."""
        defaults = self._defaults()
        params, own = to_params(merge({}, defaults))
        problems = [_prefixed("defaults", p) for p in own]
        for name in images:
            entry = self._entry(name)
            if not entry:
                continue
            _, found = to_params(merge(defaults, entry))
            problems.extend(
                _prefixed(f"images.{name}", p)
                for p in found
                # A problem coming from the defaults is listed once.
                if p not in own and _sets(entry, p.path)
            )
        # Without params the defaults have problems, which are raised.
        return params or NEUTRAL, problems


def _merge_into(target: Any, data: Mapping[str, Any]) -> None:
    for key, value in data.items():
        old = target.get(key)
        if isinstance(value, Mapping) and isinstance(old, Mapping):
            _merge_into(old, value)
        else:
            _yaml.replace(target, key, value)


def _leaf_paths(data: Mapping[str, Any]) -> list[list[str]]:
    """Paths of the values set by ``data``.

    Mappings are walked into; an empty one is a value (pruned too).
    """
    paths: list[list[str]] = []
    for key, value in data.items():
        if isinstance(value, Mapping) and value:
            paths.extend([key, *sub] for sub in _leaf_paths(value))
        else:
            paths.append([key])
    return paths


def _walk(value: Any, parts: list[str]) -> Any:
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _sets(entry: Mapping[str, Any], path: str) -> bool:
    """Whether ``entry`` sets ``path``, a key below it or one above."""
    value: Any = entry
    for part in _split(path):
        if isinstance(value, Mapping) and isinstance(part, str):
            if part not in value:
                return False
            value = value[part]
        else:
            # A list or value set above the path replaces all of it.
            return True
    return True


def _split(path: str) -> list[str | int]:
    parts: list[str | int] = []
    for key, index in re.findall(r"([^.\[\]]+)|\[(\d+)\]", path):
        parts.append(int(index) if index else key)
    return parts


def _shape_problems(doc: Any) -> list[Problem]:
    if not isinstance(doc, Mapping):
        return [Problem("", f"must be a mapping, got {_show(doc)}")]
    problems = [
        Problem(str(key), f"unknown key (valid: {', '.join(_TOP_KEYS)})")
        for key in doc
        if key not in _TOP_KEYS
    ]
    for key in _TOP_KEYS:
        value = doc.get(key)
        if value is not None and not isinstance(value, Mapping):
            problems.append(
                Problem(key, f"must be a mapping, got {_show(value)}")
            )
    images = doc.get("images")
    if isinstance(images, Mapping):
        for name, entry in images.items():
            if not isinstance(name, str):
                problems.append(
                    Problem(
                        "images",
                        f"key {name!r} is not text; quote image names in "
                        f"{CONFIG_NAME}, e.g. '007'",
                    )
                )
            elif entry is not None and not isinstance(entry, Mapping):
                problems.append(
                    Problem(
                        f"images.{name}",
                        f"must be a mapping, got {_show(entry)}",
                    )
                )
    return problems


def _prefixed(prefix: str, problem: Problem) -> Problem:
    path = f"{prefix}.{problem.path}" if problem.path else prefix
    return Problem(path, problem.message)


def _raise(path: Path, problems: list[Problem]) -> None:
    if not problems:
        return
    count = len(problems)
    lines = [
        f"  {p.path}: {p.message}" if p.path else f"  {p.message}"
        for p in problems
    ]
    noun = "problem" if count == 1 else "problems"
    raise ParamsError(f"{path.name} has {count} {noun}:\n" + "\n".join(lines))


def _show(value: Any) -> str:
    return repr(merge({}, {"v": value})["v"])
