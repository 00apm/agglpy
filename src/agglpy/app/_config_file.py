"""``agglpy.yml``: the params of a collection, defaults and per image.

The file holds ``defaults:`` and ``images:`` (per-image differences
only). ``ConfigFile`` keeps the loaded document, comments included, and
gives each image's params; every problem is reported once, where it was
written.
"""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agglpy import _yaml
from agglpy.app._config import (
    NEUTRAL,
    ImageParams,
    Problem,
    merge,
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
