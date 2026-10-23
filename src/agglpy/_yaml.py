"""Read and write YAML, keeping comments.

The one module that imports the YAML library, so replacing it means
changing one file. Files load in round-trip mode: a person's comments
in ``agglpy.yml`` survive agglpy writing the file. Files agglpy writes
quote the YAML 1.1 words (``yes``, ``on``, ...) and write ``null``, so
PyYAML-based tools read them the same.
"""

import io
import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import CommentMark, MarkedYAMLError, YAMLError
from ruamel.yaml.representer import RoundTripRepresenter
from ruamel.yaml.tokens import CommentToken

from agglpy.errors import FileFormatError

__all__ = [
    "dumps",
    "load",
    "loads",
    "new_mapping",
    "remove",
    "replace",
    "set_comment",
    "set_top_comment",
]

# Words that YAML 1.1 (PyYAML) reads as bools or None.
_YAML11_WORDS = re.compile(
    r"yes|no|on|off|y|n|true|false|null|~", re.IGNORECASE
)


class _Representer(RoundTripRepresenter):
    """Writes ``None`` as ``null`` and quotes the YAML 1.1 words."""


def _represent_none(r: RoundTripRepresenter, data: None) -> Any:
    return r.represent_scalar("tag:yaml.org,2002:null", "null")


def _represent_str(r: RoundTripRepresenter, data: str) -> Any:
    if _YAML11_WORDS.fullmatch(data):
        return r.represent_scalar("tag:yaml.org,2002:str", data, style="'")
    return RoundTripRepresenter.represent_str(r, data)


# A subclass, so the YAML library's own representer stays unchanged.
_Representer.add_representer(type(None), _represent_none)
_Representer.add_representer(str, _represent_str)


def _yaml() -> YAML:
    # A YAML instance is not thread-safe: one per call.
    y = YAML()
    y.Representer = _Representer
    y.width = 4096
    return y


def loads(text: str, source: str) -> Any:
    """Load YAML text, keeping comments.

    Raises FileFormatError on a syntax error or a repeated key, naming
    ``source`` and the line (1-based).
    """
    try:
        return _yaml().load(text)
    except MarkedYAMLError as e:
        mark = e.problem_mark or e.context_mark
        where = f", line {mark.line + 1}" if mark is not None else ""
        problem = e.problem or e.context or "invalid YAML"
        raise FileFormatError(f"{source}{where}: {problem}") from None
    except YAMLError as e:
        raise FileFormatError(f"{source}: {e}") from None


def load(path: Path) -> Any:
    """Load a UTF-8 YAML file (BOM tolerated), keeping comments."""
    text = Path(path).read_text(encoding="utf-8-sig")
    return loads(text, str(path))


def dumps(data: Any) -> str:
    """Write ``data`` as block-style YAML text (UTF-8, not wrapped).

    Comments held by an empty mapping or list (written ``{}`` / ``[]``)
    move to the lines after it, where they stay valid YAML.
    """
    _settle(data)
    stream = io.StringIO()
    _yaml().dump(data, stream)
    return stream.getvalue()


def new_mapping() -> MutableMapping[str, Any]:
    """An empty mapping that can carry comments."""
    return CommentedMap()


def set_comment(
    mapping: MutableMapping[str, Any], key: str, text: str, column: int
) -> None:
    """Set the end-of-line comment of ``key`` at ``column`` (0-based).

    Each further line of ``text`` becomes a comment line aligned at the
    same column.
    """
    first, *rest = text.split("\n")
    value = f"# {first}\n" + "".join(
        f"{' ' * column}# {line}\n" for line in rest
    )
    token = CommentToken(value, CommentMark(column), None)
    _ca(mapping).items[key] = [None, None, token, None]


def set_top_comment(mapping: MutableMapping[str, Any], text: str) -> None:
    """Set full-line comments above the first key of ``mapping``."""
    _commented(mapping).yaml_set_start_comment(text)


def remove(mapping: MutableMapping[str, Any], key: str) -> None:
    """Delete ``key`` with its own comments, keeping the others.

    The key takes its end-of-line comment and every comment inside its
    block. Comments after it stay: the YAML library stores them on the
    removed key, so they move to the previous key (or above the first
    key). An emptied mapping carries them, and ``dumps`` writes them
    after it.
    """
    cmap = _commented(mapping)
    keys = list(cmap)
    index = keys.index(key)
    kept = _carried(cmap, key) + _kept_after(_end_token(cmap, key))
    kept += _take_end(cmap[key])
    del cmap[key]
    cmap.ca.items.pop(key, None)
    if not cmap:
        cmap.ca.items.clear()
        text = _take_start(cmap) + kept
        if text:
            _set_start(cmap, text)
        return
    if not kept:
        return
    if index > 0:
        _append_after(cmap, keys[index - 1], kept)
    else:
        _prepend_start(cmap, kept)


def replace(mapping: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set ``mapping[key] = value``, keeping the comments after it.

    A new key is appended, after the comment lines that followed the
    last key. A replaced scalar keeps its end-of-line comment; a
    replaced block takes the comments inside it. Comment lines after
    the key stay after it. Plain dicts in ``value`` become mappings
    that can carry comments.
    """
    cmap = _commented(mapping)
    value = _document(value)
    if key not in cmap or not _is_block(cmap[key]):
        # The comment lines after the last key (or after a scalar
        # becoming a block) would end up above the new lines.
        anchor = key if key in cmap else next(reversed(cmap), None)
        moved = "" if anchor is None else _take_after(cmap, anchor)
        cmap[key] = value
        if moved:
            _append_after(cmap, key, moved)
        return
    kept = _kept_after(_end_token(cmap, key)) + _take_end(cmap[key])
    cmap[key] = value
    cmap.ca.items.pop(key, None)
    if kept:
        _append_after(cmap, key, kept)


def _document(value: Any) -> Any:
    if isinstance(value, CommentedMap | CommentedSeq):
        return value
    if isinstance(value, Mapping):
        result = CommentedMap()
        for key, item in value.items():
            result[key] = _document(item)
        return result
    if isinstance(value, list | tuple):
        return CommentedSeq(_document(item) for item in value)
    return value


def _commented(mapping: MutableMapping[str, Any]) -> CommentedMap:
    if not isinstance(mapping, CommentedMap):
        raise TypeError("expected a mapping loaded or made by agglpy._yaml")
    return mapping


def _ca(mapping: MutableMapping[str, Any]) -> Any:
    return _commented(mapping).ca


def _is_block(value: Any) -> bool:
    return (
        isinstance(value, CommentedMap | CommentedSeq)
        and len(value) > 0
        and not value.fa.flow_style()
    )


def _end_slot(container: Any, key: Any) -> tuple[list[Any], int]:
    """The comment slot after the last line of ``container[key]``."""
    value = container[key]
    if _is_block(value):
        last = (
            next(reversed(value))
            if isinstance(value, CommentedMap)
            else len(value) - 1
        )
        return _end_slot(value, last)
    if isinstance(value, CommentedMap | CommentedSeq) and not value:
        # An empty ``{}`` / ``[]`` holds the comment written after it.
        if value.ca.comment is None:
            value.ca.comment = [None, None]
        return value.ca.comment, 0
    slots = container.ca.items.setdefault(key, [None, None, None, None])
    # A mapping keeps the end-of-line comment in slot 2, a sequence in 0.
    return slots, 2 if isinstance(container, CommentedMap) else 0


def _end_token(container: Any, key: Any) -> CommentToken | None:
    slots, i = _end_slot(container, key)
    token: CommentToken | None = slots[i]
    return token


def _kept_after(token: CommentToken | None) -> str:
    """The comment lines after the key's own end-of-line comment."""
    if token is None:
        return ""
    first, _, rest = token.value.partition("\n")
    lines = rest.splitlines(keepends=True)
    if first.lstrip().startswith("#"):
        # Continuation lines aligned with the end-of-line comment belong
        # to it.
        indent = " " * token.column + "#"
        while lines and lines[0].startswith(indent):
            lines.pop(0)
    return "".join(lines)


def _take_after(container: Any, key: Any) -> str:
    """Detach the comment lines after the key's end-of-line comment."""
    slots, i = _end_slot(container, key)
    token = slots[i]
    text = _kept_after(token)
    if text:
        token.value = token.value[: len(token.value) - len(text)]
        if not token.value.strip():
            slots[i] = None
            _drop_shared(container, key, token)
    return text + _take_end(container[key])


def _take_end(value: Any) -> str:
    """Detach the comment lines the YAML library keeps after a block
    (``ca.end``, e.g. after a list ending in ``- {d_min: 5}``)."""
    if not _is_block(value):
        return ""
    last = next(reversed(value)) if isinstance(value, CommentedMap) else -1
    text = _take_end(value[last])
    tokens = getattr(value.ca, "end", None) or []
    text += "".join(_render(t) for t in tokens if t is not None)
    if tokens:
        value.ca.end = []
    return text


def _drop_shared(container: Any, key: Any, token: CommentToken) -> None:
    """Clear the parent's slot 2 of ``key`` when it holds ``token``.

    A loaded ``key: []  # x`` keeps the same token on the parent's
    slot and on the empty value.
    """
    if isinstance(container, CommentedMap):
        slots = container.ca.items.get(key)
        if slots and slots[2] is token:
            slots[2] = None


def _render(token: CommentToken) -> str:
    """A comment token as full lines of text."""
    text: str = token.value
    if text.startswith("\n"):
        return text[1:]
    return " " * token.column + text


def _take_start(value: Any) -> str:
    """Detach the comment lines above the first item of ``value``.

    Slot 0 starts with the end-of-line comment of the line that opens
    ``value`` (``ranges: []  # none``); it belongs to the parent key
    and stays. Its further lines and slot 1 are detached.
    """
    comment = value.ca.comment
    if not comment:
        return ""
    text = ""
    if comment[0] is not None:
        text = _kept_after(comment[0])
        if text:
            comment[0].value = comment[0].value[: -len(text)]
        if not comment[0].value.strip():
            comment[0] = None
    if len(comment) > 1 and comment[1]:
        text += "".join(_render(t) for t in comment[1] if t is not None)
        comment[1] = None
    if comment[0] is None:
        value.ca.comment = None
    return text


def _set_start(value: Any, text: str) -> None:
    """Make ``text`` the comment lines above the first item."""
    if value.ca.comment is None:
        value.ca.comment = [None, None]
    value.ca.comment[1] = [CommentToken(text, CommentMark(0), None)]


def _carried(container: Any, key: Any) -> str:
    """Detach the comments an empty mapping or list carries.

    The YAML library also keeps the comment lines above the first item
    on the parent's slot 3 of the key (the same lines as the value's
    slot 1); an emptied value no longer has a first item.
    """
    value = container[key]
    if not isinstance(value, CommentedMap | CommentedSeq) or value:
        return ""
    text = ""
    if isinstance(container, CommentedMap):
        slots = container.ca.items.get(key)
        if slots and slots[3]:
            comment = value.ca.comment
            if not (comment and len(comment) > 1 and comment[1]):
                text = "".join(_render(t) for t in slots[3] if t is not None)
            slots[3] = None
    comment = value.ca.comment
    eol = comment[0] if comment else None
    text += _take_start(value)
    if eol is not None and not eol.value.strip():
        _drop_shared(container, key, eol)
    if isinstance(container, CommentedMap) and not any(
        container.ca.items.get(key) or ()
    ):
        # An entry of empty slots hides the value's own comment.
        container.ca.items.pop(key, None)
    return text


def _settle(node: Any) -> None:
    """Move the comments of empty mappings and lists after them."""
    if isinstance(node, CommentedMap):
        keys: list[Any] = list(node)
    elif isinstance(node, CommentedSeq):
        keys = list(range(len(node)))
    else:
        return
    for key in keys:
        value = node[key]
        if isinstance(value, CommentedMap | CommentedSeq) and not value:
            # Written {} / [] either way; in flow style its comment goes
            # after it, not between the key and the {}.
            value.fa.set_flow_style()
        text = _carried(node, key)
        if text:
            _append_after(node, key, text)
        _settle(value)


def _append_after(container: Any, key: Any, text: str) -> None:
    slots, i = _end_slot(container, key)
    token = slots[i]
    if token is None:
        slots[i] = CommentToken("\n" + text, CommentMark(0), None)
    else:
        token.value += text


def _prepend_start(cmap: CommentedMap, text: str) -> None:
    if cmap.ca.comment is None:
        cmap.ca.comment = [None, None]
    token = cmap.ca.comment[0]
    if token is None:
        cmap.ca.comment[0] = CommentToken("\n" + text, CommentMark(0), None)
    else:
        token.value += text
