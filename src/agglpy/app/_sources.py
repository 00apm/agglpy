"""Image sources: the files a folder gives, stored paths, the rule.

``find_images`` turns folders and files into image files, each with
the text the filename rule sees (``Found.key``); ``apply_rule`` turns
those keys into image names and pre-filled sheet columns. Paths go
into the sheet relative to the collection (``stored_path``), so a
moved tree keeps working.
"""

import os
import re
import unicodedata
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agglpy.app._sheet import RESERVED_COLUMNS
from agglpy.errors import ParamsError, UnmatchedPathsWarning

__all__ = [
    "IMAGE_SUFFIXES",
    "Found",
    "apply_rule",
    "find_images",
    "resolve_stored",
    "stored_path",
]

IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

_LISTED = 10

PathLike = str | os.PathLike[str]


@dataclass(frozen=True)
class Found:
    """One image file and the text the filename rule sees.

    Attributes:
        path: The file, absolute.
        key: The path relative to the folder given, ``/``-separated,
            NFC; the file name for a file given directly.
    """

    path: Path
    key: str


def find_images(
    images: PathLike | Sequence[PathLike],
    pattern: str | Sequence[str] | None = None,
) -> list[Found]:
    """List the image files of folders and files, in the order given.

    A folder gives the files with an image suffix (any case) directly
    inside it, or with ``pattern`` every file matching the glob(s)
    (``"**/*.tif"`` reaches subfolders); its files are sorted by key,
    so the order is the same on every OS. A file is taken as is. A
    file found twice is listed once.

    Raises:
        FileNotFoundError: If a path does not exist.
    """
    paths = [images] if isinstance(images, str | os.PathLike) else images
    globs = [pattern] if isinstance(pattern, str) else pattern
    result: list[Found] = []
    seen: set[Path] = set()
    for given in map(Path, paths):
        if not given.exists():
            raise FileNotFoundError(f"{given} does not exist")
        if given.is_dir():
            found = [
                Found(p.resolve(), _key(p.relative_to(given)))
                for p in _files(given, globs)
            ]
            found.sort(key=lambda f: f.key)
        else:
            found = [Found(given.resolve(), _key(Path(given.name)))]
        for f in found:
            if f.path not in seen:
                seen.add(f.path)
                result.append(f)
    return result


def _files(folder: Path, globs: Sequence[str] | None) -> list[Path]:
    if globs is None:
        return [
            p
            for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
        ]
    # Several globs may match one file (case-insensitive on Windows).
    matched = {p for g in globs for p in folder.glob(g) if p.is_file()}
    return list(matched)


def _key(relative: Path) -> str:
    # macOS may give file names in NFD: one rule must match everywhere.
    return unicodedata.normalize("NFC", relative.as_posix())


def stored_path(path: Path, folder: Path) -> str:
    """Give ``path`` as stored in the sheet: relative to ``folder``.

    ``/``-separated on every OS; absolute when ``path`` is on another
    drive than ``folder``.
    """
    try:
        return Path(os.path.relpath(path, folder)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def resolve_stored(text: str, folder: Path) -> Path:
    """Give the absolute file of a path stored in the sheet."""
    return (folder / text).resolve()


def apply_rule(
    found: Sequence[Found], rule: str | None
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Match the filename rule against each file's key.

    ``rule`` is a regular expression with named groups, searched in
    the key. The group ``image`` sets the image name (else the file
    stem); the other groups are sheet columns.

    Returns:
        ``(names, columns, values)``: one name per file, the rule's
        columns in pattern order, and per file the columns' values
        (``{}`` when the key does not match).

    Raises:
        ParamsError: If ``rule`` is not a valid regular expression or
            a group is named like a reserved column.
    """
    names = [unicodedata.normalize("NFC", f.path.stem) for f in found]
    if rule is None:
        return names, [], [{} for _ in found]
    try:
        compiled = re.compile(rule)
    except re.error as e:
        raise ParamsError(
            f"columns_from_path {rule!r} is not a valid regular "
            f"expression: {e}"
        ) from None
    groups = sorted(compiled.groupindex, key=compiled.groupindex.__getitem__)
    reserved = [g for g in groups if g in RESERVED_COLUMNS and g != "image"]
    if reserved:
        raise ParamsError(
            f"columns_from_path: group names {reserved} are reserved "
            "column names; rename the groups"
        )
    columns = [g for g in groups if g != "image"]
    values: list[dict[str, str]] = []
    unmatched: list[str] = []
    for i, f in enumerate(found):
        match = compiled.search(f.key)
        if match is None:
            unmatched.append(f.key)
            values.append({})
            continue
        if match.groupdict().get("image") is not None:
            names[i] = match["image"]
        values.append({c: match[c] for c in columns if match[c] is not None})
    if unmatched:
        listed = ", ".join(unmatched[:_LISTED])
        if len(unmatched) > _LISTED:
            listed += f" and {len(unmatched) - _LISTED} more"
        warnings.warn(
            f"{len(unmatched)} image paths did not match columns_from_path "
            f"{rule!r}; their columns were left empty: {listed}",
            UnmatchedPathsWarning,
            stacklevel=2,
        )
    return names, columns, values
