"""Writing result tables with their record to a folder (the export edge).

``write_results`` writes ``results.xlsx`` (one sheet per table, then
``units`` and ``info``), every table as ``csv/<name>.csv`` and the
record as ``record.yml``. It knows nothing about images or collections:
the caller passes the tables, what to record (params, which particle
set was used) and the unit of each column; the edge adds the versions
of agglpy and its main dependencies to every export.
"""

import dataclasses
import os
import re
import shutil
from collections.abc import Mapping
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from agglpy import _yaml
from agglpy.errors import (
    OutputExistsError,
    OutputLockedError,
    ParamsError,
    TableError,
)

PathLike = str | os.PathLike[str]

# Data rows of an Excel sheet (1,048,576 rows with the header).
_XLSX_ROWS = 1_048_575
# The files and folder of an export; nothing else in the folder is
# touched.
_XLSX = "results.xlsx"
_RECORD = "record.yml"
_CSV = "csv"
# Excel sheet names: at most 31 characters; letters, digits, _ and -
# also make safe file names on every OS.
_NAME = re.compile(r"[A-Za-z0-9_-]{1,31}")
_RESERVED = ("info", "units")
# Dependencies whose version goes into every record: record key and the
# distributions that provide it (OpenCV comes in several wheels).
_VERSIONS = {
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "opencv": (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ),
    "scipy": ("scipy",),
    "scikit-image": ("scikit-image",),
    "matplotlib": ("matplotlib",),
}


def write_results(
    folder: PathLike,
    tables: Mapping[str, pd.DataFrame],
    *,
    record: Mapping[str, object] | None = None,
    units: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write result tables, their units and a record into a folder.

    Writes into ``folder`` (created with its parents if missing):

    - ``results.xlsx``: one sheet per table in the given order, then
      ``units`` (with ``units=``) and ``info`` (the record as
      ``key | value`` rows, e.g. ``params.hct.ranges.0.d_min | 3``). A
      table longer than Excel's 1,048,575 rows is written as CSV only;
      the info sheet lists it.
    - ``csv/<name>.csv`` for every table, and ``csv/units.csv``: UTF-8,
      comma separated, ``.`` as decimal mark, a header row and no index
      column. A table whose index is not the default ``0 … n-1``
      (e.g. a ``groupby`` result) is written with its index as columns.
    - ``record.yml``: the record, under ``agglpy`` the agglpy version,
      the date and time, the versions of the main dependencies and the
      tables written.

    Column names stay as in the tables; units go into the ``units``
    sheet and ``csv/units.csv`` (``table | column | unit``). A write
    that fails part-way removes the files it wrote.

    Args:
        folder: The folder to write into.
        tables: ``{name: table}``; names of letters, digits, ``_`` and
            ``-``, at most 31 characters (Excel's sheet limit), not
            ``info`` or ``units``.
        record: What the results were made from, e.g. ``{"params":
            ..., "particle_set": {"D7-017": "corrected"}}``: text,
            numbers, bools, None, lists, dicts with text keys, and
            dataclasses such as the params (written as dicts).
        units: ``{column: unit}`` for all tables, e.g. ``{"D": "m"}``;
            columns without an entry are not listed.
        overwrite: Replace an earlier export in ``folder``. Only
            ``results.xlsx``, ``record.yml`` and ``csv/`` are removed;
            other files are never touched.

    Returns:
        The folder.

    Raises:
        TableError: If a name is not allowed or a table is not a
            DataFrame or has MultiIndex columns.
        ParamsError: If ``record`` has its own ``agglpy`` entry.
        TypeError: If a record value or a unit is of another type.
        OutputExistsError: If ``folder`` holds an export and
            ``overwrite`` is False (nothing is written).
        OutputLockedError: If a file of the earlier export can't be
            removed, e.g. open in Excel (nothing is changed).
    """
    _check_names(tables)
    written = {name: _with_index(table) for name, table in tables.items()}
    unit_rows = _unit_rows(written, units or {})
    caller = _plain(dict(record or {}), "record")
    if "agglpy" in caller:
        raise ParamsError("record: the key 'agglpy' is added by agglpy")
    in_xlsx = {
        name: len(table) <= _XLSX_ROWS for name, table in written.items()
    }
    full = {
        "agglpy": {
            "version": _version(("agglpy",)),
            "exported": datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "versions": _dependency_versions(),
            "tables": [
                {
                    "name": name,
                    "rows": len(table),
                    "written": "xlsx" if in_xlsx[name] else "csv only",
                }
                for name, table in written.items()
            ],
        },
        **caller,
    }

    text = _yaml.dumps(full)

    out = Path(folder)
    _clear(out, overwrite)
    out.mkdir(parents=True, exist_ok=True)
    try:
        (out / _CSV).mkdir()
        for name, table in written.items():
            _write_csv(table, out / _CSV / f"{name}.csv")
        if unit_rows is not None:
            _write_csv(unit_rows, out / _CSV / "units.csv")
        with pd.ExcelWriter(out / _XLSX, engine="openpyxl") as excel:
            for name, table in written.items():
                if in_xlsx[name]:
                    table.to_excel(excel, sheet_name=name, index=False)
            if unit_rows is not None:
                unit_rows.to_excel(excel, sheet_name="units", index=False)
            _info(full).to_excel(excel, sheet_name="info", index=False)
        with (out / _RECORD).open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    except BaseException:
        # A half-written export would block the retry: remove it.
        _remove(_export_items(out))
        raise
    return out


def _check_names(tables: Mapping[str, pd.DataFrame]) -> None:
    seen: set[str] = set()
    for name, table in tables.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise TableError(
                f"table name {name!r}: use letters, digits, _ and -, "
                "at most 31 characters"
            )
        # Excel and the Windows file system ignore case.
        if name.lower() in _RESERVED:
            raise TableError(f"table name {name!r} is reserved")
        if name.lower() in seen:
            raise TableError(f"table name {name!r} given twice")
        seen.add(name.lower())
        if not isinstance(table, pd.DataFrame):
            raise TableError(f"table {name!r} is not a DataFrame")
        if isinstance(table.columns, pd.MultiIndex):
            raise TableError(
                f"table {name!r} has MultiIndex columns; flatten them, "
                "e.g. table.columns = ['_'.join(c) for c in table.columns]"
            )


def _with_index(table: pd.DataFrame) -> pd.DataFrame:
    index = table.index
    if (
        isinstance(index, pd.RangeIndex)
        and index.start == 0
        and index.step == 1
        and index.name is None
    ):
        return table
    # the index holds data (e.g. the groups of a summary): keep it
    return table.reset_index()


def _unit_rows(
    tables: Mapping[str, pd.DataFrame], units: Mapping[str, str]
) -> pd.DataFrame | None:
    if not units:
        return None
    for column, unit in units.items():
        if not isinstance(unit, str):
            raise TypeError(f"units: {column!r} has {unit!r}, not text")
    rows = [
        (name, column, units[column])
        for name, table in tables.items()
        for column in table.columns
        if column in units
    ]
    return pd.DataFrame(rows, columns=["table", "column", "unit"])


def _plain(value: object, key: str) -> Any:
    """``value`` as plain YAML types; ``key`` names it in errors."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    # NumPy numbers and arrays as Python numbers and lists; checked
    # first: np.float64 is also a float, which YAML can't write.
    if type(value).__module__ == "numpy" and hasattr(value, "tolist"):
        return _plain(value.tolist(), key)
    # Exact types: a subclass (e.g. a str Enum) is not plain YAML.
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"{key}: key {k!r} is not text")
            out[k] = _plain(v, f"{key}.{k}")
        return out
    if isinstance(value, list | tuple):
        return [_plain(v, f"{key}.{i}") for i, v in enumerate(value)]
    raise TypeError(
        f"{key}: {type(value).__name__} can't be recorded; use text, "
        "numbers, bools, None, lists, dicts or dataclasses"
    )


def _info(record: Mapping[str, Any]) -> pd.DataFrame:
    """The record as ``key | value`` rows, dotted keys, list positions."""
    rows: list[tuple[str, object]] = []

    def walk(value: object, key: str) -> None:
        if isinstance(value, Mapping) and value:
            for k, v in value.items():
                walk(v, f"{key}.{k}" if key else k)
        elif isinstance(value, list) and value:
            for i, v in enumerate(value):
                walk(v, f"{key}.{i}")
        elif isinstance(value, Mapping | list):
            rows.append((key, "{}" if isinstance(value, Mapping) else "[]"))
        else:
            rows.append((key, value))

    walk(record, "")
    return pd.DataFrame(rows, columns=["key", "value"])


def _dependency_versions() -> dict[str, str | None]:
    """The installed version of each dependency in ``_VERSIONS``."""
    return {key: _version(names) for key, names in _VERSIONS.items()}


def _version(names: tuple[str, ...]) -> str | None:
    for name in names:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return None


def _clear(folder: Path, overwrite: bool) -> None:
    """Remove an earlier export, or refuse; never anything else."""
    old = _export_items(folder)
    if not old:
        return
    names = ", ".join(p.name for p in old)
    if not overwrite:
        raise OutputExistsError(
            f"{folder} already holds an export ({names}); pass "
            "overwrite=True to replace it"
        )
    # Renaming fails on a file another program holds open (Windows):
    # try each first, so a locked file stops the export before anything
    # is removed.
    for path in old:
        moved = path.with_name(path.name + ".agglpy-check")
        try:
            path.rename(moved)
        except PermissionError as exc:
            raise OutputLockedError(
                f"can't replace {path}: is it open in another program? "
                "Close it and retry"
            ) from exc
        moved.rename(path)
    _remove(old)


def _export_items(folder: Path) -> list[Path]:
    """The files and folder of an export present in ``folder``."""
    return [
        folder / n for n in (_XLSX, _RECORD, _CSV) if (folder / n).exists()
    ]


def _remove(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
