"""Reading and writing the files agglpy works with (the io edge).

Images with their pixel size, and particle files in and out. Every
quirk of a file format is absorbed here (palette TIFFs, vendor tags,
ImageJ's pixel edges and ids, byte order marks, float parsing); below
this module the core sees only arrays and particle tables in px.

io knows file formats, not the collection layout: every function takes
a path and reads or writes that one file. Names like
``particles_corrected.csv`` belong to the application.

Import the names (``from agglpy.io import read_image``), not the
module: ``from agglpy import io`` would hide the standard library's
``io`` in your script.
"""

import math
import os
import struct
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import tifffile
from numpy.typing import NDArray

from agglpy.errors import (
    DuplicateParticlesWarning,
    FileFormatError,
    NonCircularParticlesWarning,
    ParamsError,
    ParticleTableError,
    PixelSizeWarning,
)
from agglpy.tables import (
    PARTICLE_COLUMNS,
    _duplicates_message,
    find_duplicates,
    validate_particles,
)

PathLike = str | os.PathLike[str]

# ---------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------


def read_image(path: PathLike) -> NDArray[np.uint8]:
    """Read an image as a 2-D grey ``uint8`` array.

    Any format OpenCV decodes (TIFF, PNG, JPEG, BMP). Palette and
    colour images become grey with the BT.601 weights, 16-bit images
    keep their top byte, an alpha channel is dropped and a multipage
    TIFF gives its first page: the same array as
    ``cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)``, also for
    paths with characters outside the system code page. The file is
    opened read-only.

    Args:
        path: The image file.

    Returns:
        The grey image, shape ``(height, width)``.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FileFormatError: If OpenCV can't decode the file.
    """
    path = Path(path)
    # imdecode on the bytes: cv2.imread fails on Windows for paths
    # outside the system code page
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise FileFormatError(f"{path}: not an image OpenCV can decode")
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


@dataclass(frozen=True)
class ImageMetadata:
    """What ``read_image_metadata`` found in an image file.

    Attributes:
        pixel_size: Length of one pixel in metres, or ``None`` when no
            reader finds it (a normal value, not an error).
        pixel_size_source: Where ``pixel_size`` came from, for the
            record of an analysis, or ``None``.
        width: Width of the whole file in px (a data bar included).
        height: Height of the whole file in px.
    """

    pixel_size: float | None
    pixel_size_source: str | None
    width: int
    height: int


def read_image_metadata(path: PathLike) -> ImageMetadata:
    """Read an image's size and, from its tags, the pixel size.

    TIFF files are read from their headers only (first page of a
    multipage file). The pixel size comes from the first of these that
    the file carries: Zeiss ``CZ_SEM``, Thermo/FEI ``[Scan]
    PixelWidth``, an ImageJ calibration. The plain TIFF resolution is
    never used on its own. Other formats are decoded once for their
    size and have no pixel size.

    Args:
        path: The image file.

    Returns:
        The size in px and the pixel size in metres (or ``None``).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FileFormatError: If the TIFF header can't be read, or another
            format can't be decoded.

    Warns:
        PixelSizeWarning: If the Zeiss text field and raw value differ
            by more than 1 %, or the pixels are not square.
    """
    path = Path(path)
    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic not in _TIFF_MAGIC:
        height, width = read_image(path).shape
        return ImageMetadata(None, None, width, height)
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages.first
            width, height = int(page.imagewidth), int(page.imagelength)
            if width < 1 or height < 1:
                raise ValueError(f"image size {width} x {height} px")
            found = None
            for reader in _PIXEL_SIZE_READERS:
                found = reader(tif, width, path)
                if found is not None:
                    break
    except _TIFF_ERRORS as exc:
        raise FileFormatError(
            f"{path}: unreadable TIFF header: {exc}"
        ) from exc
    size, source = found if found is not None else (None, None)
    return ImageMetadata(size, source, width, height)


_TIFF_MAGIC = (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+")  # classic, BigTIFF
# What tifffile raises on a broken header, across the supported
# versions (TiffFileError is a ValueError only in newer ones); a tag
# of the wrong type gives TypeError, KeyError or OverflowError. Not
# OSError: a missing or locked file stays a built-in error.
_TIFF_ERRORS = (
    tifffile.TiffFileError,
    ValueError,
    IndexError,
    struct.error,
    TypeError,
    KeyError,
    OverflowError,
)
_ZEISS_REFERENCE_WIDTH = 1024
_ZEISS_UNITS = {
    "nm": 1e-9,
    "\N{MICRO SIGN}m": 1e-6,
    "\N{GREEK SMALL LETTER MU}m": 1e-6,
    "mm": 1e-3,
    "m": 1.0,
}
_IMAGEJ_UNITS = {
    "micron": 1e-6,
    "\N{MICRO SIGN}m": 1e-6,
    "\N{GREEK SMALL LETTER MU}m": 1e-6,
    "um": 1e-6,
    "nm": 1e-9,
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
}
_TOLERANCE = 0.01  # 1 %: text vs raw value, x vs y pixel size


def _zeiss(
    tif: tifffile.TiffFile, width: int, path: Path
) -> tuple[float, str] | None:
    sem = tif.sem_metadata
    if not isinstance(sem, dict):
        return None
    # The raw value is the pixel size at a 1024-px reference width,
    # whatever the stored width (item 2.2 Findings, D-050).
    raw = _positive(_item(sem.get(""), 3))
    if raw is None:
        return None
    size = raw * _ZEISS_REFERENCE_WIDTH / width
    text = sem.get("ap_image_pixel_size")
    value, unit = _positive(_item(text, 1)), _item(text, 2)
    if value is not None and unit in _ZEISS_UNITS:
        stated = value * _ZEISS_UNITS[unit]
        if abs(stated - size) > _TOLERANCE * size:
            warnings.warn(
                f"{path}: Zeiss pixel size {size:.6g} m (raw value "
                f"scaled to {width} px) and 'Image Pixel Size' "
                f"{value} {unit} differ by more than 1 %; the raw value "
                f"is used",
                PixelSizeWarning,
                stacklevel=3,
            )
    return size, f"Zeiss CZ_SEM (1024-px reference scaled to {width} px)"


def _thermo_fei(
    tif: tifffile.TiffFile, width: int, path: Path
) -> tuple[float, str] | None:
    fei = tif.fei_metadata
    scan = fei.get("Scan") if isinstance(fei, dict) else None
    if not isinstance(scan, dict):
        return None
    x = _positive(scan.get("PixelWidth"))
    if x is None:
        return None
    _check_square(x, _positive(scan.get("PixelHeight")), path)
    return x, "Thermo/FEI [Scan] PixelWidth"


def _imagej_calibration(
    tif: tifffile.TiffFile, width: int, path: Path
) -> tuple[float, str] | None:
    ij = tif.imagej_metadata
    unit = ij.get("unit") if isinstance(ij, dict) else None
    factor = _IMAGEJ_UNITS.get(unit) if isinstance(unit, str) else None
    tags = tif.pages.first.tags
    x_res = _resolution(tags.get(282))  # XResolution, px per unit
    if factor is None or x_res is None:
        return None
    y_res = _resolution(tags.get(283))  # YResolution
    _check_square(factor / x_res, y_res and factor / y_res, path)
    return factor / x_res, f"ImageJ calibration (unit={unit})"


# First hit wins (D-050). A new vendor comes with a real sample file.
_PIXEL_SIZE_READERS: tuple[
    Callable[[tifffile.TiffFile, int, Path], tuple[float, str] | None], ...
] = (_zeiss, _thermo_fei, _imagej_calibration)


def _item(values: object, index: int) -> object:
    if isinstance(values, tuple) and len(values) > index:
        return values[index]
    return None


def _positive(value: object) -> float | None:
    """``value`` as a finite float > 0, else None (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _resolution(tag: Any) -> float | None:
    value = getattr(tag, "value", None)
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        return _positive(value[0] / value[1])
    return None


def _check_square(x: float, y: float | None, path: Path) -> None:
    if y is not None and abs(x - y) > _TOLERANCE * x:
        warnings.warn(
            f"{path}: pixels are not square (x {x:.6g} m, y {y:.6g} m); "
            f"the x value is used",
            PixelSizeWarning,
            stacklevel=4,
        )


# ---------------------------------------------------------------------
# Particle files
# ---------------------------------------------------------------------

# Header columns that recognise each format (all present, case-sensitive).
_FORMATS: dict[str, tuple[str, ...]] = {
    "agglpy": ("id", "x", "y", "r"),
    "imagej": ("Index", "Type", "X", "Y", "Width", "Height"),
    # Internal: files of earlier agglpy versions, kept so the author's
    # old data still reads. Found by auto-detection and accepted by
    # format=, but not documented: v0.4 HCT_multi output (0-based ids)
    # and the format before it, whose ids restart at 0 for each HCT
    # range, so its rows are numbered 0..n-1 instead (author,
    # 2026-10-12).
    "agglpy_v04": ("ID", "X", "Y", "R"),
    "agglpy_v00": ("ID", "X (pixels)", "Y (pixels)", "Radius (pixels)"),
}
_PUBLIC_FORMATS = ("agglpy", "imagej")
_RENAMES = {
    "agglpy_v04": {"ID": "id", "X": "x", "Y": "y", "R": "r"},
    "agglpy_v00": {
        "ID": "id",
        "X (pixels)": "x",
        "Y (pixels)": "y",
        "Radius (pixels)": "r",
    },
}
_MAX_LISTED = 10
# An ImageJ oval is read as a circle up to this |Width - Height| (px):
# the grain of ImageJ's integer bounding boxes.
_NEAR_CIRCLE_PX = 1.0


def read_particles(path: PathLike, *, format: str = "auto") -> pd.DataFrame:
    """Read a particle file into a particle table (px).

    Formats, recognised by their header columns:

    - ``"agglpy"``: ``id, x, y, r``, written by ``write_particles``;
      read unchanged, ids included.
    - ``"imagej"``: the CSV of Fiji's ROI Manager (``Index, Type, X,
      Y, Width, Height``). Converted to pixel centres
      (``x = X + Width/2 - 0.5``), ``r = (Width + Height)/4`` and
      ``id = Index + 1``, the number Fiji shows. Ovals with
      ``|Width - Height| <= 1`` px are read as circles; other shapes
      are dropped. Either gives one ``NonCircularParticlesWarning``.

    Only the geometry is read: the result has exactly the columns
    ``id, x, y, r``, and any other column in the file is ignored. A
    file with a header and no rows gives an empty table.

    Args:
        path: The particle file (UTF-8, a byte order mark is fine).
        format: ``"auto"`` (default) recognises the format from the
            header; a format name skips that and requires its columns.

    Returns:
        A validated particle table.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FileFormatError: If no format or more than one matches, a
            column is missing or a value breaks the particle table
            schema.
        ParamsError: If ``format`` is not a known format name.

    Warns:
        NonCircularParticlesWarning: If ImageJ shapes were approximated
            or dropped.
        DuplicateParticlesWarning: If the file holds near-identical
            circles (see ``agglpy.tables.find_duplicates``).
    """
    path = Path(path)
    if format != "auto" and format not in _FORMATS:
        raise ParamsError(
            f"format must be 'auto' or one of {list(_PUBLIC_FORMATS)}, "
            f"got {format!r}"
        )
    try:
        raw = pd.read_csv(
            path, encoding="utf-8-sig", float_precision="round_trip"
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise FileFormatError(f"{path}: not a CSV table: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise FileFormatError(f"{path}: not UTF-8 text: {exc}") from exc
    # rows one field longer than the header: pandas takes the first
    # field as the index and shifts every value one column left
    if not isinstance(raw.index, pd.RangeIndex):
        raise FileFormatError(
            f"{path}: rows have more fields than the header "
            f"{list(raw.columns)} (a decimal comma or a trailing comma?)"
        )
    name = _detect(raw, path) if format == "auto" else format
    missing = [c for c in _FORMATS[name] if c not in raw.columns]
    if missing:
        raise FileFormatError(
            f"{path}: columns {missing} of format {name!r} are missing"
        )
    try:
        if name == "imagej":
            table = _from_imagej(raw, path)
        else:
            table = raw.loc[:, list(_FORMATS[name])]
            table = table.rename(columns=_RENAMES.get(name, {}))
            if name == "agglpy_v00":
                table["id"] = np.arange(len(table))
        table = validate_particles(table, warn_duplicates=False)
    except ParticleTableError as exc:
        raise FileFormatError(f"{path}: {exc}") from exc
    # warned here, not in validate_particles: the message names the file
    # and points at the caller's line
    pairs = find_duplicates(table)
    if pairs:
        warnings.warn(
            f"{path}: {_duplicates_message(pairs)}",
            DuplicateParticlesWarning,
            stacklevel=2,
        )
    return table


def _detect(raw: pd.DataFrame, path: Path) -> str:
    found = [
        name
        for name, columns in _FORMATS.items()
        if all(c in raw.columns for c in columns)
    ]
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileFormatError(
            f"{path}: not a particle file: header {list(raw.columns)} "
            f"matches none of the formats {list(_PUBLIC_FORMATS)}"
        )
    raise FileFormatError(
        f"{path}: the header matches several formats {found}; "
        f"choose one with format="
    )


def _from_imagej(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    numbers = {
        c: _numeric(raw[c], c) for c in ("Index", "X", "Y", "Width", "Height")
    }
    # a ROI without an index or a position can't be placed: fail,
    # naming the file's column and line (an empty size only drops it)
    for column in ("Index", "X", "Y"):
        rows = np.flatnonzero(~np.isfinite(numbers[column].to_numpy()))
        if rows.size:
            lines = [str(row + 2) for row in rows]  # line 1: header
            word = "line" if len(lines) == 1 else "lines"
            raise ParticleTableError(
                f"column {column!r} is empty or not finite at {word} "
                f"{_listed(lines)}"
            )
    w, h = numbers["Width"], numbers["Height"]
    ids = numbers["Index"] + 1  # the number Fiji shows
    oval = raw["Type"] == "Oval"
    gap = (w - h).abs()
    near = oval & (gap > 0) & (gap <= _NEAR_CIRCLE_PX)
    keep = oval & (gap <= _NEAR_CIRCLE_PX)
    if near.any() or not keep.all():
        _warn_shapes(raw, ids, near, ~keep, path)
    # ImageJ counts from pixel edges, agglpy from pixel centres (D-051)
    return pd.DataFrame(
        {
            "id": ids[keep],
            "x": numbers["X"][keep] + w[keep] / 2 - 0.5,
            "y": numbers["Y"][keep] + h[keep] / 2 - 0.5,
            "r": (w[keep] + h[keep]) / 4,
        }
    )


def _numeric(values: pd.Series, column: str) -> pd.Series:
    try:
        return pd.to_numeric(values, errors="raise").astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise ParticleTableError(
            f"column {column!r} must be numeric: {exc}"
        ) from exc


def _warn_shapes(
    raw: pd.DataFrame,
    ids: pd.Series,
    near: pd.Series,
    dropped: pd.Series,
    path: Path,
) -> None:
    def size(i: Any) -> str:
        return f"{raw.at[i, 'Width']:g} x {raw.at[i, 'Height']:g}"

    parts = [f"{path}: {len(raw)} ROIs"]
    if near.any():
        labels = [f"{int(ids[i])}" for i in raw.index[near]]
        parts.append(
            f"{len(labels)} near-circular ovals read as circles "
            f"(|W - H| <= 1 px; ids {_listed(labels)})"
        )
    if dropped.any():
        labels = [
            f"{int(ids[i])}: {raw.at[i, 'Type']} {size(i)}"
            for i in raw.index[dropped]
        ]
        parts.append(
            f"{len(labels)} non-circular ROIs dropped (ids {_listed(labels)})"
        )
    warnings.warn(
        "; ".join(parts) + ".", NonCircularParticlesWarning, stacklevel=4
    )


def _listed(labels: list[str]) -> str:
    """Labels joined by commas, cut after ``_MAX_LISTED``."""
    more = len(labels) - _MAX_LISTED
    tail = f" and {more} more" if more > 0 else ""
    return ", ".join(labels[:_MAX_LISTED]) + tail


def write_particles(
    particles: pd.DataFrame, path: PathLike, *, overwrite: bool = False
) -> None:
    """Write a particle table as an agglpy particle file.

    Writes exactly the columns ``id, x, y, r`` (px), in that order:
    extra columns in memory are not written. One header row, comma
    separated, ``.`` as decimal mark, UTF-8 without a byte order mark,
    LF line endings on every OS, so ImageJ can open the file and the
    same table always gives the same bytes. Floats are written as the
    shortest decimal that reads back to the same value:
    ``read_particles`` returns the table exactly.

    Args:
        particles: A particle table.
        path: The file to write.
        overwrite: Replace an existing file. By default an existing
            file is never touched.

    Raises:
        ParticleTableError: If the table breaks the schema (nothing is
            written).
        FileExistsError: If the file exists and ``overwrite`` is False.
        FileNotFoundError: If the folder doesn't exist.
    """
    table = validate_particles(particles, warn_duplicates=False)
    table = table.loc[:, list(PARTICLE_COLUMNS)]
    with Path(path).open(
        "w" if overwrite else "x", encoding="utf-8", newline=""
    ) as fh:
        table.to_csv(fh, index=False, lineterminator="\n")
