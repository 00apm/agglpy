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

import cv2
import numpy as np
import tifffile
from numpy.typing import NDArray

from agglpy.errors import FileFormatError, PixelSizeWarning

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
# versions (TiffFileError is a ValueError only in newer ones).
_TIFF_ERRORS = (tifffile.TiffFileError, ValueError, IndexError, struct.error)
_ZEISS_REFERENCE_WIDTH = 1024
_ZEISS_UNITS = {
    "nm": 1e-9,
    "\N{MICRO SIGN}m": 1e-6,
    "\N{GREEK SMALL LETTER MU}m": 1e-6,
    "mm": 1e-3,
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


# First hit wins (D-050). A new vendor comes with a real sample file.
_PIXEL_SIZE_READERS: tuple[
    Callable[[tifffile.TiffFile, int, Path], tuple[float, str] | None], ...
] = (_zeiss,)


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
