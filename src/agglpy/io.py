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

import os
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from agglpy.errors import FileFormatError

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
