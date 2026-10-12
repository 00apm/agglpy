"""Tiny synthetic TIFFs carrying only the tag entries a reader uses.

The values are copied from real files (item 2.2 Findings); a whole
vendor tag is never copied, because real tags carry private paths,
user names and serial numbers (D-023).
"""

from pathlib import Path

import numpy as np
import tifffile

ZEISS_TAG = 34118  # CZ_SEM


def _pixels(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width), np.uint8)


def write_zeiss(
    path: Path,
    width: int,
    height: int,
    raw: float,
    image_pixel_size: tuple[float, str] | None = None,
    pages: int = 1,
    encoding: str = "cp1252",
) -> Path:
    """A TIFF with a Zeiss ``CZ_SEM`` tag.

    Args:
        path: File to write.
        width: Image width in px.
        height: Image height in px.
        raw: The unnamed value ``CZ_SEM[""][3]`` (m at 1024 px).
        image_pixel_size: ``(value, unit)`` of ``ap_image_pixel_size``,
            or None to leave the field out.
        pages: Number of pages; only the first carries the tag.
        encoding: Of the tag text. Zeiss writes cp1252 (µ as the byte
            0xB5); "utf-8" carries a Greek mu.
    """
    # unnamed values first (index 3 is the pixel size), then pairs of
    # an upper-case key line and a "Name = value unit" line
    text = f"0\r\n0\r\n0\r\n{raw!r}\r\n"
    if image_pixel_size is not None:
        value, unit = image_pixel_size
        text += f"AP_IMAGE_PIXEL_SIZE\r\nImage Pixel Size = {value} {unit}\r\n"
    data = text.encode(encoding)
    with tifffile.TiffWriter(path) as tif:
        tif.write(
            _pixels(width, height),
            extratags=[(ZEISS_TAG, 2, len(data) + 1, data, True)],
        )
        for _ in range(pages - 1):
            tif.write(_pixels(width, height))
    return path
