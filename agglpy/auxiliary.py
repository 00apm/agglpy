import os
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import tifffile as tf  # type: ignore



def RGB_convert_to256(color: Tuple[float, ...]) -> Tuple[int, ...]:
    c256: List[int] = []
    for c in color:
        c256.append(int(c * 255))
    return tuple(c256)


def RGB_convert_to01(color: Tuple[int, ...]) -> Tuple[float, ...]:
    c01: List[float] = []
    for c in color:
        c01.append(c / 255)
    return tuple(c01)


def RGB_shader(
    color: Tuple[int] | npt.NDArray[np.int_],
    factor: float,
) -> npt.NDArray[np.int_]:
    if isinstance(color, tuple):
        wcol = np.full([1, 4], color)
    else:
        wcol = np.copy(color)
    wcol[:, :3] = wcol[:, :3] * (1 - factor)
    wcol[wcol > 255] = 255
    wcol[wcol < 0] = 0
    return wcol


def txt_is_default_or_none(txt: str) -> bool:
    c1: bool = txt is None
    c2: bool = txt_is_default(txt)
    c3: bool = txt_is_none(txt)
    c4: bool = txt_is_empty(txt)
    return c1 or c2 or c3 or c4

def txt_is_none_plus(txt: str) -> bool:
    c1: bool = txt is None
    c2: bool = txt_is_none(txt)
    c3: bool = txt_is_empty(txt)
    return c1 or c2 or c3

def txt_is_true(txt: str) -> bool:
    accepted_strings = {"true", "1", "t", "y", "yes", "yeah", "yup"}
    return txt in accepted_strings


def txt_is_default(txt: str) -> bool:
    accepted_strings = {"default", "auto", "normal"}
    return txt in accepted_strings


def txt_is_none(txt: str) -> bool:
    accepted_strings = {"none", "null", "nan"}
    return txt in accepted_strings


def txt_is_empty(txt: str) -> bool:
    return txt == ""


def txt_is_number(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False


# TODO: type hints
# typing module for matplotlib is not ready for colormaps used in this class
class nlcmap:
    def __init__(self, cmap, levels):
        self.name = cmap.name
        self.cmap = cmap
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype="float64")
        self._x = self.levels
        self.levmax = self.levels.max()
        self.transformed_levels = np.linspace(
            0.0, self.levmax, len(self.levels)
        )

    def __call__(self, xi, alpha=1.0, **kwargs):
        yi = np.interp(xi, self._x, self.transformed_levels)
        return self.cmap(yi / self.levmax, alpha)

def read_tiff_tags(file: os.PathLike) -> dict:
    """
    Method for reading .tif exif based metadata. Produces dictionary of
    exif tags from SEM tif file.

    Args:
        file (os.PathLike) : path to tiff image file

    Returns:
        dict: dictionary of exif tags from SEM tif file.
            SEM specific tags are included in CZ_SEM key (SEM_tags["CZ_SEM"])
    """

    with tf.TiffFile(file) as tif:
        tif_tags:dict = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
    return tif_tags



def get_floor(val: float, order: bool = True) -> float:
    """Get the floor value of a number based on order of magnitude.

    When `order` is True, the function returns the floor of the closest order 
    of magnitude.
    When `order` is False, the function rounds down to the closest value in the
      same order of magnitude.

    Args:
        val (float): The input value to compute the floor for.
        order (bool, optional): If True, computes based on order of magnitude. 
            If False, rounds down to the nearest number in the same magnitude. 
            Defaults to True.

    Raises:
        ValueError: If the input value is zero.

    Returns:
        float: The computed floor value based on the input and order.
    """
    if val == 0:
        raise ValueError("Value must be non-zero.")
    if order:
        # Compute the log10 of the absolute value to handle both positive 
        # and negative numbers
        log_val: float = np.log10(abs(val))
        log_floor: float = np.floor(log_val)
        return 10 ** log_floor
    else:
        # Get the order of magnitude (e.g., for 512 it will be 100)
        magnitude: float = 10 ** np.floor(np.log10(abs(val)))
        return np.floor(val / magnitude) * magnitude


def get_ceil(val: float, order: bool = True) -> float:
    """Get the ceiling value of a number based on order of magnitude

    When `order` is True, the function returns the ceiling of the closest order
    of magnitude.
    When `order` is False, the function rounds up to the closest value in the 
    same order of magnitude.

    Args:
        val (float): The input value to compute the ceiling for.
        order (bool, optional): If True, computes based on order of magnitude. 
            If False, rounds up to the nearest number in the same magnitude. 
            Defaults to True.

    Raises:
        ValueError: If the input value is zero.

    Returns:
        float: The computed ceiling value based on the input and order.
    """
    if val == 0:
        raise ValueError("Value must be non-zero.")
    if order:

        log_val: float = np.log10(abs(val))
        log_ceil: float = np.ceil(log_val)
        return 10 ** log_ceil
    else:
        magnitude: float = 10 ** np.floor(np.log10(abs(val)))
        return np.ceil(val / magnitude) * magnitude
