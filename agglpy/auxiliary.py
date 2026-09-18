import os
from pathlib import Path
from typing import Any, List, Tuple, Union

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
    def __init__(self, cmap, levels: np.ndarray) -> None:  # type: ignore
        self.name = cmap.name
        self.cmap = cmap
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype="float64")
        self._x = self.levels
        self.levmax = self.levels.max()
        self.transformed_levels = np.linspace(0.0, self.levmax, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kwargs):  # type: ignore
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
        tif_tags: dict = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
    return tif_tags


def get_floor(
    val: Union[float, int, npt.ArrayLike],
    order: bool = True,
) -> Union[float, int, npt.NDArray[np.float64]]:
    """Get the floor value of a number or array based on order of magnitude.

    When `order` is True, the function returns the floor of the closest order
    of magnitude.
    When `order` is False, the function rounds down to the closest value in the
    same order of magnitude.

    Args:
        val (Union[float, int, ArrayLike]): Input value(s) to compute the floor for.
            Can be a single number (float or int) or an array-like structure
            (list, tuple, np.ndarray).

        order (bool, optional): If True, computes based on order of magnitude.
            If False, rounds down to the nearest number in the same magnitude.
            Defaults to True.

    Raises:
        ValueError: If any input value is zero.

    Returns:
        Union[float, int, NDArray[np.float64]]: The computed floor value(s).
            Returns a float for float inputs, an int for int inputs,
            and a NumPy array for array-like inputs.
    """
    # Convert input to a NumPy array
    val_array = np.asarray(val)

    if np.any(val_array == 0):
        raise ValueError("All values must be non-zero.")

    if order:
        log_val = np.log10(np.abs(val_array))
        log_floor = np.floor(log_val)
        result = 10**log_floor
    else:
        magnitude = 10 ** np.floor(np.log10(np.abs(val_array)))
        result = np.floor(val_array / magnitude) * magnitude

    # If the input was scalar, return a scalar of the appropriate type
    if np.isscalar(val):
        if isinstance(val, int):
            return int(result)  # Convert to int
        return float(result)  # Convert to float

    # Otherwise, return the result as a NumPy array
    return np.asarray(result, dtype=np.float64)


def get_ceil(
    val: Union[float, int, npt.ArrayLike],
    order: bool = True,
) -> Union[float, int, npt.NDArray[np.float64]]:
    """Get the ceiling value of a number or array based on order of magnitude.

    When `order` is True, the function returns the ceiling of the closest order
    of magnitude.
    When `order` is False, the function rounds up to the closest value in the
    same order of magnitude.

    Args:
        val (Union[float, int, ArrayLike]): Input value(s) to compute the ceiling for.
            Can be a single number (float or int) or an array-like structure
            (list, tuple, np.ndarray).

        order (bool, optional): If True, computes based on order of magnitude.
            If False, rounds up to the nearest number in the same magnitude.
            Defaults to True.

    Raises:
        ValueError: If any input value is zero.

    Returns:
        Union[float, int, NDArray[np.float64]]: The computed ceiling value(s).
            Returns a float for float inputs, an int for int inputs,
            and a NumPy array for array-like inputs.
    """
    # Convert input to a NumPy array
    val_array = np.asarray(val)

    if np.any(val_array == 0):
        raise ValueError("All values must be non-zero.")

    if order:
        log_val = np.log10(np.abs(val_array))
        log_ceil = np.ceil(log_val)
        result = 10**log_ceil
    else:
        magnitude = 10 ** np.floor(np.log10(np.abs(val_array)))
        result = np.ceil(val_array / magnitude) * magnitude

    # If the input was scalar, return a scalar of the appropriate type
    if np.isscalar(val):
        if isinstance(val, int):
            return int(result)  # Convert to int
        return float(result)  # Convert to float

    # Otherwise, return the result as a NumPy array
    return np.asarray(result, dtype=np.float64)
