"""The image input check shared by preprocessing and detection."""

import numpy as np
from numpy.typing import NDArray

from agglpy.errors import ImageError


def check_grey_image(image: object) -> NDArray[np.uint8]:
    """Return ``image`` if it is a 2-D ``uint8`` array of >= 1 x 1 px.

    Nothing is converted: scaling a float or 16-bit image is a choice
    that changes what is detected, so it is left to the caller.

    Raises:
        ImageError: For anything else, naming what was given.
    """
    if not isinstance(image, np.ndarray):
        got = type(image).__name__
    elif image.dtype != np.uint8 or image.ndim != 2 or image.size == 0:
        got = f"{image.dtype} with shape {image.shape}"
    else:
        return image
    raise ImageError(
        "expected a 2-D uint8 grey image (as from agglpy.io.read_image), "
        f"got {got}"
    )
