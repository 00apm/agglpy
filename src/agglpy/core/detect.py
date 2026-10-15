"""Particle detection with the Hough Circle Transform (OpenCV).

``detect_hct`` finds circles on a preprocessed grey image and returns
them as a particle table in that image's frame. It knows nothing of the
analysed region or of preprocessing: the caller cuts the region, runs
``preprocess`` and adds the region's offset to the coordinates.
"""

import warnings

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.core._grey_image import check_grey_image
from agglpy.errors import DuplicateParticlesWarning, NoParticlesWarning
from agglpy.params import HCTParams, HCTRange
from agglpy.tables import (
    _duplicates_message,
    find_duplicates,
    validate_particles,
)


def detect_hct(image: NDArray[np.uint8], params: HCTParams) -> pd.DataFrame:
    """Detect circular particles, one HCT pass per range.

    Each ``HCTRange`` is one ``cv2.HoughCircles`` call (see its
    docstring for how the fields map to OpenCV's arguments). The passes
    are independent: a particle found by two ranges, typically at the
    boundary between them, appears twice and is reported with a
    ``DuplicateParticlesWarning``; nothing is removed.

    OpenCV finds only circles whose centre lies inside the image; a
    partly visible particle with its centre outside comes from manual
    correction.

    Args:
        image: A 2-D ``uint8`` grey image, usually preprocessed.
        params: The HCT ranges, run in the given order.

    Returns:
        A particle table ``id, x, y, r`` in px, in the frame of
        ``image``: ids ``1..n``, rows range by range, strongest circle
        first within a range. Values are OpenCV's float32 results as
        their shortest decimals (``r`` 29.4, not 29.399999618530273).
        An empty table when nothing is found, with one
        ``NoParticlesWarning``.

    Raises:
        ImageError: If ``image`` is not a 2-D ``uint8`` array of at
            least 1 x 1 px.
    """
    image = check_grey_image(image)
    circles = np.concatenate([_hough(image, r) for r in params.ranges])
    if len(circles) == 0:
        height, width = image.shape
        warnings.warn(
            f"no circles found in the {width} x {height} px image (pixel "
            f"intensities {image.min()}-{image.max()}, of possible "
            "0-255); check preprocessing and HCT params",
            NoParticlesWarning,
            stacklevel=2,
        )
    # float32 -> its shortest decimal -> float64: 29.4, not the float64
    # neighbour of the float32 value (29.399999618530273)
    values = circles.astype(str).astype(np.float64)
    table = validate_particles(
        pd.DataFrame(
            {
                "id": np.arange(1, len(values) + 1),
                "x": values[:, 0],
                "y": values[:, 1],
                "r": values[:, 2],
            }
        ),
        warn_duplicates=False,
    )
    # warned here, not in validate_particles, so the warning points at
    # the caller's line
    pairs = find_duplicates(table)
    if pairs:
        warnings.warn(
            _duplicates_message(pairs), DuplicateParticlesWarning, stacklevel=2
        )
    return table


def _hough(image: NDArray[np.uint8], hct: HCTRange) -> NDArray[np.float32]:
    """One HoughCircles pass: an (n, 3) array of x, y, r (float32)."""
    found = cv2.HoughCircles(
        image,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=hct.min_dist_ratio * hct.d_max / 2,
        param1=hct.canny_threshold,
        param2=hct.accumulator_threshold,
        minRadius=int(hct.d_min / 2),
        maxRadius=int(hct.d_max / 2),
    )
    if found is None:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(found, dtype=np.float32).reshape(-1, 3)
