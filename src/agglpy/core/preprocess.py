"""Image preprocessing before detection.

``preprocess`` runs the steps of ``PreprocessParams`` in a fixed order:
median blur, rolling-ball background subtraction, CLAHE. Noise is
removed first, since the later steps amplify it; the background is
subtracted while intensity is still signal plus background; CLAHE
evens out edge strength last, for the fixed thresholds of the edge
detector inside HCT. Each step is optional.

The input image is never modified; a new array is returned.
"""

from collections.abc import Callable
from functools import partial

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.restoration import rolling_ball

from agglpy.core._grey_image import check_grey_image
from agglpy.params import ClaheParams, PreprocessParams, RollingBallParams

_Step = Callable[[NDArray[np.uint8]], NDArray[np.uint8]]


def preprocess(
    image: NDArray[np.uint8], params: PreprocessParams
) -> NDArray[np.uint8]:
    """Preprocess a grey image for detection.

    Args:
        image: A 2-D ``uint8`` grey image, usually the analysed region
            cut from the raw image (``core.region.region_slices``).
        params: The steps to run; ``PreprocessParams()`` runs none.

    Returns:
        A new array of the same shape and dtype; with no step, an equal
        copy.

    Raises:
        ImageError: If ``image`` is not a 2-D ``uint8`` array of at
            least 1 x 1 px.
    """
    out = check_grey_image(image).copy()
    for _, step in _steps(params):
        out = step(out)
    return out


def _steps(params: PreprocessParams) -> list[tuple[str, _Step]]:
    """The steps to run, in order, by name (kept for diagnostics)."""
    steps: list[tuple[str, _Step]] = []
    if params.median_blur is not None:
        steps.append(
            ("median_blur", partial(_median_blur, kernel=params.median_blur))
        )
    if params.rolling_ball is not None:
        steps.append(
            (
                "rolling_ball",
                partial(_subtract_background, params=params.rolling_ball),
            )
        )
    if params.clahe is not None:
        steps.append(("clahe", partial(_clahe, params=params.clahe)))
    return steps


def _median_blur(image: NDArray[np.uint8], kernel: int) -> NDArray[np.uint8]:
    return np.asarray(cv2.medianBlur(image, kernel))


def _clahe(image: NDArray[np.uint8], params: ClaheParams) -> NDArray[np.uint8]:
    clahe = cv2.createCLAHE(
        clipLimit=params.clip_limit, tileGridSize=params.tile_grid
    )
    return np.asarray(clahe.apply(image))


def _subtract_background(
    image: NDArray[np.uint8], params: RollingBallParams
) -> NDArray[np.uint8]:
    """Rolling-ball background subtraction, as ImageJ does it.

    The background is estimated on a float copy (presmoothed if asked)
    and subtracted from the unsmoothed image; the result is rounded and
    clipped to 0-255 once, at the end.
    """
    values = image.astype(np.float64)
    if params.light_background:
        # The ball rolls under the inverted image, so it follows the
        # bright background; inverting back keeps the polarity.
        values = 255.0 - values
    estimate = cv2.blur(values, (3, 3)) if params.presmooth else values
    background = rolling_ball(estimate, radius=params.radius)
    result = values - background
    if params.light_background:
        result = 255.0 - result
    return np.asarray(np.clip(np.rint(result), 0, 255), dtype=np.uint8)
