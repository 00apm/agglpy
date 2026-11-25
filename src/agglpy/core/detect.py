"""Particle detection with the Hough Circle Transform (OpenCV).

`detect_hct` finds circles on a preprocessed grey image and returns
them as a particle table in that image's frame. It knows nothing of the
analysed region or of preprocessing: the caller cuts the region, runs
`agglpy.core.preprocess.preprocess` and adds the region's offset to the
coordinates.
"""

import warnings
from collections.abc import Sequence

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

    Each `agglpy.params.HCTRange` is one ``cv2.HoughCircles`` call (see
    its docstring for how the fields map to OpenCV's arguments). A
    particle larger than a range is reported by it at the range's
    largest radius; when another range found it too, that circle is
    dropped. Other near-identical circles are kept and warned.

    Parameters
    ----------
    image : ndarray
        A 2-D ``uint8`` grey image, usually preprocessed.
    params : HCTParams
        The HCT ranges, run in the given order.

    Returns
    -------
    DataFrame
        A particle table ``id, x, y, r`` in px, in the frame of
        ``image``: ids ``1..n``, rows range by range, strongest circle
        first within a range. Values are OpenCV's float32 results as
        their shortest decimals (``r`` 29.4, not 29.399999618530273).
        An empty table when nothing is found.

    Raises
    ------
    ImageError
        If `image` is not a 2-D ``uint8`` array of at least 1 x 1 px.

    Warns
    -----
    NoParticlesWarning
        If no circle is found.
    DuplicateParticlesWarning
        If near-identical circles remain
        (`agglpy.tables.find_duplicates`), e.g. from overlapping ranges.

    Notes
    -----
    OpenCV finds only circles whose centre lies inside the image; a
    partly visible particle with its centre outside comes from manual
    correction.
    """
    image = check_grey_image(image)
    passes = [_hough(image, r) for r in params.ranges]
    return _merge_passes(image, passes, params.ranges)


def hct_edges(
    image: NDArray[np.uint8], hct_range: HCTRange
) -> NDArray[np.uint8]:
    """Return the edges ``cv2.HoughCircles`` builds for one HCT pass.

    A circle can only be found on these edges.

    Parameters
    ----------
    image : ndarray
        A 2-D ``uint8`` grey image, usually preprocessed.
    hct_range : HCTRange
        The pass; only ``canny_threshold`` is used.

    Returns
    -------
    ndarray
        A ``uint8`` image of the shape of `image`: 255 on an edge, 0
        elsewhere.

    Raises
    ------
    ImageError
        If `image` is not a 2-D ``uint8`` array of at least 1 x 1 px.

    Notes
    -----
    The same arithmetic as OpenCV's ``HOUGH_GRADIENT`` (``dp=1``): 3 x 3
    Sobel derivatives (``CV_16S``, ``BORDER_REPLICATE``), then Canny on
    them with ``high = round(canny_threshold)`` (OpenCV's ``cvRound``,
    half to even), ``low = max(1, high // 2)`` and the L1 gradient.
    """
    image = check_grey_image(image)
    dx, dy = (
        cv2.Sobel(
            image, cv2.CV_16S, *order, ksize=3, borderType=cv2.BORDER_REPLICATE
        )
        for order in ((1, 0), (0, 1))
    )
    high = round(hct_range.canny_threshold)
    edges = cv2.Canny(dx, dy, max(1, high // 2), high, L2gradient=False)
    return np.asarray(edges, dtype=np.uint8)


def _merge_passes(
    image: NDArray[np.uint8],
    passes: Sequence[NDArray[np.float32]],
    ranges: Sequence[HCTRange],
) -> pd.DataFrame:
    """Build the particle table of `detect_hct` from its passes.

    Rows pass by pass, in order, without the range-boundary duplicates
    (`_range_boundary_duplicates`).

    Also warns as ``detect_hct`` does, pointing at its caller.
    """
    circles = np.concatenate(passes)
    circles = circles[~_range_boundary_duplicates(passes, ranges)]
    if len(circles) == 0:
        height, width = image.shape
        warnings.warn(
            f"no circles found in the {width} x {height} px image (pixel "
            f"intensities {image.min()}-{image.max()}, of possible "
            "0-255); check preprocessing and HCT params",
            NoParticlesWarning,
            stacklevel=3,
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
            _duplicates_message(pairs), DuplicateParticlesWarning, stacklevel=3
        )
    return table


def _hough(image: NDArray[np.uint8], hct: HCTRange) -> NDArray[np.float32]:
    """Run one HoughCircles pass: (n, 3) array of x, y, r (float32)."""
    min_radius, max_radius = _radius_limits(hct)
    found = cv2.HoughCircles(
        image,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=hct.min_dist_ratio * hct.d_max / 2,
        param1=hct.canny_threshold,
        param2=hct.accumulator_threshold,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if found is None:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(found, dtype=np.float32).reshape(-1, 3)


def _radius_limits(hct: HCTRange) -> tuple[int, int]:
    """Return the ``minRadius, maxRadius`` of one HoughCircles pass."""
    # rounded outward, so an odd diameter stays inside the search
    return hct.d_min // 2, (hct.d_max + 1) // 2


def _range_boundary_duplicates(
    passes: Sequence[NDArray[np.float32]], ranges: Sequence[HCTRange]
) -> NDArray[np.bool_]:
    """Mark the circles a range clipped to its largest radius.

    A circle at its range's upper radius limit is marked when a larger
    circle of another pass has its centre within half its radius.

    Parameters
    ----------
    passes : sequence of ndarray
        One `_hough` result per range.
    ranges : sequence of HCTRange
        The range of each pass, in the same order.

    Returns
    -------
    ndarray
        True for each circle to drop, over the rows of
        ``np.concatenate(passes)``.
    """
    circles = np.concatenate(passes).astype(np.float64)
    pass_of = np.repeat(np.arange(len(passes)), [len(p) for p in passes])
    limits = np.array([_radius_limits(r)[1] for r in ranges], dtype=float)
    drop = np.zeros(len(circles), dtype=bool)
    # HCT reports radii in about 1 px steps: the top step holds the
    # particles larger than the range, their radius clipped
    at_limit = np.flatnonzero(circles[:, 2] > limits[pass_of] - 1)
    for i in at_limit:
        x, y, r = circles[i]
        # measured: one particle's circles lie <= 12 px apart, different
        # particles' >= 22.8 px (at r about 29)
        near = np.hypot(circles[:, 0] - x, circles[:, 1] - y) < r / 2
        drop[i] = bool(
            np.any(near & (circles[:, 2] > r) & (pass_of != pass_of[i]))
        )
    return drop
