"""The analysed region of an image, in px.

The region is needed in two forms half a pixel apart (pixel centres at
integers, see ``agglpy.tables``): array indices to cut the image, and
pixel-centre coordinates for ``touches_edge``; its area is the field of
view. All three come from one ``Bounds``, so they can't disagree.
``outside_region`` marks the circles that lie entirely outside it.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from agglpy.errors import ParamsError, ParticleTableError
from agglpy.params import RegionParams, _integer


@dataclass(frozen=True)
class Bounds:
    """The analysed rectangle in px, pixel-centre convention.

    Pixel ``i`` covers ``[i - 0.5, i + 0.5)``, so a region of whole
    pixels has its edges at ``k - 0.5``.

    Attributes:
        x_min: Left edge.
        y_min: Top edge.
        x_max: Right edge.
        y_max: Bottom edge.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def area(self) -> float:
        """Area in px², the field of view of the analysed region."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


def region_bounds(region: RegionParams, width: int, height: int) -> Bounds:
    """Turn the region params into a rectangle on an image of this size.

    The rows kept are ``(1 - bottom_fraction) * height`` rounded to the
    nearest whole row, a half up: 2304 rows and 0.1 keep 2074 (2073.6).
    Width and height are plain numbers, so the region of an image is
    known from its metadata without its pixels.

    Args:
        region: The region params.
        width: Image width in px.
        height: Image height in px.

    Returns:
        The analysed rectangle, e.g. ``Bounds(-0.5, -0.5, 3071.5,
        2073.5)`` for a 3072 x 2304 image and ``bottom_fraction=0.1``.

    Raises:
        ParamsError: If ``width`` or ``height`` is not an integer >= 1,
            or the cut leaves no row.
    """
    for name, value in (("width", width), ("height", height)):
        if _integer(name, value) < 1:
            raise ParamsError(f"{name} must be >= 1, got {value!r}")
    # exact decimals: in floats (1 - 0.8) * 10 is 1.999... and
    # (1 - 0.34) * 25 is 16.499..., which round to the wrong row
    kept = (1 - Fraction(str(region.bottom_fraction))) * int(height)
    rows = math.floor(kept + Fraction(1, 2))
    if rows < 1:
        raise ParamsError(
            f"bottom_fraction={region.bottom_fraction} leaves {rows} of "
            f"{height} rows"
        )
    return Bounds(-0.5, -0.5, int(width) - 0.5, rows - 0.5)


def region_slices(bounds: Bounds) -> tuple[slice, slice]:
    """Array slices (rows, columns) that cut the region from the image.

    ``image[rows, cols]`` is a view, not a copy. Coordinates found on
    the cut lie in its own frame; add ``(bounds.x_min + 0.5,
    bounds.y_min + 0.5)`` to bring them back to the image, which is
    (0, 0) for a bottom cut.

    Args:
        bounds: A rectangle from ``region_bounds``.

    Returns:
        ``(rows, cols)``, e.g. ``(slice(0, 2074), slice(0, 3072))``.
    """
    rows = slice(round(bounds.y_min + 0.5), round(bounds.y_max + 0.5))
    cols = slice(round(bounds.x_min + 0.5), round(bounds.x_max + 0.5))
    return rows, cols


def outside_region(
    particles: pd.DataFrame, bounds: Bounds
) -> "pd.Series[bool]":
    """Mark the circles that don't overlap the analysed region at all.

    A circle is outside when the distance from its centre to the
    ``bounds`` rectangle is larger than ``r``. A circle that only
    touches the rectangle, is cut by an edge or has its centre outside
    but still reaches in is not outside. The region lies inside the
    image, so a circle outside the image is outside the region too.

    Args:
        particles: A validated particle table (columns ``x, y, r`` in
            px); a NaN coordinate, which validation rejects, would
            count as not outside.
        bounds: The analysed rectangle, from ``region_bounds``.

    Returns:
        A boolean mask on ``particles.index``: ``particles[~mask]``
        keeps the circles that overlap the region.

    Raises:
        ParticleTableError: If ``x``, ``y`` or ``r`` is missing.
    """
    missing = [c for c in ("x", "y", "r") if c not in particles.columns]
    if missing:
        raise ParticleTableError(f"missing columns: {missing}")
    x, y, r = (particles[c].to_numpy(dtype=float) for c in ("x", "y", "r"))
    # distance from the centre to the nearest point of the rectangle
    dx = np.maximum(np.maximum(bounds.x_min - x, x - bounds.x_max), 0.0)
    dy = np.maximum(np.maximum(bounds.y_min - y, y - bounds.y_max), 0.0)
    return pd.Series(np.hypot(dx, dy) > r, index=particles.index)
