"""The analysed region of an image, in px.

The region is needed in two forms half a pixel apart (pixel centres at
integers, see ``agglpy.tables``): array indices to cut the image, and
pixel-centre coordinates for ``touches_edge``; its area is the field of
view. All three come from one ``Bounds``, so they can't disagree.
"""

from dataclasses import dataclass

from agglpy.errors import ParamsError
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

    The rows kept are ``int((1 - bottom_fraction) * height)``: 2304
    rows and 0.1 keep 2073. Width and height are plain numbers, so the
    region of an image is known from its metadata without its pixels.

    Args:
        region: The region params.
        width: Image width in px.
        height: Image height in px.

    Returns:
        The analysed rectangle, e.g. ``Bounds(-0.5, -0.5, 3071.5,
        2072.5)`` for a 3072 x 2304 image and ``bottom_fraction=0.1``.

    Raises:
        ParamsError: If ``width`` or ``height`` is not an integer >= 1,
            or the cut leaves no row.
    """
    for name, value in (("width", width), ("height", height)):
        if _integer(name, value) < 1:
            raise ParamsError(f"{name} must be >= 1, got {value!r}")
    rows = int((1 - region.bottom_fraction) * height)
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
        ``(rows, cols)``, e.g. ``(slice(0, 2073), slice(0, 3072))``.
    """
    rows = slice(round(bounds.y_min + 0.5), round(bounds.y_max + 0.5))
    cols = slice(round(bounds.x_min + 0.5), round(bounds.x_max + 0.5))
    return rows, cols
