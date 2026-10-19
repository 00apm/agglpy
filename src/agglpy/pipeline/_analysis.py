"""Analysis of one image's particle table: ``analyze``, ``ImageResult``."""

import warnings
from dataclasses import dataclass

import pandas as pd

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.properties import agglomerate_properties, particle_properties
from agglpy.core.region import Bounds, outside_region, region_bounds
from agglpy.errors import ParamsError, ParticlesOutsideRegionWarning
from agglpy.params import RegionParams
from agglpy.tables import validate_particles

_MAX_LISTED = 10


@dataclass(frozen=True, eq=False)
class ImageResult:
    """What ``analyze`` found on one image, in px.

    Plain data: no reference to the image, so it pickles and can be
    sent to another process. No equality: compare the tables
    themselves.

    Attributes:
        particles: The kept particles: ``id, x, y, r``, then
            ``agglomerate_id``, ``enclosed``, ``D``, ``area``,
            ``volume``, ``surface`` and, with an image size,
            ``touches_edge``.
        agglomerates: One row per agglomerate
            (``core.properties.agglomerate_properties``), with
            ``touches_edge`` when the image size was given.
        outside_ids: Ids of the particles left out as lying entirely
            outside the analysed region, ascending; empty without an
            image size.
        width: Image width in px, or None.
        height: Image height in px, or None.
        region: The region params used.
        bounds: The analysed rectangle, or None without an image size.
        fov_area: Area of the analysed region (field of view) in px²,
            or None without an image size.
    """

    particles: pd.DataFrame
    agglomerates: pd.DataFrame
    outside_ids: tuple[int, ...]
    width: int | None
    height: int | None
    region: RegionParams
    bounds: Bounds | None
    fov_area: float | None


def analyze(
    particles: pd.DataFrame,
    *,
    width: int | None = None,
    height: int | None = None,
    region: RegionParams = RegionParams(),
) -> ImageResult:
    """Group one image's particles into agglomerates and measure both.

    The table may come from detection or from a corrected file; the
    result is the same kind either way. With the image size (from the
    image's metadata, no pixels needed), the analysed region is known:
    circles lying entirely outside it are left out (one warning), and
    particles and agglomerates get ``touches_edge``. Without it,
    nothing is left out or flagged.

    Args:
        particles: A particle table in px; not modified.
        width: Image width in px; give both sizes or neither.
        height: Image height in px.
        region: The analysed region; needs the image size unless it is
            the whole image.

    Returns:
        The tables and the region, in px.

    Raises:
        ParamsError: If only one size is given, a size is not an
            integer >= 1, or ``region`` cuts the image without a size.
        ParticleTableError: If the table breaks the schema.

    Warns:
        ParticlesOutsideRegionWarning: If circles were left out.
        DuplicateParticlesWarning: About duplicate enclosed particles
            (see ``core.agglomerates.find_agglomerates``).
    """
    if (width is None) != (height is None):
        raise ParamsError("give both width and height, or neither")
    bounds = None
    if width is None or height is None:
        if region != RegionParams():
            raise ParamsError(
                "region needs the image size: give width and height"
            )
    else:
        bounds = region_bounds(region, width, height)
    # Whoever made the table (reader or detector) already warned about
    # duplicates.
    table = validate_particles(particles, warn_duplicates=False)
    outside_ids: tuple[int, ...] = ()
    if bounds is None:
        # a touches_edge from an earlier run with a size is stale
        table = table.drop(columns="touches_edge", errors="ignore")
    else:
        outside = outside_region(table, bounds)
        outside_ids = tuple(sorted(int(i) for i in table["id"][outside]))
        if outside_ids:
            _warn_outside(outside_ids, len(table), bounds)
        table = table[~outside]
    found = find_agglomerates(table)
    return ImageResult(
        particles=particle_properties(found, bounds=bounds),
        agglomerates=agglomerate_properties(found, bounds=bounds),
        outside_ids=outside_ids,
        width=None if width is None else int(width),
        height=None if height is None else int(height),
        region=region,
        bounds=bounds,
        fov_area=None if bounds is None else float(bounds.area),
    )


def _warn_outside(ids: tuple[int, ...], total: int, bounds: Bounds) -> None:
    listed = ", ".join(map(str, ids[:_MAX_LISTED]))
    more = len(ids) - _MAX_LISTED
    suffix = f" and {more} more" if more > 0 else ""
    warnings.warn(
        f"{len(ids)} of {total} particles lie entirely outside the "
        f"analysed region (x {bounds.x_min:g} to {bounds.x_max:g}, "
        f"y {bounds.y_min:g} to {bounds.y_max:g} px) and were left out: "
        f"ids {listed}{suffix}; widen the region if they are real",
        ParticlesOutsideRegionWarning,
        # skip this helper and analyze: point at analyze's caller
        stacklevel=3,
    )
