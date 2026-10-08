"""Size classes and distributions of any numeric column.

``size_classes`` builds the class edges; ``distribution`` counts the
values of one column per class and group of images (particle size,
agglomerate size, member count or any other column), in number and
optionally weighted by other columns; ``distribution_across_images``
gives the mean ± confidence interval of each class's fraction over the
images of a group.

Unit-agnostic, like ``metrics``: edges and values must share one unit.
"""

import math
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from agglpy.core._images import (
    Grouping,
    check_images,
    group_images,
    image_positions,
    names,
    numbers,
    require,
)
from agglpy.core.metrics import values_across_images
from agglpy.errors import ParamsError, TableError, ValuesNotCountedWarning
from agglpy.params import _integer, _number, _positive

# A value this close to an edge (relative) lies on it: whole-px D times
# the pixel size misses round edges by an ulp (3 * 0.1 = 0.300...04).
_EDGE_TOL = 1e-9
_MAX_LISTED = 10
_CLASS_COLUMNS = ("left", "mid", "right", "width")


def size_classes(
    start: float,
    end: float,
    *,
    count: int | None = None,
    step: float | None = None,
    scale: str = "linear",
) -> NDArray[np.float64]:
    """Edges of size classes from ``start`` to ``end``, both included.

    Give exactly one of ``count`` (number of classes) or ``step``:

    - ``scale="linear"``: ``step`` is the class width. It must divide
      ``end - start``: ``size_classes(0, 10, step=2.5)`` gives
      ``[0, 2.5, 5, 7.5, 10]``; ``step=3`` raises.
    - ``scale="log"``: ``step`` is the **factor** between consecutive
      edges, ``start`` must be > 0. ``size_classes(0.1, 0.8, step=2)``
      gives ``[0.1, 0.2, 0.4, 0.8]``; ten classes per decade is
      ``step=10**0.1``.

    Any increasing array of edges works in ``distribution`` too; this
    is a helper for the usual ones.

    Returns:
        The edges, float64; ``count + 1`` of them.

    Raises:
        ParamsError: If an argument is invalid or ``step`` does not
            divide the range.
    """
    low = _number("start", start)
    high = _number("end", end)
    if high <= low:
        raise ParamsError(f"end must be > start, got {start!r}, {end!r}")
    if scale not in ("linear", "log"):
        raise ParamsError(f"scale must be 'linear' or 'log', got {scale!r}")
    if (count is None) == (step is None):
        raise ParamsError("give exactly one of count and step")
    if scale == "log" and low <= 0:
        raise ParamsError(f"log size classes need start > 0, got {start!r}")
    if count is not None:
        n = _integer("count", count)
        if n < 1:
            raise ParamsError(f"count must be >= 1, got {count!r}")
    else:
        n = _classes_per_step(low, high, step, scale)
    if scale == "log":
        edges = np.geomspace(low, high, n + 1)
    else:
        edges = np.linspace(low, high, n + 1)
    # Exact outer edges (geomspace may miss them by an ulp).
    edges[0], edges[-1] = low, high
    return edges


def distribution(
    images: pd.DataFrame,
    table: pd.DataFrame,
    column: str,
    edges: ArrayLike,
    by: str | Iterable[str] | None = None,
    *,
    closed: str = "right",
    weights: str | Iterable[str] = (),
) -> pd.DataFrame:
    """Distribution of ``column`` over size classes, per group of images.

    One function for the particle size distribution (particle ``D``),
    the agglomerate size distribution (agglomerate ``D``), the member
    count distribution (``member_count``) and any numeric column.

    - Classes are ``(a, b]`` (``closed="right"``) or ``[a, b)``
      (``"left"``); the outer edge is always included, so a value equal
      to the first (or last) edge is counted.
    - A value within 1e-9 (relative) of an edge counts as on the edge.
    - Values outside the classes and NaN values are not counted; a
      warning says how many per group. Fractions refer to the counted
      values.
    - Every class appears in every group, zeros included; a group
      without values has zero amounts and NaN fractions.

    Args:
        images: The images table; the groups come from it.
        table: A table with ``image``, ``column`` and the ``weights``
            columns (particle or agglomerate table).
        column: The column to classify.
        edges: Increasing class edges (e.g. ``size_classes``).
        by: As ``metrics.summary``.
        closed: ``"right"`` or ``"left"``.
        weights: Column names to weight by, e.g. ``"volume"``; each
            adds one ``basis``. Its values must be finite wherever the
            value is counted.

    Returns:
        A long table: the ``by`` columns, ``left, mid, right, width``,
        ``basis`` (``"number"``, then each weight column), ``amount``
        (count, or the sum of the weight column, per class),
        ``fraction`` (sums to 1 over the classes), ``cumulative`` (ends
        at 1), ``density = fraction / width`` and ``log_density =
        fraction / (log10 right - log10 left)`` (dN/dlogD; NaN where
        ``left <= 0``).

    Raises:
        TableError: If a column is missing or not numeric, a weight is
            not finite on a counted value, or an image is unknown.
        ParamsError: If ``edges`` or ``closed`` is invalid.

    Warns:
        ValuesNotCountedWarning: If values were left out.
    """
    check_images(images)
    groups = group_images(images, by)
    weight_columns = names(weights, "weights")
    bounds = _edges(edges)
    if closed not in ("right", "left"):
        raise ParamsError(f"closed must be 'right' or 'left', got {closed!r}")
    require(table, [column, *weight_columns], "table")
    group = groups.codes[image_positions(images, table, "table")]
    values = numbers(table, column, "table")
    classes = _classify(values, bounds, closed)
    counted = classes >= 0
    bases: dict[str, NDArray[np.float64] | None] = {"number": None}
    for name in weight_columns:
        given = numbers(table, name, "table")
        bad = counted & ~np.isfinite(given)
        if bad.any():
            raise TableError(
                f"weights: column {name!r} is NaN or infinite on "
                f"{int(bad.sum())} counted value(s)"
            )
        bases[name] = given
    _warn_not_counted(groups, group, values, counted, column)

    k, m = groups.k, len(bounds) - 1
    left, right = bounds[:-1], bounds[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_width = np.where(left > 0, np.log10(right / left), np.nan)
    # Each counted value's (group, class) as one number, for bincount.
    cell = group[counted] * m + classes[counted]
    frames = []
    for basis, weight in bases.items():
        amount = np.bincount(
            cell,
            weights=None if weight is None else weight[counted],
            minlength=k * m,
        ).astype(np.float64)
        amount = amount.reshape(k, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = amount / amount.sum(axis=1, keepdims=True)
            frames.append(
                pd.DataFrame(
                    {
                        "_group": np.repeat(np.arange(k), m),
                        "left": np.tile(left, k),
                        "mid": np.tile((left + right) / 2, k),
                        "right": np.tile(right, k),
                        "width": np.tile(right - left, k),
                        "basis": basis,
                        "amount": amount.ravel(),
                        "fraction": fraction.ravel(),
                        "cumulative": np.cumsum(fraction, axis=1).ravel(),
                        "density": (fraction / (right - left)).ravel(),
                        "log_density": (fraction / log_width).ravel(),
                    }
                )
            )
    # Group, then basis, then class.
    out = pd.concat(frames, ignore_index=True).sort_values(
        "_group", kind="stable", ignore_index=True
    )
    labels = groups.labels.iloc[out.pop("_group")].reset_index(drop=True)
    return pd.concat([labels, out], axis=1)


def distribution_across_images(
    images: pd.DataFrame,
    table: pd.DataFrame,
    column: str,
    edges: ArrayLike,
    by: str | Iterable[str] | None = None,
    *,
    closed: str = "right",
    weights: str | Iterable[str] = (),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """The fraction of each class per image, then mean ± CI over images.

    ``distribution(..., by="image")``, then ``values_across_images``
    with the class and basis as keys. An image without counted values
    has NaN fractions: it is left out of the means and of ``n``.

    Args:
        images, table, column, edges, by, closed, weights: As
            ``distribution``.
        confidence: Level of the confidence interval.

    Returns:
        A long table: the ``by`` columns, ``left, mid, right, width``,
        ``basis``, ``metric`` (``"fraction"``), ``mean``, ``std``,
        ``n``, ``ci_low``, ``ci_high``.
    """
    per_image = distribution(
        images, table, column, edges, "image", closed=closed, weights=weights
    )
    keys = [*_CLASS_COLUMNS, "basis"]
    return values_across_images(
        images,
        per_image[["image", *keys, "fraction"]],
        by,
        keys=keys,
        confidence=confidence,
    )


def _classes_per_step(
    low: float, high: float, step: float | None, scale: str
) -> int:
    size = _positive("step", step)
    if scale == "log":
        if size <= 1:
            raise ParamsError(
                f"a log step is the factor between edges and must be > 1 "
                f"(e.g. 2, or 10**0.1 for ten classes per decade), "
                f"got {step!r}"
            )
        exact = math.log(high / low) / math.log(size)
    else:
        exact = (high - low) / size
    n = round(exact)
    # Count the classes first, then place the edges (np.arange would
    # add an edge beyond end through rounding).
    if n < 1 or abs(exact - n) > _EDGE_TOL * n:
        raise ParamsError(
            f"step {step!r} does not divide the range {low!r} … {high!r} "
            f"({exact:.6g} classes)"
        )
    return n


def _edges(edges: ArrayLike) -> NDArray[np.float64]:
    try:
        bounds = np.asarray(edges, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ParamsError(f"edges must be numbers: {exc}") from exc
    if bounds.ndim != 1 or len(bounds) < 2:
        raise ParamsError("edges must be a 1-D array of at least 2 edges")
    if not np.isfinite(bounds).all() or (np.diff(bounds) <= 0).any():
        raise ParamsError("edges must be finite and strictly increasing")
    return bounds


def _classify(
    values: NDArray[np.float64], edges: NDArray[np.float64], closed: str
) -> NDArray[np.intp]:
    """Class index of each value; -1 if outside the classes or NaN."""
    m = len(edges) - 1
    # Move a value lying within the tolerance of an edge onto it.
    upper = np.clip(np.searchsorted(edges, values), 1, m)
    below, above = edges[upper - 1], edges[upper]
    nearest = np.where(
        np.abs(values - below) <= np.abs(above - values), below, above
    )
    on_edge = np.abs(values - nearest) <= _EDGE_TOL * np.abs(nearest)
    v = np.where(on_edge, nearest, values)
    if closed == "right":
        # (a, b]: a value on edge i belongs to class i - 1
        index = np.searchsorted(edges, v, side="left") - 1
        index[v == edges[0]] = 0
    else:
        # [a, b): a value on edge i belongs to class i
        index = np.searchsorted(edges, v, side="right") - 1
        index[v == edges[-1]] = m - 1
    inside = (index >= 0) & (index < m) & ~np.isnan(v)
    out: NDArray[np.intp] = np.where(inside, index, -1)
    return out


def _warn_not_counted(
    groups: Grouping,
    group: NDArray[np.intp],
    values: NDArray[np.float64],
    counted: NDArray[np.bool_],
    column: str,
) -> None:
    if counted.all():
        return
    missing = np.isnan(values)
    outside = np.bincount(group[~counted & ~missing], minlength=groups.k)
    gaps = np.bincount(group[missing], minlength=groups.k)
    parts = []
    for g in np.flatnonzero(outside + gaps):
        where = ""
        if groups.columns:
            label = groups.labels.iloc[g].tolist()
            where = f"{label[0] if len(label) == 1 else tuple(label)}: "
        parts.append(f"{where}{outside[g]} outside, {gaps[g]} NaN")
    listed = "; ".join(parts[:_MAX_LISTED])
    more = len(parts) - _MAX_LISTED
    suffix = f"; and {more} more groups" if more > 0 else ""
    warnings.warn(
        f"distribution of {column!r}: values not counted ({listed}"
        f"{suffix}); the fractions refer to the counted values",
        ValuesNotCountedWarning,
        stacklevel=3,
    )
