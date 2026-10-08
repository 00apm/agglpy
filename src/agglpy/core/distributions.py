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

import numpy as np
from numpy.typing import NDArray

from agglpy.errors import ParamsError
from agglpy.params import _integer, _number, _positive

# A value this close to an edge (relative) lies on it: whole-px D times
# the pixel size misses round edges by an ulp (3 * 0.1 = 0.300...04).
_EDGE_TOL = 1e-9


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
