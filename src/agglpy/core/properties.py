"""Properties of particles and agglomerates, in px.

``particle_properties`` adds each particle's size to the particle
table; ``agglomerate_properties`` builds the agglomerate table, one row
per ``agglomerate_id``. Both take the table from
``agglomerates.find_agglomerates`` and never modify it.

Every value describes the circle model: each particle is a sphere
whose projection is its detected circle. Enclosed particles are members
everywhere: they count in every sum, count and statistic; only the
``*_with_hidden`` values count them a second time
(``docs/methodology.md``, section 7).

``DIMENSIONS`` gives the length exponent of every property column, for
the one conversion from px to physical units.
"""

from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.core.agglomerates import find_contacts
from agglpy.errors import ParticleTableError

# Length exponent of each property column: 0 count or ratio, 1 length
# (px), 2 area (px²), 3 volume (px³). D, area and volume mean the same
# for a particle and an agglomerate.
DIMENSIONS: MappingProxyType[str, int] = MappingProxyType(
    {
        "member_count": 0,
        "enclosed_count": 0,
        "volume": 3,
        "D": 1,
        "D_mean": 1,
        "D_std": 1,
        "D_largest": 1,
        "size_ratio": 0,
        "volume_with_hidden": 3,
        "D_with_hidden": 1,
        "member_count_with_hidden": 0,
        "x_com": 1,
        "y_com": 1,
        "rg": 1,
        "area": 2,
        "D_pa": 1,
        "D_feret_x": 1,
        "D_feret_y": 1,
        "D_feret_max": 1,
    }
)

_AGGLOMERATE_INPUT = ("x", "y", "r", "agglomerate_id", "enclosed")

# Agglomerates up to this size get their max Feret diameter from all
# member pairs at once; larger ones are pruned first.
_ALL_PAIRS_SIZE = 64
# Elements per temporary array in the pair loops (8 MB of float64).
_CHUNK = 2**20
# Directions searched for the first max Feret estimate of a large
# agglomerate; any number works, the pruning keeps the result exact.
_DIRECTIONS = 64


def particle_properties(table: pd.DataFrame) -> pd.DataFrame:
    """Add each particle's size, as a sphere, to the particle table.

    Adds ``D = 2r`` (px), ``area = πr²`` (px², the circle) and
    ``volume = (4/3)πr³`` (px³, the sphere).

    Args:
        table: A particle table with at least the column ``r``.

    Returns:
        A copy with the three columns, replacing any from an earlier
        run; every other column, the row order and the index are kept.

    Raises:
        ParticleTableError: If ``r`` is missing, not numeric, not
            finite or not > 0.
    """
    _require(table, ("r",))
    r = _radii(table)
    out = table.copy()
    out["D"] = 2 * r
    out["area"] = np.pi * r**2
    out["volume"] = _sphere_volume(r)
    return out


def agglomerate_properties(table: pd.DataFrame) -> pd.DataFrame:
    """Measure every agglomerate of one image.

    Members are all particles with the same ``agglomerate_id``,
    enclosed ones included. Column definitions (px, px², px³):

    - ``member_count``, ``enclosed_count``: members, enclosed members.
    - ``volume``: sum of the member sphere volumes; ``D``: diameter of
      one sphere with that volume (the agglomerate's size).
    - ``D_mean``, ``D_std``: mean and sample std (n - 1) of the member
      diameters; ``D_std`` is NaN for one member.
    - ``D_largest``: largest member diameter; ``size_ratio``: second
      largest / largest member diameter, NaN for one member.
    - ``volume_with_hidden``, ``D_with_hidden``,
      ``member_count_with_hidden``: enclosed members counted twice,
      for the particles hidden on the far side (methodology, section 7).
    - ``x_com``, ``y_com``: volume-weighted centre of mass.
    - ``rg``: radius of gyration of the member spheres with their
      heights unknown, so a lower bound of the 3D value; exact for one
      sphere (D-040).
    - ``area``: projected area, the exact area of the union of the
      member circles (overlaps counted once), not the sum of member
      areas; ``D_pa``: diameter of the circle with that area.
    - ``D_feret_x``, ``D_feret_y``: Feret diameter along the image x
      and y axis; ``D_feret_max``: largest Feret diameter.

    ``area`` and the Feret diameters describe the circle model, not the
    agglomerate's outline on the image.

    Args:
        table: The particle table from ``find_agglomerates``; only
            ``x, y, r, agglomerate_id, enclosed`` are read.

    Returns:
        A new table, one row per agglomerate sorted by
        ``agglomerate_id`` (first column), with a fresh ``RangeIndex``.
        An empty particle table gives an empty table with every column.

    Raises:
        ParticleTableError: If a needed column is missing; ``x, y, r``
            are not finite numbers or ``r <= 0``; ``agglomerate_id``
            does not hold integers; ``enclosed`` is not True / False.
    """
    _require(table, _AGGLOMERATE_INPUT)
    # A table fresh from find_agglomerates is clean, but one read back
    # from a file or merged by hand may not be: a NaN in ``enclosed``
    # would count as True and silently inflate the with-hidden values.
    x = _finite(table, "x")
    y = _finite(table, "y")
    r = _radii(table)
    enclosed = _flags(table, "enclosed")
    # Renumber the agglomerates 0 ... k-1 ([7, 3, 7] -> ids [3, 7],
    # codes [1, 0, 1]): each property is then one array operation over
    # all agglomerates, not a Python loop (thousands per image).
    ids, codes = np.unique(
        _integers(table, "agglomerate_id"), return_inverse=True
    )
    groups = _Groups(codes, len(ids), r)

    columns: dict[str, NDArray[np.generic]] = {"agglomerate_id": ids}
    columns.update(_sizes(groups, r, enclosed))
    x_com, y_com, rg = _centre_of_mass(groups, x, y, r)
    columns.update(x_com=x_com, y_com=y_com, rg=rg)
    area = _union_area(groups, x, y, r, x_com, y_com)
    columns.update(area=area, D_pa=np.sqrt(4 * area / np.pi))
    # Along a fixed axis, a union of circles reaches exactly from the
    # lowest edge (c - r) to the highest (c + r): no search needed.
    columns.update(
        D_feret_x=groups.max(x + r) - groups.min(x - r),
        D_feret_y=groups.max(y + r) - groups.min(y - r),
        D_feret_max=_feret_max(groups, x, y, r),
    )
    return pd.DataFrame(columns)


class _Groups:
    """Members of each agglomerate as positions 0 … k-1 (``codes``).

    Computed once, shared by every property, so no property loops over
    agglomerates in Python: a per-agglomerate value is an array of
    length ``k``, indexed like the output rows.

    ``order`` lists the rows agglomerate by agglomerate, largest member
    first; ``starts`` is where each agglomerate begins in it.
    """

    def __init__(
        self, codes: NDArray[np.intp], k: int, r: NDArray[np.float64]
    ) -> None:
        self.codes = codes
        self.k = k
        self.count = np.bincount(codes, minlength=k).astype(np.int64)
        # One sort, agglomerate by agglomerate, largest member first:
        # the largest and second largest members then sit at fixed
        # positions (starts, starts + 1), no search per agglomerate.
        self.order = np.lexsort((-r, codes))
        self.starts = np.cumsum(self.count) - self.count

    def sum(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        # float64 also when there are no rows (bincount gives int then)
        return np.bincount(
            self.codes, weights=values, minlength=self.k
        ).astype(np.float64)

    def max(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.full(self.k, -np.inf)
        np.maximum.at(out, self.codes, values)
        return out

    def min(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return -self.max(-values)


def _sizes(
    groups: _Groups, r: NDArray[np.float64], enclosed: NDArray[np.bool_]
) -> dict[str, NDArray[np.generic]]:
    d = 2 * r
    count = groups.count
    enclosed_count = np.bincount(
        groups.codes[enclosed], minlength=groups.k
    ).astype(np.int64)
    sphere = _sphere_volume(r)
    volume = groups.sum(sphere)
    # Each enclosed member counts once more, for its twin hidden on
    # the far side (methodology, section 7).
    with_hidden = volume + groups.sum(np.where(enclosed, sphere, 0.0))

    # Two passes (mean, then deviations): equal diameters give exactly 0.
    d_mean = groups.sum(d) / count
    deviation = d - d_mean[groups.codes]
    variance = np.full(groups.k, np.nan)
    np.divide(
        groups.sum(deviation * deviation),
        count - 1,
        out=variance,
        where=count > 1,
    )

    # Largest member first in each agglomerate (groups.order).
    d_sorted = d[groups.order]
    d_largest = d_sorted[groups.starts]
    size_ratio = np.full(groups.k, np.nan)
    many = count > 1
    size_ratio[many] = d_sorted[groups.starts[many] + 1] / d_largest[many]

    return {
        "member_count": count,
        "enclosed_count": enclosed_count,
        "volume": volume,
        "D": _equivalent_diameter(volume),
        "D_mean": d_mean,
        "D_std": np.sqrt(variance),
        "D_largest": d_largest,
        "size_ratio": size_ratio,
        "volume_with_hidden": with_hidden,
        "D_with_hidden": _equivalent_diameter(with_hidden),
        "member_count_with_hidden": count + enclosed_count,
    }


def _centre_of_mass(
    groups: _Groups,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    r: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    # Mass ∝ volume ∝ r³ (D-040).
    mass = r**3
    total = groups.sum(mass)
    x_com = groups.sum(mass * x) / total
    y_com = groups.sum(mass * y) / total
    dx = x - x_com[groups.codes]
    dy = y - y_com[groups.codes]
    # Each sphere adds its own (3/5)r² about its centre; the unknown
    # height offsets would add more, so this is a lower bound.
    inertia = groups.sum(mass * (dx * dx + dy * dy + 3 / 5 * r * r))
    return x_com, y_com, np.sqrt(inertia / total)


def _union_area(
    groups: _Groups,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    r: NDArray[np.float64],
    x_com: NDArray[np.float64],
    y_com: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exact area of the union of each agglomerate's member circles.

    Green's theorem: the area is a sum over the parts of each circle's
    outline that lie outside every other circle. Each circle starts
    with its full outline (area πr²); the parts other circles cover are
    subtracted, each covered angle once.
    """
    n = len(r)
    circle, start, end = _covered_arcs(groups.codes, x, y, r)

    # Sort the arcs circle by circle, by start angle. Each arc counts
    # only beyond the furthest end of the arcs before it on its circle,
    # so overlapping covered parts are subtracted once.
    order = np.lexsort((start, circle))
    circle, start, end = circle[order], start[order], end[order]
    # Ends lie in [0, 2π]; adding 8 per circle keeps a running max from
    # reaching into the next circle.
    offset = 8.0 * circle
    reached = np.maximum.accumulate(offset + end) - offset
    before = np.empty_like(reached)
    before[1:] = reached[:-1]
    first = np.ones(len(circle), dtype=bool)
    first[1:] = circle[1:] != circle[:-1]
    before[first] = -np.inf
    a = np.maximum(start, before)
    b = np.maximum(end, before)

    # Green's theorem, ½∮(x dy - y dx), over the arc a…b of the circle.
    # Centres relative to the agglomerate's centre of mass keep the
    # terms small (only here: the contact search needs image positions).
    rr = r[circle]
    cx = (x - x_com[groups.codes])[circle]
    cy = (y - y_com[groups.codes])[circle]
    covered = 0.5 * (
        rr * rr * (b - a)
        + cx * rr * (np.sin(b) - np.sin(a))
        - cy * rr * (np.cos(b) - np.cos(a))
    )
    outside = np.pi * r * r - np.bincount(circle, weights=covered, minlength=n)
    return groups.sum(outside)


def _covered_arcs(
    codes: NDArray[np.intp],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    r: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
    """Arcs of each circle's outline covered by another circle.

    Returns (circle, start, end) with ``0 <= start <= end <= 2π``; an
    arc across angle 0 is split in two.
    """
    n = len(r)
    contacts = find_contacts(
        pd.DataFrame({"id": np.arange(n), "x": x, "y": y, "r": r})
    )
    # Only members of one agglomerate cover each other.
    same = codes[contacts.larger] == codes[contacts.smaller]
    big, small = contacts.larger[same], contacts.smaller[same]
    dx: NDArray[np.float64] = x[small] - x[big]
    dy: NDArray[np.float64] = y[small] - y[big]
    d = np.sqrt(dx * dx + dy * dy)

    # The smaller circle inside the larger one: its whole outline is
    # covered (identical circles: the one Contacts lists as smaller).
    inside = d <= r[big] - r[small]
    # Outlines crossing: each covers an arc of the other. Tangent
    # circles touch in one point and cover nothing.
    crossing = ~inside & (d < r[big] + r[small])
    n_inside = int(inside.sum())
    circles = [small[inside]]
    starts = [np.zeros(n_inside)]
    ends = [np.full(n_inside, 2 * np.pi)]
    for own, other, sign in ((big, small, 1.0), (small, big, -1.0)):
        own, other = own[crossing], other[crossing]
        dist = d[crossing]
        # Direction to the other centre, ± the half-angle of the arc
        # (law of cosines; clipped against rounding near tangency).
        towards = np.arctan2(sign * dy[crossing], sign * dx[crossing])
        cos_half = (r[own] ** 2 + dist**2 - r[other] ** 2) / (
            2 * r[own] * dist
        )
        half = np.arccos(np.clip(cos_half, -1.0, 1.0))
        start = np.mod(towards - half, 2 * np.pi)
        end = start + 2 * half
        wraps = end > 2 * np.pi
        circles += [own, own[wraps]]
        starts += [start, np.zeros(int(wraps.sum()))]
        ends += [np.minimum(end, 2 * np.pi), end[wraps] - 2 * np.pi]
    return (
        np.concatenate(circles),
        np.concatenate(starts),
        np.concatenate(ends),
    )


def _feret_max(
    groups: _Groups,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    r: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Largest Feret diameter: max of ``d_ij + r_i + r_j`` over pairs.

    The pair ``i = j`` is included (``2r``). Small agglomerates take all
    pairs at once, grouped by size; large ones are pruned first.
    """
    out = np.zeros(groups.k)
    xs, ys, rs = x[groups.order], y[groups.order], r[groups.order]
    # Agglomerates of equal size stack into one (n, size, size) array:
    # thousands of small agglomerates take a few NumPy calls, no loop.
    for size in np.unique(groups.count):
        same_size = np.flatnonzero(groups.count == size)
        if size > _ALL_PAIRS_SIZE:
            for g in same_size:
                one = slice(groups.starts[g], groups.starts[g] + size)
                out[g] = _pruned_feret_max(xs[one], ys[one], rs[one])
            continue
        # One block of (agglomerates, size, size) pairs per chunk.
        step = max(1, _CHUNK // int(size * size))
        for first in range(0, len(same_size), step):
            chunk = same_size[first : first + step]
            rows = groups.starts[chunk][:, None] + np.arange(size)
            out[chunk] = _pair_max_rows(xs[rows], ys[rows], rs[rows])
    return out


def _pruned_feret_max(
    x: NDArray[np.float64], y: NDArray[np.float64], r: NDArray[np.float64]
) -> float:
    # A first estimate from the members furthest out in a few
    # directions: an exact value of some pair, so a lower bound.
    angle = np.linspace(0, np.pi, _DIRECTIONS, endpoint=False)
    along = x[:, None] * np.cos(angle) + y[:, None] * np.sin(angle)
    ends = np.concatenate(
        [
            np.argmax(along + r[:, None], axis=0),
            np.argmin(along - r[:, None], axis=0),
        ]
    )
    candidates = np.unique(ends)
    low = _pair_max(x[candidates], y[candidates], r[candidates])
    # Triangle inequality: d_ij + r_i + r_j <= reach_i + reach_j, with
    # reach = distance from the bounding box centre + r. A member whose
    # reach plus the largest reach can't beat the estimate can't be in
    # a longer pair. The tiny margin covers rounding.
    cx = (np.max(x + r) + np.min(x - r)) / 2
    cy = (np.max(y + r) + np.min(y - r)) / 2
    reach = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) + r
    keep = reach + reach.max() >= low * (1 - 1e-9)
    return max(low, _pair_max(x[keep], y[keep], r[keep]))


def _pair_max(
    x: NDArray[np.float64], y: NDArray[np.float64], r: NDArray[np.float64]
) -> float:
    """Max of ``d_ij + r_i + r_j`` over all pairs, in chunks of rows."""
    best = 0.0
    step = max(1, _CHUNK // len(x))
    for i in range(0, len(x), step):
        dx = x[i : i + step, None] - x
        dy = y[i : i + step, None] - y
        span = np.sqrt(dx * dx + dy * dy) + r[i : i + step, None] + r
        best = max(best, float(span.max()))
    return best


def _pair_max_rows(
    x: NDArray[np.float64], y: NDArray[np.float64], r: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``_pair_max`` for each row of (agglomerates, members) arrays."""
    dx = x[:, :, None] - x[:, None, :]
    dy = y[:, :, None] - y[:, None, :]
    span = np.sqrt(dx * dx + dy * dy) + r[:, :, None] + r[:, None, :]
    best: NDArray[np.float64] = span.reshape(len(x), -1).max(axis=1)
    return best


def _sphere_volume(r: NDArray[np.float64]) -> NDArray[np.float64]:
    return 4 / 3 * np.pi * r**3


def _equivalent_diameter(
    volume: NDArray[np.float64],
) -> NDArray[np.float64]:
    # Diameter of the sphere with this volume.
    diameter: NDArray[np.float64] = np.cbrt(6 * volume / np.pi)
    return diameter


def _require(table: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ParticleTableError(f"missing columns: {missing}")


def _finite(table: pd.DataFrame, column: str) -> NDArray[np.float64]:
    try:
        values = pd.to_numeric(table[column], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ParticleTableError(
            f"column {column!r} must be numeric: {exc}"
        ) from exc
    array = values.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(array).all():
        raise ParticleTableError(f"column {column!r} must be finite")
    return array


def _radii(table: pd.DataFrame) -> NDArray[np.float64]:
    r = _finite(table, "r")
    if (r <= 0).any():
        raise ParticleTableError("column 'r' must be > 0")
    return r


def _integers(table: pd.DataFrame, column: str) -> NDArray[np.int64]:
    if pd.api.types.is_integer_dtype(table[column]):
        return table[column].to_numpy(dtype=np.int64)
    # Floats are accepted when they are whole numbers (e.g. 3.0 after
    # a CSV round trip); 0.5 or NaN would be cut or crash silently.
    values = _finite(table, column)
    if (values != np.round(values)).any():
        raise ParticleTableError(f"column {column!r} must hold integers")
    return values.astype(np.int64)


def _flags(table: pd.DataFrame, column: str) -> NDArray[np.bool_]:
    # Only real True / False: casting would turn NaN, "False" or 2
    # into True without a word.
    values = table[column]
    if values.isna().any() or (
        len(values)
        and pd.api.types.infer_dtype(values, skipna=False) != "boolean"
    ):
        raise ParticleTableError(
            f"column {column!r} must hold True / False, without gaps"
        )
    return values.to_numpy(dtype=bool)
