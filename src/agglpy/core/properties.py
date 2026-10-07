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
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.core.agglomerates import find_contacts
from agglpy.errors import ParticleTableError

_AGGLOMERATE_INPUT = ("x", "y", "r", "agglomerate_id", "enclosed")


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
        ParticleTableError: If ``r`` is missing.
    """
    _require(table, ("r",))
    out = table.copy()
    r = out["r"].to_numpy(dtype=np.float64)
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

    ``area`` describes the circle model, not the agglomerate's outline
    on the image.

    Args:
        table: The particle table from ``find_agglomerates``; only
            ``x, y, r, agglomerate_id, enclosed`` are read.

    Returns:
        A new table, one row per agglomerate sorted by
        ``agglomerate_id`` (first column), with a fresh ``RangeIndex``.
        An empty particle table gives an empty table with every column.

    Raises:
        ParticleTableError: If a needed column is missing.
    """
    _require(table, _AGGLOMERATE_INPUT)
    x = table["x"].to_numpy(dtype=np.float64)
    y = table["y"].to_numpy(dtype=np.float64)
    r = table["r"].to_numpy(dtype=np.float64)
    enclosed = table["enclosed"].to_numpy(dtype=bool)
    # Renumber the agglomerates 0 ... k-1 ([7, 3, 7] -> ids [3, 7],
    # codes [1, 0, 1]): each property is then one array operation over
    # all agglomerates, not a Python loop (thousands per image).
    ids, codes = np.unique(
        table["agglomerate_id"].to_numpy(dtype=np.int64), return_inverse=True
    )
    groups = _Groups(codes, len(ids), r)

    columns: dict[str, NDArray[np.generic]] = {"agglomerate_id": ids}
    columns.update(_sizes(groups, r, enclosed))
    x_com, y_com, rg = _centre_of_mass(groups, x, y, r)
    columns.update(x_com=x_com, y_com=y_com, rg=rg)
    area = _union_area(groups, x, y, r, x_com, y_com)
    columns.update(area=area, D_pa=np.sqrt(4 * area / np.pi))
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
