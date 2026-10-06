"""Group particles into agglomerates.

Two particles touch when the distance of their centres is at most the
sum of their radii, compared exactly in px (D-037). An agglomerate is a
set of particles linked by a chain of contacts. Every particle belongs
to exactly one, so the whole grouping is one column of the particle
table, ``agglomerate_id``. ``enclosed`` marks a particle lying
completely inside a larger particle (old name ``idj``).

The functions take a validated particle table
(``tables.validate_particles``) and never modify it; ``group`` runs
them all.
"""

import itertools
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import cKDTree

# Widens the candidate search a little, so rounding inside the KD-tree
# can't drop a pair that touches exactly; the exact test decides.
_SEARCH_MARGIN = 1e-9


class Contacts(NamedTuple):
    """Pairs of touching particles, as row positions in the table.

    Each pair appears once. ``larger[k]`` is the member with the larger
    radius (equal radii: the lower id), ``smaller[k]`` the other one.
    """

    larger: NDArray[np.intp]
    smaller: NDArray[np.intp]


def find_contacts(particles: pd.DataFrame) -> Contacts:
    """Find every pair of touching particles.

    Each particle searches the distance ``2 * r`` around its centre
    and keeps only partners not larger than itself, so a large particle
    searches far once and the many small ones search only nearby.

    Args:
        particles: A validated particle table.

    Returns:
        The touching pairs (see ``Contacts``).
    """
    n = len(particles)
    if n < 2:
        empty = np.empty(0, dtype=np.intp)
        return Contacts(empty, empty.copy())
    xy = particles[["x", "y"]].to_numpy(dtype=np.float64)
    r = particles["r"].to_numpy(dtype=np.float64)
    ids = particles["id"].to_numpy()

    # A partner no larger than particle i that touches it lies within
    # r_i + r_j <= 2 * r_i: one query, with a radius per point.
    found = cKDTree(xy).query_ball_point(xy, 2 * r * (1 + _SEARCH_MARGIN))
    counts = np.fromiter(map(len, found), dtype=np.intp, count=n)
    larger = np.repeat(np.arange(n, dtype=np.intp), counts)
    smaller = np.fromiter(
        itertools.chain.from_iterable(found),
        dtype=np.intp,
        count=int(counts.sum()),
    )

    # Keep each pair once, owned by its larger member (equal radii: the
    # lower id). This also drops every particle finding itself.
    owned = (r[smaller] < r[larger]) | (
        (r[smaller] == r[larger]) & (ids[smaller] > ids[larger])
    )
    larger, smaller = larger[owned], smaller[owned]

    touching = _distance(xy, larger, smaller) <= r[larger] + r[smaller]
    return Contacts(larger[touching], smaller[touching])


def _distance(
    xy: NDArray[np.float64], a: NDArray[np.intp], b: NDArray[np.intp]
) -> NDArray[np.float64]:
    # sqrt(dx² + dy²) like the legacy code: IEEE sqrt is correctly
    # rounded, so a distance that is exact in theory is exact here.
    # Annotated: numpy's stubs type xy[a, 0] as Any.
    dx: NDArray[np.float64] = xy[a, 0] - xy[b, 0]
    dy: NDArray[np.float64] = xy[a, 1] - xy[b, 1]
    return np.sqrt(dx * dx + dy * dy)
