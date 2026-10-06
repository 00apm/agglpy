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
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from agglpy.errors import DuplicateParticlesWarning
from agglpy.tables import find_duplicates, validate_particles

# Widens the candidate search a little, so rounding inside the KD-tree
# can't drop a pair that touches exactly; the exact test decides.
_SEARCH_MARGIN = 1e-9
_MAX_LISTED = 10


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


def find_agglomerates(
    particles: pd.DataFrame, contacts: Contacts
) -> pd.Series:
    """Give each particle the id of the agglomerate it belongs to.

    Particles linked by a chain of contacts share an agglomerate; a
    particle without contacts is an agglomerate of its own. Ids run
    from 0, ordered by each agglomerate's smallest particle id, so
    they depend only on the particles, not on the row order.

    Args:
        particles: A validated particle table.
        contacts: Its contacts (``find_contacts``).

    Returns:
        ``agglomerate_id`` (int64) per particle, on the table's index.
    """
    n = len(particles)
    if n == 0:
        return pd.Series(
            [], index=particles.index, dtype=np.int64, name="agglomerate_id"
        )
    # Contacts are the edges of a graph whose nodes are the particles;
    # scipy's "labels" are its connected groups: our agglomerates.
    edges = np.ones(len(contacts.larger), dtype=np.int8)
    graph = coo_matrix(
        (edges, (contacts.larger, contacts.smaller)), shape=(n, n)
    )
    count, labels = connected_components(graph, directed=False)

    # scipy numbers the groups in row order; renumber them by their
    # smallest particle id.
    ids = particles["id"].to_numpy(dtype=np.int64)
    smallest = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(smallest, labels, ids)
    rank = np.empty(count, dtype=np.int64)
    rank[np.argsort(smallest)] = np.arange(count)
    return pd.Series(
        rank[labels], index=particles.index, name="agglomerate_id"
    )


def find_enclosed(particles: pd.DataFrame, contacts: Contacts) -> pd.Series:
    """Flag particles lying completely inside a larger particle.

    A particle is enclosed when its circle, outline included, is inside
    the circle of a particle it touches: ``d <= r_large - r_small``.
    For two identical circles only the one with the higher id is
    enclosed.

    Args:
        particles: A validated particle table.
        contacts: Its contacts (``find_contacts``).

    Returns:
        ``enclosed`` (bool) per particle, on the table's index.

    Warns:
        DuplicateParticlesWarning: If an enclosed particle and the one
            enclosing it are near-identical (see
            ``tables.find_duplicates``): a detection error that is
            counted as an enclosed particle.
    """
    enclosed = np.zeros(len(particles), dtype=bool)
    if len(contacts.larger):
        xy = particles[["x", "y"]].to_numpy(dtype=np.float64)
        r = particles["r"].to_numpy(dtype=np.float64)
        larger, smaller = contacts
        # Contacts are oriented: only the smaller member can be inside.
        inside = _distance(xy, larger, smaller) <= r[larger] - r[smaller]
        enclosed[smaller[inside]] = True
        if inside.any():
            _warn_enclosed_duplicates(
                particles, larger[inside], smaller[inside]
            )
    return pd.Series(enclosed, index=particles.index, name="enclosed")


def _warn_enclosed_duplicates(
    particles: pd.DataFrame,
    larger: NDArray[np.intp],
    smaller: NDArray[np.intp],
) -> None:
    # A circle enclosed by a near-identical one is a detection error
    # (D-038); "near-identical" means what tables.find_duplicates says.
    ids = particles["id"].to_numpy()
    enclosed_pairs = {
        (int(min(a, b)), int(max(a, b)))
        for a, b in zip(ids[larger], ids[smaller], strict=True)
    }
    pairs = sorted(enclosed_pairs.intersection(find_duplicates(particles)))
    if not pairs:
        return
    listed = ", ".join(map(str, pairs[:_MAX_LISTED]))
    more = len(pairs) - _MAX_LISTED
    suffix = f" and {more} more" if more > 0 else ""
    warnings.warn(
        f"{len(pairs)} enclosed particle(s) lie inside a near-identical "
        f"circle (id pairs): {listed}{suffix}; each counts as enclosed",
        DuplicateParticlesWarning,
        stacklevel=3,
    )


def _distance(
    xy: NDArray[np.float64], a: NDArray[np.intp], b: NDArray[np.intp]
) -> NDArray[np.float64]:
    # sqrt(dx² + dy²) like the legacy code: IEEE sqrt is correctly
    # rounded, so a distance that is exact in theory is exact here.
    # Annotated: numpy's stubs type xy[a, 0] as Any.
    dx: NDArray[np.float64] = xy[a, 0] - xy[b, 0]
    dy: NDArray[np.float64] = xy[a, 1] - xy[b, 1]
    return np.sqrt(dx * dx + dy * dy)


def group(particles: pd.DataFrame) -> pd.DataFrame:
    """Group particles into agglomerates and flag enclosed particles.

    Args:
        particles: A particle table.

    Returns:
        A validated copy (fresh index, rows in input order) with the
        columns ``agglomerate_id`` (int64) and ``enclosed`` (bool),
        replacing any from an earlier run.

    Raises:
        ParticleTableError: If the table breaks the schema.

    Warns:
        DuplicateParticlesWarning: See ``find_enclosed``.
    """
    # The caller saw the general duplicate warning when the table was
    # made; only the one about enclosed duplicates comes from here.
    table = validate_particles(particles, warn_duplicates=False)
    contacts = find_contacts(table)
    table["agglomerate_id"] = find_agglomerates(table, contacts)
    table["enclosed"] = find_enclosed(table, contacts)
    return table
