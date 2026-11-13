"""Group particles into agglomerates and list their contacts.

Two particles touch when the distance of their centres is at most the
sum of their radii, compared exactly in px. An agglomerate is a set of
particles linked by a chain of contacts. Every particle belongs to
exactly one, so the whole grouping is one column of the particle table,
``agglomerate_id``. ``enclosed`` marks a particle lying completely
inside a larger particle (old name ``idj``).

The contacts table lists the *possible* contacts: every pair of
overlapping circles except an outer X and an inner C with some B such
that C lies inside B and B inside X (an inner particle cannot reach an
outer one through the one between). A circle shows only the projection
of a sphere, so overlapping circles need not touch: within an
agglomerate of n members, ``n - 1 <= true contacts <= possible
contacts``. Caveats: 2D projection; tangency decided within about 1 px
of the detection; duplicate circles (warned) add contacts; agglomerates
cut by the image edge (``touches_edge``) and particles hidden behind
others miss theirs.

``find_agglomerates`` takes a validated particle table
(``tables.validate_particles``), never modifies it and runs the private
steps (contacts, labels, enclosed, possible contacts) in order.
``select_agglomerates`` takes a consistent subset of the particle and
agglomerate tables.
"""

import itertools
import warnings
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from agglpy.errors import DuplicateParticlesWarning, TableError
from agglpy.tables import find_duplicates, validate_particles

# Widens the candidate search a little, so rounding inside the KD-tree
# can't drop a pair that touches exactly; the exact test decides.
_SEARCH_MARGIN = 1e-9
_MAX_LISTED = 10


class _Contacts(NamedTuple):
    """Pairs of touching particles, as row positions in the table.

    Each pair appears once. ``larger[k]`` is the member with the larger
    radius (equal radii: the lower id), ``smaller[k]`` the other one.
    """

    larger: NDArray[np.intp]
    smaller: NDArray[np.intp]


def _find_contacts(particles: pd.DataFrame) -> _Contacts:
    """Find every pair of touching particles.

    Each particle searches the distance ``2 * r`` around its centre
    and keeps only partners not larger than itself, so a large particle
    searches far once and the many small ones search only nearby.

    Parameters
    ----------
    particles : DataFrame
        A validated particle table.

    Returns
    -------
    _Contacts
        The touching pairs (see ``_Contacts``).
    """
    n = len(particles)
    if n < 2:
        empty = np.empty(0, dtype=np.intp)
        return _Contacts(empty, empty.copy())
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
    return _Contacts(larger[touching], smaller[touching])


def _label_agglomerates(
    particles: pd.DataFrame, contacts: _Contacts
) -> pd.Series:
    """Give each particle the id of the agglomerate it belongs to.

    Particles linked by a chain of contacts share an agglomerate; a
    particle without contacts is an agglomerate of its own. Ids run
    from 0, ordered by each agglomerate's smallest particle id, so
    they depend only on the particles, not on the row order.

    Parameters
    ----------
    particles : DataFrame
        A validated particle table.
    contacts : _Contacts
        Its contacts (``_find_contacts``).

    Returns
    -------
    Series
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


def _find_inside(
    particles: pd.DataFrame, contacts: _Contacts
) -> NDArray[np.bool_]:
    """Per contact: the smaller circle lies inside the larger one.

    Outline included: ``d <= r_large - r_small``. Contacts are
    oriented, so only the smaller member can be inside; of two
    identical circles, the one with the higher id.
    """
    if not len(contacts.larger):
        return np.zeros(0, dtype=bool)
    xy = particles[["x", "y"]].to_numpy(dtype=np.float64)
    r = particles["r"].to_numpy(dtype=np.float64)
    larger, smaller = contacts
    inside: NDArray[np.bool_] = (
        _distance(xy, larger, smaller) <= r[larger] - r[smaller]
    )
    return inside


def _find_enclosed(
    particles: pd.DataFrame, contacts: _Contacts, inside: NDArray[np.bool_]
) -> pd.Series:
    """Flag particles lying completely inside a larger particle.

    A particle is enclosed when its circle, outline included, is inside
    the circle of a particle it touches (``_find_inside``). For two
    identical circles only the one with the higher id is enclosed.

    Parameters
    ----------
    particles : DataFrame
        A validated particle table.
    contacts : _Contacts
        Its contacts (``_find_contacts``).
    inside : ndarray
        Per contact, whether it is an enclosure
        (``_find_inside``).

    Returns
    -------
    Series
        ``enclosed`` (bool) per particle, on the table's index.

    Warns
    -----
    DuplicateParticlesWarning
        If an enclosed particle and the one
        enclosing it are near-identical (see
        ``tables.find_duplicates``): a detection error that is
        counted as an enclosed particle.
    """
    enclosed = np.zeros(len(particles), dtype=bool)
    if inside.any():
        larger, smaller = contacts.larger[inside], contacts.smaller[inside]
        enclosed[smaller] = True
        _warn_enclosed_duplicates(particles, larger, smaller)
    return pd.Series(enclosed, index=particles.index, name="enclosed")


def _possible(
    contacts: _Contacts, inside: NDArray[np.bool_], n: int
) -> NDArray[np.bool_]:
    """Per contact: it can be real (not X-C with C inside B inside X)."""
    keep = np.ones(len(inside), dtype=bool)
    if inside.any():
        big, small = contacts.larger[inside], contacts.smaller[inside]
        held = csr_matrix(
            (np.ones(len(big), dtype=np.int64), (big, small)), shape=(n, n)
        )
        # (held @ held)[X, C] > 0: some B holds C and is held by X. Only
        # enclosures can be dropped: a crossing pair always stays.
        via = np.asarray((held @ held)[big, small]).ravel()
        keep[np.flatnonzero(inside)[via > 0]] = False
    return keep


# The columns a contacts table needs; ``inside`` is for information.
_PAIR_COLUMNS = ("agglomerate_id", "id_a", "id_b")


class _Pairs(NamedTuple):
    """A checked contacts table, read once (``_check_contacts``).

    ``ids`` are the particle ids in row order; ``agglomerate_id``,
    ``id_a`` and ``id_b`` one value per pair.
    """

    ids: NDArray[np.int64]
    agglomerate_id: NDArray[np.int64]
    id_a: NDArray[np.int64]
    id_b: NDArray[np.int64]


def _contacts_table(
    particles: pd.DataFrame,
    contacts: _Contacts,
    inside: NDArray[np.bool_],
    keep: NDArray[np.bool_],
) -> pd.DataFrame:
    """The kept contacts by particle id, sorted by agglomerate and ids."""
    ids = particles["id"].to_numpy(dtype=np.int64)
    labels = particles["agglomerate_id"].to_numpy(dtype=np.int64)
    larger, smaller = contacts.larger[keep], contacts.smaller[keep]
    table = pd.DataFrame(
        {
            "agglomerate_id": labels[larger],
            "id_a": ids[larger],
            "id_b": ids[smaller],
            "inside": inside[keep],
        }
    ).astype({"agglomerate_id": np.int64, "id_a": np.int64, "id_b": np.int64})
    order = np.lexsort((table["id_b"], table["id_a"], table["agglomerate_id"]))
    return table.iloc[order].reset_index(drop=True)


def _warn_enclosed_duplicates(
    particles: pd.DataFrame,
    larger: NDArray[np.intp],
    smaller: NDArray[np.intp],
) -> None:
    # A circle enclosed by a near-identical one is a detection error
    # (D-038); "near-identical" means what tables.find_duplicates says.
    ids = particles["id"].to_numpy()
    # Pair as find_duplicates orders it -> id of its enclosed member.
    enclosed_by_pair = {
        (int(min(a, b)), int(max(a, b))): int(b)
        for a, b in zip(ids[larger], ids[smaller], strict=True)
    }
    pairs = sorted(enclosed_by_pair.keys() & set(find_duplicates(particles)))
    if not pairs:
        return
    # Count particles, not pairs: three identical circles are 3 pairs
    # but only 2 enclosed particles.
    count = len({enclosed_by_pair[p] for p in pairs})
    listed = ", ".join(map(str, pairs[:_MAX_LISTED]))
    more = len(pairs) - _MAX_LISTED
    suffix = f" and {more} more" if more > 0 else ""
    warnings.warn(
        f"{count} enclosed particle(s) lie inside a near-identical "
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


def find_agglomerates(
    particles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group particles into agglomerates and list their contacts.

    Parameters
    ----------
    particles : DataFrame
        A particle table.

    Returns
    -------
    tuple
        ``(particles, contacts)``. ``particles``: a validated copy
        (fresh index, rows in input order) with the columns
        ``agglomerate_id`` (int64), ``enclosed`` (bool) and
        ``coordination`` (int64, the particle's possible contacts),
        replacing any from an earlier run. ``contacts``: one row per
        possible contact (see the module docstring), columns
        ``agglomerate_id``, ``id_a``, ``id_b`` (int64; ``id_a`` the
        larger particle, of equal ones the lower id) and ``inside``
        (bool: ``id_b`` lies inside ``id_a``), sorted by
        ``agglomerate_id, id_a, id_b``, with a fresh ``RangeIndex``.
        Possible contacts are an upper bound: ``coordination`` may
        count a particle that only overlaps in the projection.

    Raises
    ------
    ParticleTableError
        If the table breaks the schema.

    Warns
    -----
    DuplicateParticlesWarning
        See ``_find_enclosed``.
    """
    # The caller saw the general duplicate warning when the table was
    # made; only the one about enclosed duplicates comes from here.
    table = validate_particles(particles, warn_duplicates=False)
    n = len(table)
    contacts = _find_contacts(table)
    table["agglomerate_id"] = _label_agglomerates(table, contacts)
    inside = _find_inside(table, contacts)
    table["enclosed"] = _find_enclosed(table, contacts, inside)
    keep = _possible(contacts, inside, n)
    ends = np.concatenate([contacts.larger[keep], contacts.smaller[keep]])
    table["coordination"] = np.bincount(ends, minlength=n).astype(np.int64)
    return table, _contacts_table(table, contacts, inside, keep)


def _check_contacts(particles: pd.DataFrame, contacts: pd.DataFrame) -> _Pairs:
    """Check that ``contacts`` are the contact pairs of these particles.

    Particle ids must be unique. Every ``id_a`` / ``id_b`` must be a
    particle id, both particles must belong to the row's
    ``agglomerate_id``, no pair may join a particle with itself or
    repeat, and the pairs must hold every agglomerate together: a
    filtered table would give wrong areas, outlines and negative loop
    counts. Ids may be whole floats (a table read back from a CSV
    file). ``inside`` is not needed.

    Returns
    -------
    _Pairs
        The ids, read once, for the caller to use.

    Raises
    ------
    TableError
        If a column is missing, an id is not an integer or
        repeats, or the pairs do not match the particles.
    """
    for name, table, columns in (
        ("particles", particles, ("id", "agglomerate_id")),
        ("contacts", contacts, _PAIR_COLUMNS),
    ):
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise TableError(f"{name}: missing columns {missing}")
    ids = _whole(particles, "id", "particles")
    repeats = pd.Index(ids).duplicated()
    if repeats.any():
        raise TableError(
            f"particles: ids {_some(np.unique(ids[repeats]))} repeat; "
            "each particle needs its own id"
        )
    labels = pd.Series(_whole(particles, "agglomerate_id", "particles"), ids)
    pair_label = _whole(contacts, "agglomerate_id", "contacts")
    ends = {e: _whole(contacts, e, "contacts") for e in ("id_a", "id_b")}
    unknown = np.setdiff1d(np.concatenate(list(ends.values())), ids)
    if len(unknown):
        raise TableError(
            f"contacts: ids {_some(unknown)} not in the particle table; "
            "filter the contacts like the particles (e.g. to the same "
            "agglomerates)"
        )
    wrong = np.zeros(len(contacts), dtype=bool)
    for end in ends.values():
        wrong |= labels.reindex(end).to_numpy() != pair_label
    if wrong.any():
        rows = contacts[wrong].head(_MAX_LISTED)
        listed = ", ".join(
            f"({a}, {b}) in {g}"
            for g, a, b in rows[["agglomerate_id", "id_a", "id_b"]].to_numpy()
        )
        raise TableError(
            "contacts: pairs whose ends are not both in the row's "
            f"agglomerate: {listed}"
        )
    a, b = ends["id_a"], ends["id_b"]
    if (a == b).any():
        raise TableError(
            "contacts: pairs of a particle with itself: ids "
            f"{_some(np.unique(a[a == b]))}"
        )
    pairs = pd.DataFrame({"a": np.minimum(a, b), "b": np.maximum(a, b)})
    repeated = pairs[pairs.duplicated()].drop_duplicates()
    if len(repeated):
        listed = _some([f"({x}, {y})" for x, y in repeated.to_numpy()])
        raise TableError(f"contacts: pairs repeat: {listed}")
    # An agglomerate is held together exactly when its members form one
    # connected group of the pairs.
    n = len(labels)
    at = _contact_rows(ids, a, b)
    graph = coo_matrix(
        (np.ones(len(a), dtype=np.int8), (at.larger, at.smaller)),
        shape=(n, n),
    )
    _, group = connected_components(graph, directed=False)
    groups = pd.Series(group).groupby(labels.to_numpy()).nunique()
    split = groups.index[groups.to_numpy() > 1]
    if len(split):
        raise TableError(
            "contacts: agglomerates not held together by their pairs: "
            f"{_some(split)}; give the whole contacts table of "
            "find_agglomerates for these particles"
        )
    return _Pairs(ids, pair_label, a, b)


def _some(values: Sequence[object] | NDArray[Any] | pd.Index) -> str:
    """Up to ten values, then how many more."""
    listed = ", ".join(map(str, values[:_MAX_LISTED]))
    more = len(values) - _MAX_LISTED
    return listed + (f" and {more} more" if more > 0 else "")


def _contact_rows(
    ids: NDArray[np.int64],
    id_a: NDArray[np.int64],
    id_b: NDArray[np.int64],
) -> _Contacts:
    """Pairs of particle ids as row positions in ``ids``.

    ``larger`` is the row of ``id_a``. Every id of the pairs must be in
    ``ids`` (``_check_contacts``, or pairs chosen for these rows).
    """
    index = pd.Index(ids)
    return _Contacts(
        index.get_indexer(pd.Index(id_a)).astype(np.intp),
        index.get_indexer(pd.Index(id_b)).astype(np.intp),
    )


def _whole(table: pd.DataFrame, column: str, name: str) -> NDArray[np.int64]:
    # Whole floats pass (3.0 after a CSV round trip); 0.5, NaN or text
    # would be cut or match nothing without a word.
    values = table[column]
    if pd.api.types.is_integer_dtype(values):
        return values.to_numpy(dtype=np.int64)
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if not (np.isfinite(numbers) & (numbers == np.round(numbers))).all():
        raise TableError(f"{name}: column {column!r} must hold integers")
    return numbers.astype(np.int64)


def select_agglomerates(
    particles: pd.DataFrame, agglomerates: pd.DataFrame, mask: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the agglomerates where ``mask`` is True, with their particles.

    Filtering the agglomerate table alone would leave the particles of
    the dropped agglomerates in the particle table, and the metrics,
    which read both tables, would mix the two (``N_ppA``, ``n_ppA``).
    A particle stays when its ``(image, agglomerate_id)`` stays.

    Parameters
    ----------
    particles : DataFrame
        Particle table with ``image`` and ``agglomerate_id``.
    agglomerates : DataFrame
        Agglomerate table with the same two columns.
    mask : Series
        True / False per agglomerate, on the agglomerate table's
        index, e.g. ``agglomerates["D_feret_max"] > 1.5``.

    Returns
    -------
    tuple
        ``(particles, agglomerates)``: copies with the kept rows; index
        and columns unchanged. The images table is not touched, so an
        image with nothing left still counts (zeros).

    Raises
    ------
    TableError
        If a column is missing, or ``mask`` is not a
        True / False Series on the agglomerate table's index.
    """
    columns = ["image", "agglomerate_id"]
    for name, table in (
        ("particles", particles),
        ("agglomerates", agglomerates),
    ):
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise TableError(f"{name}: missing columns: {missing}")
    if not isinstance(mask, pd.Series) or not mask.index.equals(
        agglomerates.index
    ):
        raise TableError(
            "mask must be a Series on the agglomerate table's index, "
            "e.g. agglomerates['D_feret_max'] > 1.5"
        )
    if mask.isna().any() or not pd.api.types.is_bool_dtype(mask):
        raise TableError("mask must hold True / False, without gaps")
    kept = agglomerates[mask.to_numpy(dtype=bool)]
    member = pd.MultiIndex.from_frame(particles[columns]).isin(
        pd.MultiIndex.from_frame(kept[columns])
    )
    return particles[member].copy(), kept.copy()
