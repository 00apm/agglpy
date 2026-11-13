"""Structure of agglomerates: segments, branches and longest chains.

``structure_properties`` adds the structure columns to one image's
particle and agglomerate tables (``docs/agglomerate-structure.md``).
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.core import _graph
from agglpy.core.agglomerates import (
    _check_contacts,
    _contact_rows,
    _some,
    _whole,
)
from agglpy.core.properties import _radii, _require
from agglpy.errors import ParamsError, TableError
from agglpy.params import _integer

__all__ = [
    "AGGLOMERATE_COLUMNS",
    "PARTICLE_COLUMNS",
    "SEGMENT_TYPES",
    "structure_properties",
]

SEGMENT_TYPES = (
    "core",
    "small core",
    "link",
    "loop",
    "dangling end",
    "no core",
)
_CORE, _SMALL, _LINK, _LOOP, _DANGLING, _NO_CORE = range(len(SEGMENT_TYPES))

PARTICLE_COLUMNS = (
    "segment_type",
    "segment_id",
    "branch_depth",
    "branch_length",
    "tip",
    "dangling_end_depth",
    "dangling_end_length",
)
AGGLOMERATE_COLUMNS = (
    "longest_chain",
    "longest_chain_length",
    "branch_count",
    "branch_depth_max",
    "branch_length_max",
    "core_count",
    "core_member_count",
    "small_core_member_count",
    "link_count",
    "dangling_end_count",
    "dangling_end_member_count",
    "dangling_end_depth_max",
    "dangling_end_length_max",
)


def structure_properties(
    particles: pd.DataFrame,
    agglomerates: pd.DataFrame,
    contacts: pd.DataFrame,
    *,
    min_core_members: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Segments, branches and longest chains of one image's agglomerates.

    The contacts are the edges of a graph; the base of an agglomerate is
    its largest member (equal radii: the lower id). Lengths are sums of
    diameters, in the units of ``r``.

    Args:
        particles: One image's particle table with ``id``, ``r`` and
            ``agglomerate_id``.
        agglomerates: Its agglomerate table: one row per
            ``agglomerate_id`` of ``particles``, no other.
        contacts: Its contacts table (``find_agglomerates``).
        min_core_members: Particles a group of triangles needs to be a
            core; an integer of 3 or more.

    Returns:
        Copies of ``particles`` and ``agglomerates`` with the columns
        ``PARTICLE_COLUMNS`` and ``AGGLOMERATE_COLUMNS``, replacing any
        from an earlier run; other columns, row order and index kept.

    Raises:
        ParamsError: If ``min_core_members`` is not an integer >= 3.
        TableError: If a column is missing or invalid, or the three
            tables do not belong together.
    """
    m = _min_core_members(min_core_members)
    _require(particles, ("id", "r", "agglomerate_id"))
    pairs = _check_contacts(particles, contacts)
    r = _radii(particles)
    labels = _whole(particles, "agglomerate_id", "particles")
    groups, inverse = np.unique(labels, return_inverse=True)
    agg = inverse.reshape(-1).astype(np.intp)
    count = len(groups)
    rows = _agglomerate_rows(agglomerates, groups)
    at = _contact_rows(pairs.ids, pairs.id_a, pairs.id_b)
    g = _graph.graph(len(pairs.ids), at.larger, at.smaller)
    kind, segment = _segments(g, pairs.ids, agg, m)

    columns: dict[str, NDArray[np.generic]] = {
        "segment_type": np.array(SEGMENT_TYPES, dtype=object)[kind],
        "segment_id": segment,
        **_branches(g, pairs.ids, r, agg, count),
        **_dangling(g, r, kind, segment),
    }
    found = {
        "longest_chain": _graph.diameters(g, agg, count, r=None).astype(
            np.int64
        ),
        "longest_chain_length": _graph.diameters(g, agg, count, r=r),
        **_counts(kind, segment, agg, count),
    }
    size = np.bincount(agg, minlength=count)
    found["branch_count"] = np.bincount(
        agg, weights=columns["tip"], minlength=count
    ).astype(np.int64)
    several = size[agg] >= 2
    dangling = kind == _DANGLING
    for stem, rows_used in (("branch", several), ("dangling_end", dangling)):
        for measure in ("depth", "length"):
            values = columns[f"{stem}_{measure}"]
            found[f"{stem}_{measure}_max"] = _group_max(
                values, agg, count, rows_used
            )

    p = particles.copy()
    for column in PARTICLE_COLUMNS:
        p[column] = columns[column]
    a = agglomerates.copy()
    for column in AGGLOMERATE_COLUMNS:
        a[column] = found[column][rows]
    return p, a


def _min_core_members(value: object) -> int:
    m = _integer("min_core_members", value)
    if m < 3:
        raise ParamsError(
            f"min_core_members must be 3 or more (a triangle), got {value!r}"
        )
    return m


def _agglomerate_rows(
    agglomerates: pd.DataFrame, groups: NDArray[np.int64]
) -> NDArray[np.intp]:
    """Position in ``groups`` of each row's ``agglomerate_id``."""
    if "agglomerate_id" not in agglomerates.columns:
        raise TableError("agglomerates: missing columns ['agglomerate_id']")
    ids = _whole(agglomerates, "agglomerate_id", "agglomerates")
    repeats = pd.Index(ids).duplicated()
    if repeats.any():
        raise TableError(
            f"agglomerates: ids {_some(np.unique(ids[repeats]))} repeat; "
            "one row per agglomerate"
        )
    extra = np.setdiff1d(ids, groups)
    if len(extra):
        raise TableError(
            f"agglomerates: ids {_some(extra)} have no particles; give "
            "the agglomerate table of these particles"
        )
    missing = np.setdiff1d(groups, ids)
    if len(missing):
        raise TableError(
            f"agglomerates: no row for agglomerates {_some(missing)}; "
            "give the agglomerate table of these particles"
        )
    return np.searchsorted(groups, ids).astype(np.intp)


def _segments(
    g: _graph.Graph,
    ids: NDArray[np.int64],
    agg: NDArray[np.intp],
    m: int,
) -> tuple[NDArray[np.intp], NDArray[np.int64]]:
    """Segment type (index into SEGMENT_TYPES) and segment id per node."""
    n = g.n
    count = int(agg.max()) + 1 if n else 0

    tight = _graph.tight(g)
    core_group = _graph.components(n, g.a[tight], g.b[tight])
    is_core = np.bincount(core_group, minlength=n)[core_group] >= m
    has_core = np.bincount(agg, weights=is_core, minlength=count) > 0

    hang = _hanging(g, agg, is_core, has_core)
    no_core = (np.bincount(agg, weights=~hang, minlength=count) == 0)[agg]
    rest = ~hang & ~is_core

    kind = np.full(n, _SMALL, dtype=np.intp)
    kind[is_core] = _CORE
    kind[hang] = _DANGLING
    kind[no_core] = _NO_CORE
    run = _graph.components(n, *_within(g, rest))
    between = rest & has_core[agg]
    touched = _cores_touched(g, run, between, is_core, core_group)
    kind[between] = np.where(touched[run[between]] >= 2, _LINK, _LOOP)

    # One label per segment: cores by their triangle group, links,
    # loops and small cores by their run, dangling ends by their
    # connected group, no-core agglomerates whole.
    dangling = _graph.components(n, *_within(g, kind == _DANGLING))
    label = np.where(is_core, core_group, run)
    label = np.where(kind == _DANGLING, dangling, label)
    label = np.where(kind == _NO_CORE, agg, label)
    family = np.select(
        [is_core, kind == _DANGLING, kind == _NO_CORE], [0, 2, 3], 1
    )
    key = family.astype(np.int64) * max(n, 1) + label
    first = pd.Series(ids).groupby(key).transform("min").to_numpy()
    _, segment = np.unique(first, return_inverse=True)
    return kind, segment.astype(np.int64)


def _hanging(
    g: _graph.Graph,
    agg: NDArray[np.intp],
    is_core: NDArray[np.bool_],
    has_core: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Per node: whether it lies in a dangling end (or a tree)."""
    bridge = _graph.bridges(g)
    piece = _graph.components(g.n, g.a[~bridge], g.b[~bridge])
    pieces = int(piece.max()) + 1 if g.n else 0
    piece_agg = np.zeros(pieces, dtype=np.intp)
    piece_agg[piece] = agg
    holds_core = np.bincount(piece, weights=is_core, minlength=pieces) > 0
    size = np.bincount(piece, minlength=pieces)
    anchor = holds_core | (~has_core[piece_agg] & (size >= 3))
    # The pieces and the bridges form a tree per agglomerate; cut its
    # leaves that are not anchors until none is left.
    ea, eb = piece[g.a[bridge]], piece[g.b[bridge]]
    removed = np.zeros(pieces, dtype=bool)
    while True:
        alive = ~removed[ea] & ~removed[eb]
        degree = np.bincount(ea[alive], minlength=pieces) + np.bincount(
            eb[alive], minlength=pieces
        )
        leaf = ~removed & ~anchor & (degree <= 1)
        if not leaf.any():
            return removed[piece]
        removed |= leaf


def _within(
    g: _graph.Graph, mask: NDArray[np.bool_]
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The pairs with both ends in ``mask``."""
    both = mask[g.a] & mask[g.b]
    return g.a[both], g.b[both]


def _cores_touched(
    g: _graph.Graph,
    run: NDArray[np.intp],
    between: NDArray[np.bool_],
    is_core: NDArray[np.bool_],
    core_group: NDArray[np.intp],
) -> NDArray[np.int64]:
    """Per run label: how many different cores its particles touch."""
    ends = []
    for x, y in ((g.a, g.b), (g.b, g.a)):
        touch = between[x] & is_core[y]
        ends.append(np.stack([run[x[touch]], core_group[y[touch]]], axis=1))
    touched = np.unique(np.concatenate(ends), axis=0)
    return np.bincount(touched[:, 0], minlength=g.n)


def _branches(
    g: _graph.Graph,
    ids: NDArray[np.int64],
    r: NDArray[np.float64],
    agg: NDArray[np.intp],
    count: int,
) -> dict[str, NDArray[np.generic]]:
    """Branch depth, length and tips, from each agglomerate's base."""
    # The base: the largest member, equal radii the lower id.
    order = np.lexsort((ids, -r, agg))
    _, first = np.unique(agg[order], return_index=True)
    bases = order[first]
    depth = _graph.distances(g.matrix, bases, unweighted=True)
    dist = _graph.distances(_graph.weighted(g, r), bases, unweighted=False)
    # A path's weight counts each end's radius once: add the far end's,
    # take away the base's.
    length = dist + r - r[bases][agg]
    deeper = np.zeros(g.n, dtype=bool)
    deeper[g.a[depth[g.b] > depth[g.a]]] = True
    deeper[g.b[depth[g.a] > depth[g.b]]] = True
    tip = ~deeper
    tip[bases] = False
    return {
        "branch_depth": depth.astype(np.int64),
        "branch_length": length,
        "tip": tip,
    }


def _dangling(
    g: _graph.Graph,
    r: NDArray[np.float64],
    kind: NDArray[np.intp],
    segment: NDArray[np.int64],
) -> dict[str, NDArray[np.generic]]:
    """Depth and length inside each dangling end; NaN off the ends."""
    end = kind == _DANGLING
    depth = np.full(g.n, np.nan)
    length = np.full(g.n, np.nan)
    if end.any():
        # Each end hangs by one contact; its particle there is the root.
        one = end[g.a] != end[g.b]
        roots = np.unique(np.where(end[g.a[one]], g.a[one], g.b[one]))
        inside = _graph.graph(g.n, *_within(g, end))
        hops = _graph.distances(inside.matrix, roots, unweighted=True)
        dist = _graph.distances(
            _graph.weighted(inside, r), roots, unweighted=False
        )
        root_r = pd.Series(r[roots], index=segment[roots])
        r_root = root_r.reindex(segment[end]).to_numpy(dtype=np.float64)
        depth[end] = hops[end] + 1
        length[end] = dist[end] + r[end] + r_root
    return {"dangling_end_depth": depth, "dangling_end_length": length}


def _group_max(
    values: NDArray[np.generic],
    agg: NDArray[np.intp],
    count: int,
    used: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Per agglomerate: the largest of ``values[used]``; NaN for none."""
    out = np.full(count, -np.inf)
    np.maximum.at(out, agg[used], values[used].astype(np.float64))
    out[np.isneginf(out)] = np.nan
    return out


def _counts(
    kind: NDArray[np.intp],
    segment: NDArray[np.int64],
    agg: NDArray[np.intp],
    count: int,
) -> dict[str, NDArray[np.int64]]:
    """The count columns, one value per agglomerate (``agg`` code)."""
    types = len(SEGMENT_TYPES)
    members = np.bincount(agg * types + kind, minlength=count * types).reshape(
        count, types
    )
    _, first = np.unique(segment, return_index=True)
    segments = np.bincount(
        agg[first] * types + kind[first], minlength=count * types
    ).reshape(count, types)
    out = {
        "core_count": segments[:, _CORE],
        "core_member_count": members[:, _CORE],
        "small_core_member_count": members[:, _SMALL],
        "link_count": segments[:, _LINK],
        "dangling_end_count": segments[:, _DANGLING],
        "dangling_end_member_count": members[:, _DANGLING],
    }
    return {k: v.astype(np.int64) for k, v in out.items()}
