"""Structure of agglomerates: cores, links, loops and dangling ends.

``structure_properties`` reads one image's particle, agglomerate and
contacts tables (``agglomerates.find_agglomerates``,
``properties.agglomerate_properties``) and labels every particle with
the part of its agglomerate it belongs to. The particles are the nodes
of a graph and the contacts its edges (the *possible* contacts, see
``agglpy.core.agglomerates``).

Definitions, per agglomerate:

- A contact is *tight* when its two particles share a neighbour: they
  are two sides of a triangle. A *core* is a connected group of
  ``min_core_members`` or more particles joined by tight contacts: a
  dense, triangulated part.
- A *dangling end* is a connected group of particles that hangs on the
  rest by one contact and holds no core particle: a thread, or a thread
  carrying a small loop. Found on the contacts whose removal would split
  the agglomerate: the groups left after cutting all of them form a
  tree, whose leaves are cut off again and again unless they hold a
  core particle (or, in an agglomerate without a core, 3 or more
  particles, i.e. a loop).
- The rest, split into connected runs: a run touching two or more
  cores is a *link*, a run touching one core a *loop*. In an
  agglomerate without a core the rest is its *small core*.
- An agglomerate with nothing left (a tree, a single particle) is
  ``no core`` throughout.

Particle columns: ``segment_type`` (one of the labels above) and
``segment_id`` (each core, link, loop, dangling end, small core and
no-core agglomerate is one segment; numbered from 0 within the image by
each segment's smallest particle id). Agglomerate columns:
``core_count``, ``core_member_count``, ``small_core_member_count``,
``link_count``, ``dangling_end_count``, ``dangling_end_member_count``
(0 when there is none).

Kind of number in a 2D projection: ``core_member_count`` and
``small_core_member_count`` are upper bounds (overlapping circles need
not touch, so a projection reads denser than the agglomerate is);
``dangling_end_member_count`` is a lower bound where the agglomerate
truly has a core; the other counts and every single label describe one
image and can change with one extra contact: compare totals, not
single agglomerates.

The result does not depend on the row order or on the particle ids,
except the numbering of ``segment_id``.
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

PARTICLE_COLUMNS = ("segment_type", "segment_id")
AGGLOMERATE_COLUMNS = (
    "core_count",
    "core_member_count",
    "small_core_member_count",
    "link_count",
    "dangling_end_count",
    "dangling_end_member_count",
)


def structure_properties(
    particles: pd.DataFrame,
    agglomerates: pd.DataFrame,
    contacts: pd.DataFrame,
    *,
    min_core_members: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label the segments of every agglomerate of one image.

    Args:
        particles: One image's particle table with ``id``, ``r`` and
            ``agglomerate_id``.
        agglomerates: Its agglomerate table: one row per
            ``agglomerate_id`` of ``particles``, no other.
        contacts: Its contacts table (``find_agglomerates``).
        min_core_members: Particles a group of triangles needs to be a
            core; an integer of 3 or more.

    Returns:
        Copies of ``particles`` and ``agglomerates`` with the structure
        columns (module docstring), replacing any from an earlier run;
        every other column, the row order and the index are kept.

    Raises:
        ParamsError: If ``min_core_members`` is not an integer >= 3.
        TableError: If a column is missing or invalid, or the three
            tables do not belong together.
    """
    m = _min_core_members(min_core_members)
    _require(particles, ("id", "r", "agglomerate_id"))
    pairs = _check_contacts(particles, contacts)
    _radii(particles)
    labels = _whole(particles, "agglomerate_id", "particles")
    groups, agg = np.unique(labels, return_inverse=True)
    rows = _agglomerate_rows(agglomerates, groups)
    at = _contact_rows(pairs.ids, pairs.id_a, pairs.id_b)
    g = _graph.graph(len(pairs.ids), at.larger, at.smaller)
    kind, segment = _segments(g, pairs.ids, agg.astype(np.intp), m)

    p = particles.copy()
    p["segment_type"] = np.array(SEGMENT_TYPES, dtype=object)[kind]
    p["segment_id"] = segment
    a = agglomerates.copy()
    for column, values in _counts(kind, segment, agg, len(groups)).items():
        a[column] = values[rows]
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
