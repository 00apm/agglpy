"""Graph algorithms on contact pairs, for ``agglpy.core.structure``.

A graph is ``n`` nodes (row positions) and undirected pairs ``a[k]`` -
``b[k]``, each pair once, no pair of a node with itself. Every function
works on all components at once, so one call serves a whole image.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


class Graph(NamedTuple):
    """Nodes ``0 … n-1`` and their pairs, with the symmetric matrix.

    ``matrix[i, j]`` is 1.0 when ``i`` and ``j`` are a pair.
    """

    n: int
    a: NDArray[np.intp]
    b: NDArray[np.intp]
    matrix: csr_matrix


def graph(n: int, a: NDArray[np.intp], b: NDArray[np.intp]) -> Graph:
    """The graph of ``n`` nodes and the pairs ``a[k]`` - ``b[k]``."""
    a = np.asarray(a, dtype=np.intp)
    b = np.asarray(b, dtype=np.intp)
    return Graph(n, a, b, _symmetric(n, a, b, np.ones(len(a))))


def tight(g: Graph) -> NDArray[np.bool_]:
    """Per pair: whether its two ends share a neighbour (a triangle)."""
    if len(g.a) == 0:
        return np.zeros(0, dtype=bool)
    # (A @ A)[i, j] counts the common neighbours of i and j.
    common = (g.matrix @ g.matrix).tocsr()[g.a, g.b]
    return np.asarray(common).ravel() > 0


def bridges(g: Graph) -> NDArray[np.bool_]:
    """Per pair: whether removing it splits its component.

    One depth-first pass with low links (Tarjan), iterative, so a long
    thread does not hit Python's recursion limit.
    """
    k = len(g.a)
    out = np.zeros(k, dtype=bool)
    if k == 0:
        return out
    # Each matrix entry carries the index of its pair, so the pass can
    # skip the pair it arrived by.
    pairs = _symmetric(g.n, g.a, g.b, np.arange(1, k + 1, dtype=np.int64))
    indptr = pairs.indptr.tolist()
    indices = pairs.indices.tolist()
    pair_of = (pairs.data - 1).tolist()
    order = [-1] * g.n
    low = [0] * g.n
    time = 0
    for root in range(g.n):
        if order[root] >= 0 or indptr[root] == indptr[root + 1]:
            continue
        order[root] = low[root] = time
        time += 1
        # (node, pair it arrived by, next entry to look at)
        stack = [(root, -1, indptr[root])]
        while stack:
            v, arrived, i = stack[-1]
            if i < indptr[v + 1]:
                stack[-1] = (v, arrived, i + 1)
                w, e = indices[i], pair_of[i]
                if e == arrived:
                    continue
                if order[w] < 0:
                    order[w] = low[w] = time
                    time += 1
                    stack.append((w, e, indptr[w]))
                elif order[w] < low[v]:
                    low[v] = order[w]
                continue
            stack.pop()
            if stack:
                u = stack[-1][0]
                if low[v] < low[u]:
                    low[u] = low[v]
                if low[v] > order[u]:
                    out[arrived] = True
    return out


def components(
    n: int, a: NDArray[np.intp], b: NDArray[np.intp]
) -> NDArray[np.intp]:
    """Component label of each node, for the pairs ``a[k]`` - ``b[k]``."""
    if n == 0:
        return np.zeros(0, dtype=np.intp)
    matrix = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    _, labels = connected_components(matrix, directed=False)
    return np.asarray(labels, dtype=np.intp)


def distances(
    matrix: csr_matrix, sources: NDArray[np.intp], *, unweighted: bool
) -> NDArray[np.float64]:
    """Per node: its distance from the nearest of ``sources``.

    With one source per component, every node gets the distance from
    its own component's source, so one search serves all components.
    ``inf`` where no source is reachable.
    """
    if len(sources) == 0:
        return np.full(matrix.shape[0], np.inf)
    # The matrices here are symmetric: searching them as directed
    # graphs gives the same distances without a transposed copy.
    out = dijkstra(
        matrix,
        directed=True,
        indices=sources,
        unweighted=unweighted,
        min_only=True,
    )
    return np.asarray(out, dtype=np.float64)


def weighted(g: Graph, r: NDArray[np.float64]) -> csr_matrix:
    """The matrix of ``g`` with weight ``r_a + r_b`` on each pair.

    A path's weight is then the sum of ``2 r`` over its nodes, minus
    the radii of its two ends.
    """
    return _symmetric(g.n, g.a, g.b, r[g.a] + r[g.b])


# Groups up to this size get their diameter from a search from every
# member (batched over all such groups); larger ones from eccentricity
# bounds, which need far fewer searches.
_EVERY_PAIR_SIZE = 16


def diameters(
    g: Graph,
    codes: NDArray[np.intp],
    k: int,
    *,
    r: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Per group: the largest distance between two of its members.

    ``codes[i]`` is node ``i``'s group, ``0 … k-1``; each group must be
    connected. With ``r=None`` distances are hops (a lone node: 0);
    else the smallest sum of ``2 r`` over the nodes of a path, both ends
    included (a lone node: its ``2 r``). Exact at any size.
    """
    size = np.bincount(codes, minlength=k)
    out = np.zeros(k)
    if r is None:
        matrix, radius = g.matrix, np.zeros(g.n)
    else:
        matrix, radius = weighted(g, r), r
        lone = size[codes] == 1
        out[codes[lone]] = 2 * r[lone]
    unweighted = r is None
    order = np.argsort(codes, kind="stable")
    start = np.cumsum(size) - size
    rank = np.empty(g.n, dtype=np.intp)
    rank[order] = np.arange(g.n) - start[codes[order]]

    small = (size >= 2) & (size <= _EVERY_PAIR_SIZE)
    in_small = small[codes]
    for i in range(int(size[small].max()) if small.any() else 0):
        # Round i: the i-th member of every small group at once.
        sources = np.flatnonzero(in_small & (rank == i))
        d = distances(matrix, sources, unweighted=unweighted)
        searched = np.zeros(k, dtype=bool)
        searched[codes[sources]] = True
        source_r = np.zeros(k)
        source_r[codes[sources]] = radius[sources]
        take = searched[codes]
        # A member's distance to itself (0 + 2 r) never beats a pair.
        value = d[take] + radius[take] + source_r[codes[take]]
        np.maximum.at(out, codes[take], value)

    for c in np.flatnonzero((size >= 2) & (size > _EVERY_PAIR_SIZE)):
        members = order[start[c] : start[c] + size[c]]
        out[c] = _bounded_diameter(
            matrix[members][:, members], radius[members], unweighted
        )
    return out


def _bounded_diameter(
    matrix: csr_matrix, r: NDArray[np.float64], unweighted: bool
) -> float:
    """Exact diameter of one connected group, by eccentricity bounds.

    BoundingDiameters (Takes and Kosters, 2011). The distance is
    ``dist(u, v) + r_u + r_v`` for ``u != v`` (``r = 0`` for hops),
    which keeps the triangle inequality, so the bounds hold: for a
    searched ``v`` and any ``w``, ``ecc(w) >= max(d(v, w), ecc(v) -
    d(v, w))`` and ``ecc(w) <= ecc(v) + d(v, w)``.
    """
    n = matrix.shape[0]
    low = np.zeros(n)
    high = np.full(n, np.inf)
    open_ = np.ones(n, dtype=bool)
    best = 0.0
    pick_high = True
    while open_.any():
        # Alternate the largest upper and the smallest lower bound.
        candidates = np.flatnonzero(open_)
        bound = high if pick_high else -low
        v = candidates[np.argmax(bound[candidates])]
        pick_high = not pick_high
        d = dijkstra(matrix, directed=True, indices=v, unweighted=unweighted)
        d = np.asarray(d, dtype=np.float64) + r + r[v]
        d[v] = 0.0
        ecc = float(d.max())
        low = np.maximum(low, np.maximum(d, ecc - d))
        high = np.minimum(high, ecc + d)
        # Each lower bound is at most some eccentricity, so best stays
        # a lower bound of the diameter; a node whose upper bound is
        # not above it cannot raise it.
        best = max(best, float(low.max()))
        open_ &= high > best
        open_[v] = False
    return best


# Two path lengths are equal when they differ by less than this
# fraction: sums of floats in another order differ in the last bits.
_TIE = 1e-12


def equal(x: NDArray[np.float64] | float, y: float) -> NDArray[np.bool_]:
    """Whether ``x`` equals ``y`` up to ``_TIE`` (relative)."""
    return np.asarray(np.abs(x - y) <= _TIE * max(abs(y), 1.0))


def path(
    matrix: csr_matrix,
    start: int,
    end: int,
    ids: NDArray[np.int64],
    *,
    unweighted: bool,
) -> list[int]:
    """A shortest path from ``start`` to ``end``, as row positions.

    Each step goes to the neighbour with the lowest ``ids`` value among
    those that stay on a shortest path.
    """
    far = np.asarray(
        dijkstra(matrix, directed=True, indices=end, unweighted=unweighted),
        dtype=np.float64,
    )
    out = [start]
    v = start
    while v != end:
        at = slice(matrix.indptr[v], matrix.indptr[v + 1])
        neighbours = matrix.indices[at]
        step = 1.0 if unweighted else matrix.data[at]
        on = neighbours[equal(step + far[neighbours], far[v])]
        v = int(on[np.argmin(ids[on])])
        out.append(v)
    return out


def _symmetric(
    n: int,
    a: NDArray[np.intp],
    b: NDArray[np.intp],
    data: NDArray[np.generic],
) -> csr_matrix:
    return coo_matrix(
        (np.concatenate([data, data]), (np.r_[a, b], np.r_[b, a])),
        shape=(n, n),
    ).tocsr()
