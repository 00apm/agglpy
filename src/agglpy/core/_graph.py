"""Graph algorithms on contact pairs, for ``agglpy.core.structure``.

A graph is ``n`` nodes (row positions) and undirected pairs ``a[k]`` -
``b[k]``, each pair once, no pair of a node with itself. Every function
works on all components at once, so one call serves a whole image.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


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
