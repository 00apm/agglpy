"""Structure of agglomerates (``agglpy.core.structure``).

The catalogue (``support.structure_cases``) holds circle sets with
their segments read by eye; the hand graphs below give the contact
graph directly, as edge lists of particle ids.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from agglpy.core import _graph as graph_module
from agglpy.core.structure import (
    AGGLOMERATE_COLUMNS,
    PARTICLE_COLUMNS,
    structure_properties,
)
from agglpy.errors import ParamsError, TableError

from support.structure_cases import (
    CASES,
    CONTACTS,
    expected_segments,
    named_contacts,
    run_case,
)

COUNTS = [
    "core_count",
    "core_member_count",
    "small_core_member_count",
    "link_count",
    "dangling_end_count",
    "dangling_end_member_count",
]


PARTICLE_DTYPES = {
    "segment_type": object,
    "segment_id": np.int64,
    "branch_depth": np.int64,
    "branch_length": np.float64,
    "tip": bool,
    "dangling_end_depth": np.float64,
    "dangling_end_length": np.float64,
}
AGGLOMERATE_DTYPES = {
    "longest_chain": np.int64,
    "longest_chain_length": np.float64,
    "branch_count": np.int64,
    "branch_depth_max": np.float64,
    "branch_length_max": np.float64,
    **dict.fromkeys(COUNTS, np.int64),
    "dangling_end_depth_max": np.float64,
    "dangling_end_length_max": np.float64,
}
BRANCHES = [
    "branch_count",
    "branch_depth_max",
    "branch_length_max",
    "longest_chain",
    "longest_chain_length",
]


def _assert_columns(p, a):
    """Every structure column, in order, with its dtype."""
    assert PARTICLE_COLUMNS == tuple(PARTICLE_DTYPES)
    assert AGGLOMERATE_COLUMNS == tuple(AGGLOMERATE_DTYPES)
    assert list(p.columns[-len(PARTICLE_COLUMNS) :]) == list(PARTICLE_COLUMNS)
    for column, dtype in PARTICLE_DTYPES.items():
        assert p[column].dtype == dtype, column
    for column, dtype in AGGLOMERATE_DTYPES.items():
        assert a[column].dtype == dtype, column


def _graph(edges, r, aggl=None):
    """Tables of a contact graph: ids 0 … n-1, the edges as contacts."""
    n = len(r)
    labels = [0] * n if aggl is None else list(aggl)
    particles = pd.DataFrame(
        {
            "id": np.arange(n, dtype=np.int64),
            "r": np.asarray(r, dtype=np.float64),
            "agglomerate_id": np.asarray(labels, dtype=np.int64),
        }
    )
    a = np.array([e[0] for e in edges], dtype=np.int64)
    b = np.array([e[1] for e in edges], dtype=np.int64)
    contacts = pd.DataFrame(
        {
            "agglomerate_id": particles["agglomerate_id"].to_numpy()[a],
            "id_a": a,
            "id_b": b,
            "inside": np.zeros(len(a), dtype=bool),
        }
    )
    return particles, _agglomerates(particles), contacts


def _agglomerates(particles):
    ids = np.unique(particles["agglomerate_id"].to_numpy(dtype=np.int64))
    return pd.DataFrame({"agglomerate_id": ids})


def _wheel(centre, ring):
    """Edges of a centre joined to every ring particle, the ring cyclic."""
    ring = list(ring)
    spokes = [(centre, i) for i in ring]
    rim = [(ring[k], ring[(k + 1) % len(ring)]) for k in range(len(ring))]
    return spokes + rim


# A core of 7 with two dangling ends: 7-8 on particle 1, 9 on 3.
G4 = ([*_wheel(0, range(1, 7)), (1, 7), (7, 8), (3, 9)], [2.0] + [1.0] * 9)
# Two cores of 7, a link 14-15 between them, a loop 16-17 on the first.
G5 = (
    _wheel(0, range(1, 7))
    + _wheel(7, range(8, 14))
    + [(3, 14), (14, 15), (15, 10)]
    + [(1, 16), (16, 17), (17, 2)],
    [2.0] + [1.0] * 17,
)


@pytest.mark.parametrize("name", list(CASES))
def test_catalogue_contacts(name):
    names, _, contacts = run_case(CASES[name][0])
    assert named_contacts(names, contacts) == CONTACTS[name]


@pytest.mark.parametrize("name", list(CASES))
def test_catalogue_segments(name):
    circles, by_eye = CASES[name]
    names, particles, contacts = run_case(circles)
    p, _ = structure_properties(particles, _agglomerates(particles), contacts)
    got = dict(zip(names, p["segment_type"], strict=True))
    assert got == expected_segments(names, by_eye)
    _assert_columns(p, _)


def test_core_with_dangling_ends():
    p, a = structure_properties(*_graph(*G4))
    assert p["segment_type"].tolist() == ["core"] * 7 + ["dangling end"] * 3
    assert p["segment_id"].tolist() == [0] * 7 + [1, 1, 2]
    assert a.iloc[0][COUNTS].tolist() == [1, 7, 0, 0, 2, 3]


def test_link_and_loop():
    p, a = structure_properties(*_graph(*G5))
    assert p["segment_type"].tolist() == (
        ["core"] * 14 + ["link"] * 2 + ["loop"] * 2
    )
    assert p["segment_id"].tolist() == [0] * 7 + [1] * 7 + [2, 2, 3, 3]
    assert a.iloc[0][COUNTS].tolist() == [2, 14, 0, 1, 0, 0]


def test_two_cores_touching_have_no_link():
    edges = _wheel(0, range(1, 7)) + _wheel(7, range(8, 14)) + [(3, 10)]
    p, a = structure_properties(*_graph(edges, [1.0] * 14))
    assert p["segment_type"].tolist() == ["core"] * 14
    assert p["segment_id"].tolist() == [0] * 7 + [1] * 7
    assert a.iloc[0][COUNTS].tolist() == [2, 14, 0, 0, 0, 0]


@pytest.mark.parametrize(
    ("edges", "r", "m", "expected"),
    [
        # the cliff: a wheel of 7 is a core at 7, a small core at 8
        (*G4, 7, ["core"] * 7 + ["dangling end"] * 3),
        (*G4, 8, ["small core"] * 7 + ["dangling end"] * 3),
        # a wheel of 6 is a small core at 7
        (_wheel(0, range(1, 6)), [1.0] * 6, 7, ["small core"] * 6),
        # two triangles joined by a chain: one small core of 8
        (
            [
                (0, 1),
                (1, 2),
                (0, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (5, 7),
            ],
            [1.0] * 8,
            7,
            ["small core"] * 8,
        ),
        (
            [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)],
            [1.0] * 5,
            7,
            ["small core"] * 3 + ["dangling end"] * 2,
        ),
        # a ring alone
        (
            [(i, (i + 1) % 6) for i in range(6)],
            [1.0] * 6,
            7,
            ["small core"] * 6,
        ),
        # a tree, a pair, a single particle
        (
            [(0, 1), (1, 2), (2, 3), (1, 4)],
            [3.0, 1, 1, 1, 1],
            7,
            ["no core"] * 5,
        ),
        ([(0, 1)], [2.0, 1.0], 7, ["no core"] * 2),
        ([], [1.5], 7, ["no core"]),
    ],
)
def test_segment_types(edges, r, m, expected):
    p, a = structure_properties(*_graph(edges, r), min_core_members=m)
    assert p["segment_type"].tolist() == expected
    # a small core and a no-core agglomerate are one segment each
    for label in ("small core", "no core"):
        ids = p.loc[p["segment_type"] == label, "segment_id"]
        assert ids.nunique() <= 1
    small = expected.count("small core")
    assert a.iloc[0]["small_core_member_count"] == small
    assert a.iloc[0]["core_count"] == (1 if "core" in expected else 0)


def test_several_agglomerates_numbered_per_image():
    edges, r = G4
    edges = [*edges, (10, 11)]
    r = [*r, 2.0, 1.0, 1.5]
    aggl = [0] * 10 + [1, 1, 2]
    particles, agglomerates, contacts = _graph(edges, r, aggl)
    agglomerates = agglomerates.iloc[::-1].reset_index(drop=True)
    p, a = structure_properties(particles, agglomerates, contacts)
    assert p["segment_id"].tolist() == [0] * 7 + [1, 1, 2, 3, 3, 4]
    assert a["agglomerate_id"].tolist() == [2, 1, 0]
    assert a["core_count"].tolist() == [0, 0, 1]
    assert a["dangling_end_count"].tolist() == [0, 0, 2]
    assert a["dangling_end_member_count"].tolist() == [0, 0, 3]


@pytest.mark.parametrize("m", [2, 2.0, 7.0, True, "7", None])
def test_min_core_members_checked(m):
    with pytest.raises(ParamsError, match="min_core_members"):
        structure_properties(*_graph(*G4), min_core_members=m)


def test_min_core_members_accepts_numpy_integers():
    p, _ = structure_properties(*_graph(*G4), min_core_members=np.int64(7))
    assert p["segment_type"].tolist()[0] == "core"


def test_empty_tables_give_every_column():
    particles, agglomerates, contacts = _graph([], [])
    p, a = structure_properties(particles, agglomerates, contacts)
    assert len(p) == 0
    assert len(a) == 0
    _assert_columns(p, a)


@pytest.mark.parametrize(
    ("tables", "message"),
    [
        (
            lambda p, a, c: (p, pd.DataFrame({"agglomerate_id": [0, 5]}), c),
            "agglomerates: .*5",
        ),
        (
            lambda p, a, c: (p, a.iloc[:0], c),
            "agglomerates: .*0",
        ),
        (
            lambda p, a, c: (p, pd.DataFrame({"agglomerate_id": [0, 0]}), c),
            "repeat",
        ),
        (
            lambda p, a, c: (p, a.drop(columns="agglomerate_id"), c),
            "agglomerate_id",
        ),
        (
            lambda p, a, c: (p.iloc[:-1], a, c),
            "not in the particle table",
        ),
    ],
)
def test_mismatched_tables_refused(tables, message):
    with pytest.raises(TableError, match=message):
        structure_properties(*tables(*_graph(*G4)))


def test_earlier_columns_replaced_and_input_untouched():
    particles, agglomerates, contacts = _graph(*G5)
    before = [t.copy() for t in (particles, agglomerates, contacts)]
    p, a = structure_properties(particles, agglomerates, contacts)
    p2, a2 = structure_properties(p, a, contacts)
    pd.testing.assert_frame_equal(p2, p)
    pd.testing.assert_frame_equal(a2, a)
    for table, copy in zip(
        (particles, agglomerates, contacts), before, strict=True
    ):
        pd.testing.assert_frame_equal(table, copy)


def test_rows_index_and_other_columns_kept():
    particles, agglomerates, contacts = _graph(*G4)
    particles = particles.assign(note="x").iloc[::-1]
    particles.index = range(100, 100 + len(particles))
    p, _ = structure_properties(particles, agglomerates, contacts)
    assert p.index.tolist() == particles.index.tolist()
    assert p["id"].tolist() == particles["id"].tolist()
    assert (p["note"] == "x").all()
    assert p["segment_type"].tolist()[:3] == ["dangling end"] * 3


# A tree, base 0.
G1 = ([(0, 1), (1, 2), (2, 3), (1, 4)], [3.0, 1, 1, 1, 1])
# A ring of 5 with a large particle (1) off the hop path to 4.
G2 = ([(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)], [10.0, 8, 1, 1, 1])


def test_branches_on_a_tree():
    p, a = structure_properties(*_graph(*G1))
    assert p["branch_depth"].tolist() == [0, 1, 2, 3, 2]
    assert p["branch_length"].tolist() == [0, 2, 4, 6, 4]
    assert p["tip"].tolist() == [False, False, False, True, True]
    assert a.iloc[0][BRANCHES].tolist() == [2, 3, 6, 3, 12]
    assert np.isnan(a["dangling_end_depth_max"][0])
    assert a["dangling_end_count"][0] == 0


def test_length_takes_another_path_than_hops():
    p, a = structure_properties(*_graph(*G2))
    assert p["branch_depth"].tolist() == [0, 1, 1, 2, 2]
    # 4 is reached by 2, 3 (6), not by the large 1 (18)
    assert p["branch_length"].tolist() == [0, 16, 2, 4, 6]
    assert p["tip"].tolist() == [False, False, False, True, True]
    # the farthest particle by length (1) is not a tip
    assert (a["branch_count"][0], a["branch_length_max"][0]) == (2, 16)
    # hops: pairs 2 apart; length: the pair 0-1 (20 + 16), one contact
    assert (a["longest_chain"][0], a["longest_chain_length"][0]) == (2, 36)


def test_dangling_end_depth_and_length():
    p, a = structure_properties(*_graph(*G4))
    assert p["dangling_end_depth"].tolist()[7:] == [1, 2, 1]
    assert p["dangling_end_length"].tolist()[7:] == [2, 4, 2]
    assert p["dangling_end_depth"][:7].isna().all()
    assert p["dangling_end_length"][:7].isna().all()
    assert a["dangling_end_depth_max"][0] == 2
    assert a["dangling_end_length_max"][0] == 4
    assert p["branch_depth"].tolist() == [0, 1, 1, 1, 1, 1, 1, 2, 3, 2]
    assert p["tip"].tolist() == [
        *[False, False, True, False, True, True, True],
        *[False, True, True],
    ]
    assert a.iloc[0][BRANCHES].tolist() == [6, 3, 6, 5, 12]


def test_pair_and_single():
    p, a = structure_properties(*_graph([(0, 1)], [2.0, 1.0]))
    assert p["branch_depth"].tolist() == [0, 1]
    assert p["branch_length"].tolist() == [0, 2]
    assert p["tip"].tolist() == [False, True]
    assert a.iloc[0][BRANCHES].tolist() == [1, 1, 2, 1, 6]

    p, a = structure_properties(*_graph([], [1.5]))
    one = p.iloc[0]
    assert (one["branch_depth"], one["branch_length"]) == (0, 0.0)
    assert not one["tip"]
    assert np.isnan(one["dangling_end_depth"])
    assert np.isnan(one["dangling_end_length"])
    row = a.iloc[0]
    assert (row["longest_chain"], row["longest_chain_length"]) == (0, 3.0)
    assert row["branch_count"] == 0
    for column in [
        "branch_depth_max",
        "branch_length_max",
        "dangling_end_depth_max",
        "dangling_end_length_max",
    ]:
        assert np.isnan(row[column]), column
    assert row[COUNTS].tolist() == [0] * 6


def test_base_tie_takes_the_lower_id():
    ring = [(i, (i + 1) % 4) for i in range(4)]
    p, _ = structure_properties(*_graph(ring, [1.0] * 4))
    assert p["branch_depth"].tolist() == [0, 1, 2, 1]
    particles, agglomerates, contacts = _graph(ring, [1.0] * 4)
    ids = np.array([5, 6, 0, 7])
    particles["id"] = ids
    contacts["id_a"] = ids[contacts["id_a"]]
    contacts["id_b"] = ids[contacts["id_b"]]
    p, _ = structure_properties(particles, agglomerates, contacts)
    assert p["branch_depth"].tolist() == [2, 1, 0, 1]


def _case_tables(name):
    _, particles, contacts = run_case(CASES[name][0])
    return particles, _agglomerates(particles), contacts


@pytest.mark.parametrize(
    "tables",
    [
        lambda: _case_tables("11 two links, thread with a triangle"),
        lambda: _case_tables("21 Y link between three cores"),
        lambda: _graph(*G4),
    ],
    ids=["case 11", "case 21", "G4"],
)
def test_rows_shuffled_and_ids_renumbered(tables):
    particles, agglomerates, contacts = tables()
    p, a = structure_properties(particles, agglomerates, contacts)

    rng = np.random.default_rng(0)
    # 10 * id + 7 keeps the order of the ids, so the tie rules pick the
    # same particles
    moved = particles.sample(frac=1, random_state=rng)
    moved["id"] = 10 * moved["id"] + 7
    moved_contacts = contacts.sample(frac=1, random_state=rng)
    for end in ("id_a", "id_b"):
        moved_contacts[end] = 10 * moved_contacts[end] + 7
    p2, a2 = structure_properties(
        moved, agglomerates.iloc[::-1], moved_contacts
    )
    p2["id"] = (p2["id"] - 7) // 10
    columns = list(PARTICLE_COLUMNS)
    pd.testing.assert_frame_equal(
        p2.sort_values("id")[columns].reset_index(drop=True),
        p.sort_values("id")[columns].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        a2.sort_values("agglomerate_id").reset_index(drop=True),
        a.sort_values("agglomerate_id").reset_index(drop=True),
    )


def _grid(side=15, chords=30, seed=1):
    """A side x side grid with random chords and radii U(0.5, 2)."""
    rng = np.random.default_rng(seed)
    at = np.arange(side * side).reshape(side, side)
    edges = set()
    for u, v in ((at[:, :-1], at[:, 1:]), (at[:-1], at[1:])):
        edges |= set(zip(u.ravel().tolist(), v.ravel().tolist(), strict=True))
    while len(edges) < 2 * side * (side - 1) + chords:
        u, v = sorted(rng.choice(side * side, 2, replace=False).tolist())
        edges.add((u, v))
    return sorted(edges), rng.uniform(0.5, 2, side * side).tolist()


def _oracle(edges, r):
    """Longest chain from every pair: (hops, length)."""
    r = np.asarray(r)
    a = np.array([e[0] for e in edges])
    b = np.array([e[1] for e in edges])
    n = len(r)
    hops = shortest_path(
        coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr(),
        directed=False,
        unweighted=True,
    )
    dist = shortest_path(
        coo_matrix((r[a] + r[b], (a, b)), shape=(n, n)).tocsr(),
        directed=False,
    )
    length = dist + r[:, None] + r[None, :]
    np.fill_diagonal(length, -np.inf)
    return hops[np.isfinite(hops)].max(), length.max()


def _oracle_graphs():
    out = {"G2": G2, "grid": _grid()}
    for name, (circles, _) in CASES.items():
        _, particles, contacts = run_case(circles)
        edges = list(zip(contacts["id_a"], contacts["id_b"], strict=True))
        out[name] = (edges, particles["r"].tolist())
    return out


ORACLE = _oracle_graphs()


@pytest.mark.parametrize("size", [None, 0, 10**9])
@pytest.mark.parametrize("name", list(ORACLE))
def test_longest_chain_matches_all_pairs(monkeypatch, name, size):
    # None: the default; 0: the bounds for every agglomerate; 10**9:
    # every pair for every agglomerate
    if size is not None:
        monkeypatch.setattr(graph_module, "_EVERY_PAIR_SIZE", size)
    edges, r = ORACLE[name]
    _, a = structure_properties(*_graph(edges, r))
    hops, length = _oracle(edges, r)
    assert a["longest_chain"][0] == hops
    np.testing.assert_allclose(
        a["longest_chain_length"][0], length, rtol=1e-12
    )


def test_grid_is_above_every_pair_size():
    assert len(ORACLE["grid"][1]) > graph_module._EVERY_PAIR_SIZE


def test_csv_round_trip_gives_the_same_columns(tmp_path):
    particles, agglomerates, contacts = _case_tables("8 thread on a link")
    p, a = structure_properties(particles, agglomerates, contacts)
    particles.to_csv(tmp_path / "particles.csv", index=False)
    contacts.to_csv(tmp_path / "contacts.csv", index=False)
    ids = ["id", "agglomerate_id"]
    read = pd.read_csv(tmp_path / "particles.csv")
    read[ids] = read[ids].astype(float)
    read_contacts = pd.read_csv(tmp_path / "contacts.csv").astype(
        {"agglomerate_id": float, "id_a": float, "id_b": float}
    )
    p2, a2 = structure_properties(
        read, agglomerates.astype(float), read_contacts
    )
    columns = list(PARTICLE_COLUMNS)
    pd.testing.assert_frame_equal(p2[columns], p[columns])
    pd.testing.assert_frame_equal(
        a2[list(AGGLOMERATE_COLUMNS)], a[list(AGGLOMERATE_COLUMNS)]
    )
