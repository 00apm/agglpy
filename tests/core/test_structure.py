"""Structure of agglomerates (``agglpy.core.structure``).

The catalogue (``support.structure_cases``) holds circle sets with
their segments read by eye; the hand graphs below give the contact
graph directly, as edge lists of particle ids.
"""

import numpy as np
import pandas as pd
import pytest

from agglpy.core.structure import structure_properties
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
    assert p["segment_type"].dtype == object
    assert p["segment_id"].dtype == np.int64
    for column in COUNTS:
        assert a[column].dtype == np.int64


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
