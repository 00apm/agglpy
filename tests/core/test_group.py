"""Grouping of primary particles into agglomerates, and idj detection.

Synthetic cases from ``support.synthetic.cases`` with hand-worked
answers, run once per implementation in
``support.synthetic.adapters.ADAPTERS``. Today that is the legacy code;
Phase 2 adds the new core and the same cases become the unit tests of
``agglpy.group``.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agglpy.group import Contacts, find_agglomerates, find_contacts
from agglpy.tables import validate_particles

from support.synthetic.adapters import (
    ADAPTERS,
    REAL_PIXEL_SIZE,
    expect_known_failure,
    skip_unless_supported,
)
from support.synthetic.cases import (
    DOUBLETS,
    SINGLE,
    STRUCTURES,
    Case,
    long_chain,
)
from support.synthetic.transforms import TRANSFORMS, Transform

CASES = [SINGLE, *DOUBLETS, *STRUCTURES]

# Every case runs in px. The doublet series also runs with a realistic
# pixel size, because the legacy code scales by it before comparing.
RUNS = [
    *[pytest.param(c, 1.0, id=f"{c.name}-px1") for c in CASES],
    *[
        pytest.param(c, REAL_PIXEL_SIZE, id=f"{c.name}-px{REAL_PIXEL_SIZE}")
        for c in DOUBLETS
    ],
]

# Deeper than Python's default recursion limit (1000) from wherever a
# search starts: starting in the middle still needs n / 2 levels.
LONG_CHAIN = long_chain(2500)


@pytest.fixture
def default_recursion_limit() -> Iterator[None]:
    """Run with Python's default recursion limit (IPython raises it)."""
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    yield
    sys.setrecursionlimit(previous)


@pytest.mark.parametrize(("case", "pixel_size"), RUNS)
def test_grouping(
    request: pytest.FixtureRequest,
    adapter: str,
    case: Case,
    pixel_size: float,
    tmp_path: Path,
):
    skip_unless_supported(adapter, "grouping")
    expect_known_failure(request, adapter, "grouping", case, pixel_size)
    result = ADAPTERS[adapter](case, tmp_path, pixel_size=pixel_size)
    assert result.groups() == case.groups, case.note


@pytest.mark.parametrize(("case", "pixel_size"), RUNS)
def test_idj(
    request: pytest.FixtureRequest,
    adapter: str,
    case: Case,
    pixel_size: float,
    tmp_path: Path,
):
    skip_unless_supported(adapter, "idj")
    expect_known_failure(request, adapter, "idj", case, pixel_size)
    result = ADAPTERS[adapter](case, tmp_path, pixel_size=pixel_size)
    assert result.idj_keys() == case.idj, case.note


@pytest.mark.parametrize("transform", TRANSFORMS.values(), ids=TRANSFORMS)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_result_does_not_depend_on_order_or_position(
    adapter: str, case: Case, transform: Transform, tmp_path: Path
):
    """Input order, position (incl. negative coordinates, i.e. outside
    the image), mirroring and swapped axes must not change the result."""
    skip_unless_supported(adapter, "grouping")
    skip_unless_supported(adapter, "idj")
    result = ADAPTERS[adapter](transform(case), tmp_path)
    assert result.groups() == case.groups, case.note
    assert result.idj_keys() == case.idj, case.note


@pytest.mark.usefixtures("default_recursion_limit")
def test_long_chain_is_one_agglomerate(
    request: pytest.FixtureRequest, adapter: str, tmp_path: Path
):
    skip_unless_supported(adapter, "grouping")
    expect_known_failure(request, adapter, "grouping", LONG_CHAIN)
    result = ADAPTERS[adapter](LONG_CHAIN, tmp_path)
    assert result.groups() == LONG_CHAIN.groups, LONG_CHAIN.note


# ---------------------------------------------------------------------
# Unit tests of agglpy.group (one run, no adapter)
# ---------------------------------------------------------------------


def _table(x, y, r, ids=None) -> pd.DataFrame:
    """A validated particle table, without the duplicate warning."""
    ids = list(range(1, len(x) + 1)) if ids is None else ids
    return validate_particles(
        pd.DataFrame({"id": ids, "x": x, "y": y, "r": r, "source": "hct"}),
        warn_duplicates=False,
    )


def _pairs(contacts: Contacts) -> set[tuple[int, int]]:
    """Contacts as (larger, smaller) row-position pairs."""
    return set(
        zip(contacts.larger.tolist(), contacts.smaller.tolist(), strict=True)
    )


def _brute_force_pairs(table: pd.DataFrame) -> set[tuple[int, int]]:
    """Touching pairs from checking every pair, oriented as Contacts."""
    xy = table[["x", "y"]].to_numpy()
    r = table["r"].to_numpy()
    ids = table["id"].to_numpy()
    pairs = set()
    for a in range(len(table)):
        for b in range(a + 1, len(table)):
            dx, dy = xy[a] - xy[b]
            if np.sqrt(dx * dx + dy * dy) <= r[a] + r[b]:
                a_owns = r[a] > r[b] or (r[a] == r[b] and ids[a] < ids[b])
                pairs.add((a, b) if a_owns else (b, a))
    return pairs


def test_touching_pair_is_a_contact():
    table = _table([0, 14], [0, 0], [10, 4])  # d = 14 = 10 + 4
    assert _pairs(find_contacts(table)) == {(0, 1)}


def test_separate_pair_is_not_a_contact():
    table = _table([0, 15], [0, 0], [10, 4])  # d = 15 > 14
    assert _pairs(find_contacts(table)) == set()


def test_contact_is_owned_by_the_larger_particle():
    table = _table([0, 14], [0, 0], [4, 10])  # larger one in row 1
    assert _pairs(find_contacts(table)) == {(1, 0)}


def test_equal_radii_contact_is_owned_by_the_lower_id():
    table = _table([0, 20], [0, 0], [10, 10], ids=[7, 3])
    assert _pairs(find_contacts(table)) == {(1, 0)}


def test_large_particle_finds_small_far_neighbour():
    # The small one searches only 2 * 2 px; the big one must find it.
    table = _table([0, 101], [0, 0], [100, 2])  # d = 101 <= 102
    assert _pairs(find_contacts(table)) == {(0, 1)}


def test_contacts_match_brute_force():
    rng = np.random.default_rng(0)
    n = 300
    table = _table(
        rng.uniform(0, 150, n),
        rng.uniform(0, 150, n),
        rng.lognormal(np.log(4), 0.5, n),
    )
    assert _pairs(find_contacts(table)) == _brute_force_pairs(table)


def test_contacts_match_brute_force_on_half_pixel_grid():
    # HCT gives centres and radii on a 0.5 px grid: many pairs touch
    # exactly, which the KD-tree's own rounding must not drop.
    rng = np.random.default_rng(1)
    n = 300
    table = _table(
        rng.integers(0, 120, n) / 2,
        rng.integers(0, 120, n) / 2,
        rng.integers(2, 12, n) / 2,
    )
    expected = _brute_force_pairs(table)
    xy, r = table[["x", "y"]].to_numpy(), table["r"].to_numpy()
    tangent = [
        (a, b)
        for a, b in expected
        if np.hypot(*(xy[a] - xy[b])) == r[a] + r[b]
    ]
    assert tangent, "the grid should produce exactly touching pairs"
    assert _pairs(find_contacts(table)) == expected


@pytest.mark.parametrize("n", [0, 1])
def test_no_contacts_without_a_pair(n: int):
    contacts = find_contacts(_table([0] * n, [0] * n, [5] * n))
    assert contacts.larger.dtype == np.intp
    assert len(contacts.larger) == len(contacts.smaller) == 0


def _contacts(*pairs: tuple[int, int]) -> Contacts:
    larger = np.array([p[0] for p in pairs], dtype=np.intp)
    smaller = np.array([p[1] for p in pairs], dtype=np.intp)
    return Contacts(larger, smaller)


def test_chain_of_contacts_is_one_agglomerate():
    # A-B and B-C touch, A and C don't: still one agglomerate. D alone.
    table = _table([0, 1, 2, 3], [0] * 4, [1] * 4)
    ids = find_agglomerates(table, _contacts((0, 1), (1, 2)))
    assert ids.tolist() == [0, 0, 0, 1]


def test_particle_without_contacts_is_its_own_agglomerate():
    table = _table([0, 100, 200], [0] * 3, [1] * 3)
    assert find_agglomerates(table, _contacts()).tolist() == [0, 1, 2]


def test_agglomerate_ids_follow_the_smallest_particle_id():
    # Rows 0-1 hold ids 5, 6; rows 2-3 ids 1, 2: that group is first.
    table = _table([0, 1, 50, 51], [0] * 4, [1] * 4, ids=[5, 6, 1, 2])
    ids = find_agglomerates(table, _contacts((0, 1), (2, 3)))
    assert ids.tolist() == [1, 1, 0, 0]


def test_agglomerate_ids_do_not_depend_on_row_order():
    rng = np.random.default_rng(2)
    n = 200
    table = _table(
        rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(1, 4, n)
    )
    shuffled = table.sample(frac=1, random_state=3).reset_index(drop=True)

    def by_particle(t: pd.DataFrame) -> dict[int, int]:
        ids = find_agglomerates(t, find_contacts(t))
        return dict(zip(t["id"], ids, strict=True))

    assert by_particle(shuffled) == by_particle(table)


def test_agglomerates_of_empty_table():
    ids = find_agglomerates(_table([], [], []), _contacts())
    assert ids.dtype == np.int64
    assert ids.name == "agglomerate_id"
    assert len(ids) == 0
