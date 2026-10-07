"""Grouping of primary particles into agglomerates, and idj detection.

Synthetic cases from ``support.synthetic.cases`` with hand-worked
answers, run once per implementation in
``support.synthetic.adapters.ADAPTERS``. Today that is the legacy code;
Phase 2 adds the new core and the same cases become the unit tests of
``agglpy.core.agglomerates``.
"""

import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agglpy.core.agglomerates import (
    Contacts,
    find_agglomerates,
    find_contacts,
    find_enclosed,
    label_agglomerates,
)
from agglpy.errors import DuplicateParticlesWarning, ParticleTableError
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
# The core works in px, so for it those runs repeat the px1 runs.
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
# Unit tests of agglpy.core.agglomerates (one run, no adapter)
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
    # exactly. The arithmetic is exact on this grid, so this doesn't
    # test the search margin (see the next test).
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


def test_tangent_pair_far_from_origin_is_found():
    # d == 2 * r exactly, but the KD-tree's own rounding drops the pair
    # without _SEARCH_MARGIN (found by the 2.3 review).
    r = 49.4347672837589
    table = _table(
        [-54676.54776590472, -54661.01603785207],
        [-47423.292388719005, -47520.93433815313],
        [r, r],
    )
    assert len(find_contacts(table).larger) == 1


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
    ids = label_agglomerates(table, _contacts((0, 1), (1, 2)))
    assert ids.tolist() == [0, 0, 0, 1]


def test_particle_without_contacts_is_its_own_agglomerate():
    table = _table([0, 100, 200], [0] * 3, [1] * 3)
    assert label_agglomerates(table, _contacts()).tolist() == [0, 1, 2]


def test_agglomerate_ids_follow_the_smallest_particle_id():
    # Rows 0-1 hold ids 5, 6; rows 2-3 ids 1, 2: that group is first.
    table = _table([0, 1, 50, 51], [0] * 4, [1] * 4, ids=[5, 6, 1, 2])
    ids = label_agglomerates(table, _contacts((0, 1), (2, 3)))
    assert ids.tolist() == [1, 1, 0, 0]


def test_agglomerate_ids_do_not_depend_on_row_order():
    rng = np.random.default_rng(2)
    n = 200
    table = _table(
        rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(1, 4, n)
    )
    shuffled = table.sample(frac=1, random_state=3).reset_index(drop=True)

    def by_particle(t: pd.DataFrame) -> dict[int, int]:
        ids = label_agglomerates(t, find_contacts(t))
        return dict(zip(t["id"], ids, strict=True))

    assert by_particle(shuffled) == by_particle(table)


def test_agglomerates_of_empty_table():
    ids = label_agglomerates(_table([], [], []), _contacts())
    assert ids.dtype == np.int64
    assert ids.name == "agglomerate_id"
    assert len(ids) == 0


def _enclosed(table: pd.DataFrame) -> list[bool]:
    return find_enclosed(table, find_contacts(table)).tolist()


def test_small_particle_inside_large_is_enclosed():
    assert _enclosed(_table([0, 2], [0, 0], [10, 4])) == [False, True]


def test_internally_tangent_particle_is_enclosed():
    # d = 6 = 10 - 4: touches the big circle's outline from inside.
    assert _enclosed(_table([0, 6], [0, 0], [10, 4])) == [False, True]


def test_overlapping_particle_is_not_enclosed():
    assert _enclosed(_table([0, 7], [0, 0], [10, 4])) == [False, False]


def test_concentric_small_inside_large_gives_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _enclosed(_table([0, 0], [0, 0], [10, 4])) == [False, True]


def test_exact_duplicate_flags_only_higher_id():
    table = _table([5, 5], [5, 5], [8, 8], ids=[2, 1])
    with pytest.warns(DuplicateParticlesWarning, match=r"\(1, 2\)"):
        assert _enclosed(table) == [True, False]  # id 2 is enclosed


def test_near_identical_enclosed_pair_warns():
    # The real D7-019 pair: same centre, R = 30.5 and 29.4.
    table = _table([43.5, 43.5], [109.5, 109.5], [30.5, 29.4])
    with pytest.warns(DuplicateParticlesWarning, match="near-identical"):
        assert _enclosed(table) == [False, True]


def test_duplicate_warning_counts_enclosed_particles_not_pairs():
    # Three identical circles: 3 pairs, but only ids 2 and 3 enclosed.
    table = _table([5, 5, 5], [5, 5, 5], [8, 8, 8])
    with pytest.warns(DuplicateParticlesWarning, match=r"^2 enclosed"):
        assert _enclosed(table) == [False, True, True]


def test_enclosed_of_empty_table():
    flags = find_enclosed(_table([], [], []), _contacts())
    assert flags.dtype == bool
    assert flags.name == "enclosed"
    assert len(flags) == 0


def test_find_agglomerates_adds_both_columns():
    table = _table([0, 14, 100], [0, 0, 0], [10, 4, 3])
    out = find_agglomerates(table)
    assert out["agglomerate_id"].tolist() == [0, 0, 1]
    assert out["enclosed"].tolist() == [False, False, False]
    assert out["agglomerate_id"].dtype == np.int64
    assert out["enclosed"].dtype == bool


def test_find_agglomerates_does_not_modify_its_input():
    table = _table([0, 2], [0, 0], [10, 4])
    before = table.copy()
    find_agglomerates(table)
    pd.testing.assert_frame_equal(table, before)


def test_custom_index_is_reset():
    table = _table([0, 14, 100], [0, 0, 0], [10, 4, 3])
    table.index = [30, 10, 20]
    out = find_agglomerates(table)
    assert list(out.index) == [0, 1, 2]
    assert out["id"].tolist() == [1, 2, 3]  # rows keep the input order


def test_regrouping_gives_the_same_result():
    once = find_agglomerates(_table([0, 2, 50], [0, 0, 0], [10, 4, 3]))
    pd.testing.assert_frame_equal(find_agglomerates(once), once)


def test_empty_table():
    out = find_agglomerates(_table([], [], []))
    assert len(out) == 0
    assert out["agglomerate_id"].dtype == np.int64
    assert out["enclosed"].dtype == bool


def test_find_agglomerates_warns_once_about_duplicates():
    table = _table([5, 5], [5, 5], [8, 8])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        find_agglomerates(table)
    duplicates = [
        w for w in caught if issubclass(w.category, DuplicateParticlesWarning)
    ]
    assert len(duplicates) == 1


def test_find_agglomerates_rejects_a_broken_table():
    with pytest.raises(ParticleTableError):
        find_agglomerates(pd.DataFrame({"id": [1], "x": [0.0], "y": [0.0]}))
