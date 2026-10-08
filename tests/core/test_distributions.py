"""Size classes and distributions (``agglpy.core.distributions``).

The 1.4 checks call the functions of each implementation directly
(``STATS``) with the core's vocabulary; the unit tests below run the
core on tables written by hand, with the values worked out in the 2.5
plan.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from agglpy.core.distributions import distribution, size_classes
from agglpy.errors import ParamsError, TableError, ValuesNotCountedWarning

from support.synthetic.adapters import (
    STATS,
    expect_known_failure,
    skip_unless_supported,
)
from support.synthetic.cases import Case, make_case

RTOL = 1e-12


def _check_case(name: str) -> Case:
    """Placeholder Case so these checks can use the known-failure list."""
    return make_case(name, [], groups=[])


# name, size_classes arguments, expected edges
CLASS_CASES = [
    ("lin_count", (0, 10), {"count": 5}, [0, 2, 4, 6, 8, 10]),
    ("lin_step", (0, 10), {"step": 2.5}, [0, 2.5, 5, 7.5, 10]),
    (
        "lin_step_metres",  # as written in agglpy 0.4 settings, in metres
        (0, 10e-6),
        {"step": 0.5e-6},
        [i * 0.5e-6 for i in range(21)],  # 0 .. 10 µm, 21 edges
    ),
    (
        "lin_step_from_0.1um",
        (0.1e-6, 10e-6),
        {"step": 0.1e-6},
        [i * 0.1e-6 for i in range(1, 101)],  # 0.1 .. 10 µm, 100 edges
    ),
    (
        "log_count",
        (1e-7, 1e-5),
        {"count": 2, "scale": "log"},
        [1e-7, 1e-6, 1e-5],
    ),
    (
        "log_step_half_decade",  # the step is the factor between edges
        (1e-7, 1e-5),
        {"step": 10**0.5, "scale": "log"},
        [1e-7, 10**-6.5, 1e-6, 10**-5.5, 1e-5],
    ),
    (
        "log_step_tenth_decade",
        (1e-7, 1e-5),
        {"step": 10**0.1, "scale": "log"},
        [10 ** (-7 + i / 10) for i in range(21)],
    ),
]


@pytest.mark.parametrize(
    ("name", "ends", "kwargs", "expected"),
    CLASS_CASES,
    ids=[c[0] for c in CLASS_CASES],
)
def test_size_classes_cases(
    request: pytest.FixtureRequest,
    adapter: str,
    name: str,
    ends: tuple[float, float],
    kwargs: dict,
    expected: list[float],
):
    """Edges run from start to end inclusive: ``count`` classes, or one
    per step (for log classes the factor between edges)."""
    skip_unless_supported(adapter, "size_classes")
    expect_known_failure(request, adapter, "size_classes", _check_case(name))
    edges = STATS[adapter].size_classes(*ends, **kwargs)
    assert len(edges) == len(expected)
    assert list(edges) == pytest.approx(expected, rel=RTOL, abs=0)


def test_log_size_classes_need_a_positive_start(adapter: str):
    skip_unless_supported(adapter, "size_classes")
    with pytest.raises(ValueError):
        STATS[adapter].size_classes(0, 1e-5, count=2, scale="log")


def test_distribution_case(request: pytest.FixtureRequest, adapter: str):
    """Right-closed classes ``(a, b]``: a value on an edge belongs to
    the class it closes; the outer edge is included, so 0 counts.
    Values outside the classes are not counted, and the fractions refer
    to the counted values only (8 here, not 9)."""
    skip_unless_supported(adapter, "distribution")
    expect_known_failure(
        request, adapter, "distribution", _check_case("edges_and_range")
    )
    edges = np.array([0.0, 10, 20, 30, 40])
    values = [0, 5, 10, 15, 20, 20, 35, 40, 45]
    dist = STATS[adapter].distribution(values, edges)

    # [0,10]: 0 5 10 | (10,20]: 15 20 20 | (20,30]: - | (30,40]: 35 40
    expected = {
        "left": [0, 10, 20, 30],
        "right": [10, 20, 30, 40],
        "mid": [5, 15, 25, 35],
        "width": [10, 10, 10, 10],
        "amount": [3, 3, 0, 2],
        "fraction": [3 / 8, 3 / 8, 0, 2 / 8],
        "cumulative": [3 / 8, 6 / 8, 6 / 8, 1],
    }
    for column, values_expected in expected.items():
        assert list(dist[column]) == pytest.approx(
            values_expected, rel=RTOL, abs=1e-15
        ), column


# ---------------------------------------------------------------------
# Unit tests of agglpy.core.distributions (one run, no adapter)
# ---------------------------------------------------------------------


def test_log_step_is_a_factor():
    edges = size_classes(0.1, 0.8, step=2, scale="log")
    assert edges.tolist() == pytest.approx([0.1, 0.2, 0.4, 0.8], rel=RTOL)
    assert (edges[0], edges[-1]) == (0.1, 0.8)  # outer edges exact


def test_size_classes_are_float_with_exact_ends():
    edges = size_classes(0, 1, step=0.1)
    assert edges.dtype == np.float64
    assert len(edges) == 11
    assert (edges[0], edges[-1]) == (0, 1)


@pytest.mark.parametrize(
    ("args", "kwargs", "message"),
    [
        ((0, 10), {"step": 3}, "does not divide"),
        ((0, 10), {}, "exactly one"),
        ((0, 10), {"count": 2, "step": 5}, "exactly one"),
        ((10, 0), {"count": 2}, "end must be > start"),
        ((0, 10), {"count": 0}, "count"),
        ((0, 10), {"count": 2.5}, "count"),
        ((0, 10), {"step": -1}, "step"),
        ((1, 10), {"step": 0.5, "scale": "log"}, "factor"),
        ((0, 10), {"count": 2, "scale": "log"}, "start > 0"),
        ((1, 10), {"count": 2, "scale": "ln"}, "scale"),
    ],
)
def test_size_classes_reject_bad_arguments(args, kwargs, message):
    with pytest.raises(ParamsError, match=message):
        size_classes(*args, **kwargs)


def _one_image(values, **columns) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = pd.DataFrame({"image": ["A"]})
    table = pd.DataFrame({"image": "A", "v": values, **columns})
    return images, table


def _quiet(*args, **kwargs) -> pd.DataFrame:
    """``distribution`` with the not-counted warning silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValuesNotCountedWarning)
        return distribution(*args, **kwargs)


EDGES = [0.0, 10, 20, 30, 40]


VALUES = [0, 5, 10, 15, 20, 20, 35, 40, 45]


def test_distribution_columns_and_number_basis():
    out = _quiet(*_one_image(VALUES), "v", EDGES)
    assert list(out.columns) == [
        "left",
        "mid",
        "right",
        "width",
        "basis",
        "amount",
        "fraction",
        "cumulative",
        "density",
        "log_density",
    ]
    assert out["basis"].unique().tolist() == ["number"]
    assert out["amount"].tolist() == [3, 3, 0, 2]
    assert out["density"].tolist() == pytest.approx(
        [3 / 80, 3 / 80, 0, 2 / 80], rel=RTOL
    )


def test_left_closed_includes_the_last_edge():
    out = _quiet(*_one_image(VALUES), "v", EDGES, closed="left")
    # [0,10): 0 5 | [10,20): 10 15 | [20,30): 20 20 | [30,40]: 35 40
    assert out["amount"].tolist() == [2, 2, 2, 2]


def test_values_not_counted_warn_per_group():
    images = pd.DataFrame({"image": ["A", "B"], "condition": ["x", "y"]})
    table = pd.DataFrame(
        {"image": ["A", "A", "B", "B"], "v": [5, 45, -1, np.nan]}
    )
    with pytest.warns(ValuesNotCountedWarning) as record:
        out = distribution(images, table, "v", EDGES, by="condition")
    message = str(record[0].message)
    assert "x: 1 outside, 0 NaN" in message
    assert "y: 1 outside, 1 NaN" in message
    # group y counted nothing: zero amounts, NaN fractions
    y = out[out["condition"] == "y"]
    assert y["amount"].tolist() == [0, 0, 0, 0]
    assert y["fraction"].isna().all()


def test_nan_values_are_not_counted_and_say_so():
    # size_ratio is NaN for every single particle
    images, table = _one_image([0.5, np.nan, np.nan])
    with pytest.warns(ValuesNotCountedWarning, match="0 outside, 2 NaN"):
        out = distribution(images, table, "v", [0, 1])
    assert out["amount"].tolist() == [1]
    assert out["fraction"].tolist() == [1]


def test_no_warning_when_everything_is_counted():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        distribution(*_one_image([1, 2]), "v", [0, 10])


@pytest.mark.parametrize(
    ("closed", "expected"), [("right", [0, 0, 1, 0]), ("left", [0, 0, 0, 1])]
)
def test_value_one_ulp_off_an_edge_counts_as_on_it(closed, expected):
    # 3 x 0.1 = 0.30000000000000004: a whole-px diameter times a pixel
    # size; without the tolerance it would land in the class above 0.3
    edges = [0.0, 0.1, 0.2, 0.3, 0.4]
    out = distribution(*_one_image([3 * 0.1]), "v", edges, closed=closed)
    assert out["amount"].tolist() == expected


def test_value_one_ulp_beyond_the_outer_edge_is_counted():
    out = distribution(*_one_image([10 * 0.1]), "v", [0.5, 1.0])
    assert out["amount"].tolist() == [1]


def test_log_density_and_its_nan_at_zero():
    out = distribution(*_one_image([0.5, 5, 50]), "v", [0, 1, 10, 100])
    assert math.isnan(out["log_density"].iloc[0])  # left = 0
    # fraction 1/3 over one decade
    assert out["log_density"].iloc[1:].tolist() == pytest.approx(
        [1 / 3, 1 / 3], rel=RTOL
    )


def test_weights_add_one_basis_each():
    images, table = _one_image(
        [5, 15, 15], volume=[1.0, 2.0, 5.0], mass=[1.0, 1.0, 2.0]
    )
    out = distribution(
        images, table, "v", [0, 10, 20], weights=["volume", "mass"]
    )
    assert out["basis"].tolist() == [
        "number",
        "number",
        "volume",
        "volume",
        "mass",
        "mass",
    ]
    # the sum of the real values per class, not count x mid³
    volume = out[out["basis"] == "volume"]
    assert volume["amount"].tolist() == [1, 7]
    assert volume["fraction"].tolist() == [0.125, 0.875]
    assert volume["cumulative"].tolist() == [0.125, 1]


def test_one_weight_name_is_one_column():
    images, table = _one_image([5], volume=[2.0])
    out = distribution(images, table, "v", [0, 10], weights="volume")
    assert out["basis"].tolist() == ["number", "volume"]


def test_missing_weight_column_raises():
    with pytest.raises(TableError, match="volume"):
        distribution(*_one_image([5]), "v", [0, 10], weights="volume")


def test_nan_weight_on_a_counted_value_raises():
    images, table = _one_image([5, 50], w=[np.nan, 1.0])
    with pytest.raises(TableError, match=r"'w'.*1 counted"):
        _quiet(images, table, "v", [0, 10], weights="w")
    # a NaN weight on a value that is not counted does no harm
    images, table = _one_image([5, 50], w=[1.0, np.nan])
    assert _quiet(images, table, "v", [0, 10], weights="w")[
        "amount"
    ].tolist() == [1, 1]


def test_every_class_in_every_group_and_blank_images():
    images = pd.DataFrame({"image": ["A", "B", "blank"]})
    table = pd.DataFrame({"image": ["A", "A", "B"], "v": [5, 15, 15]})
    out = distribution(images, table, "v", [0, 10, 20], by="image")
    assert out["image"].tolist() == ["A", "A", "B", "B", "blank", "blank"]
    assert out["amount"].tolist() == [1, 1, 0, 1, 0, 0]
    # cumulative runs within each group, not across them
    assert out["cumulative"].tolist()[:4] == [0.5, 1, 0, 1]
    assert out["fraction"].iloc[4:].isna().all()


def test_distribution_is_unit_agnostic():
    p = 0.25
    images, table = _one_image([1, 3, 3, 7])
    before = distribution(images, table, "v", [0, 2, 4, 8])
    after = distribution(
        images, table.assign(v=table["v"] * p), "v", [0, 2 * p, 4 * p, 8 * p]
    )
    for column, power in [
        ("left", 1),
        ("width", 1),
        ("amount", 0),
        ("fraction", 0),
        ("density", -1),
        ("log_density", 0),
    ]:
        np.testing.assert_allclose(
            after[column], before[column] * p**power, rtol=1e-12
        )


@pytest.mark.parametrize(
    "edges", [[1.0], [0, 2, 1], [0, 1, 1], [0, np.nan], [[0, 1], [1, 2]]]
)
def test_bad_edges_raise(edges):
    with pytest.raises(ParamsError, match="edges"):
        distribution(*_one_image([1]), "v", edges)


def test_bad_closed_raises():
    with pytest.raises(ParamsError, match="closed"):
        distribution(*_one_image([1]), "v", [0, 2], closed="both")


def test_text_column_raises():
    with pytest.raises(TableError, match="'v' must be numeric"):
        distribution(*_one_image(["big"]), "v", [0, 2])
