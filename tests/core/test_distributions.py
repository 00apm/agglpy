"""Size classes and distributions (``agglpy.core.distributions``).

The 1.4 checks call the functions of each implementation directly
(``STATS``) with the core's vocabulary; the unit tests below run the
core on tables written by hand, with the values worked out in the 2.5
plan.
"""

import numpy as np
import pytest

from agglpy.core.distributions import size_classes
from agglpy.errors import ParamsError

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
