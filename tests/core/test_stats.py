"""Statistics: bin edges, binned distributions and per-image summary.

Values worked out by hand. Bin edges and distributions call the binning
functions of each implementation directly (``STATS``); summaries run a
circle case through it (``ADAPTERS``). Phase 2 turns this file into the
unit tests of ``agglpy.stats``.
"""

import math
from pathlib import Path

import pytest

from support.synthetic.adapters import (
    ADAPTERS,
    REAL_PIXEL_SIZE,
    STATS,
    expect_known_failure,
    skip_unless_supported,
)
from support.synthetic.cases import SUMMARY, Case, make_case

RTOL = 1e-12


def _bins_case(name: str) -> Case:
    """Placeholder Case so bin checks can use the known-failure list."""
    return make_case(name, [], groups=[])


# name, (start, end, periods, log, step), expected edges
BIN_CASES = [
    ("lin_periods", (0, 10, 5, False, False), [0, 2, 4, 6, 8, 10]),
    ("lin_step", (0, 10, 2.5, False, True), [0, 2.5, 5, 7.5, 10]),
    (
        "lin_step_metres",  # PSD_space as written in settings, in metres
        (0, 10e-6, 0.5e-6, False, True),
        [i * 0.5e-6 for i in range(21)],  # 0 .. 10 µm, 21 edges
    ),
    (
        "lin_step_from_0.1um",
        (0.1e-6, 10e-6, 0.1e-6, False, True),
        [i * 0.1e-6 for i in range(1, 101)],  # 0.1 .. 10 µm, 100 edges
    ),
    ("log_periods", (1e-7, 1e-5, 2, True, False), [1e-7, 1e-6, 1e-5]),
    (
        "log_step_half_decade",
        (1e-7, 1e-5, 0.5, True, True),
        [1e-7, 10**-6.5, 1e-6, 10**-5.5, 1e-5],
    ),
    (
        "log_step_tenth_decade",
        (1e-7, 1e-5, 0.1, True, True),
        [10 ** (-7 + i / 10) for i in range(21)],
    ),
]


@pytest.mark.parametrize(
    ("name", "args", "expected"), BIN_CASES, ids=[c[0] for c in BIN_CASES]
)
def test_psd_bins(
    request: pytest.FixtureRequest,
    adapter: str,
    name: str,
    args: tuple,
    expected: list[float],
):
    """Edges run from start to end inclusive: periods bins, or one bin
    per step (a step in decades for log bins)."""
    skip_unless_supported(adapter, "psd_bins")
    expect_known_failure(request, adapter, "psd_bins", _bins_case(name))
    edges = STATS[adapter].psd_bins(*args)
    assert len(edges) == len(expected)
    assert list(edges) == pytest.approx(expected, rel=RTOL, abs=0)


def test_psd_bins_log_needs_positive_start(adapter: str):
    skip_unless_supported(adapter, "psd_bins")
    with pytest.raises(ValueError):
        STATS[adapter].psd_bins(0, 1e-5, 2, True, False)


def test_distribution(adapter: str):
    """Right-closed bins ``(a, b]`` (author's decision, kept as today):
    a value on an edge belongs to the bin it closes, a value equal to the
    first edge is dropped, a value equal to the last edge is kept. Values
    outside the bins are dropped, and the normalized columns refer to
    the binned values only (7 here, not 9)."""
    skip_unless_supported(adapter, "distribution")
    bins = STATS[adapter].psd_bins(0, 40, 4, False, False)  # 0 10 20 30 40
    values = [0, 5, 10, 15, 20, 20, 35, 40, 45]
    dist = STATS[adapter].distribution(values, bins)

    # (0,10]: 5 10 | (10,20]: 15 20 20 | (20,30]: - | (30,40]: 35 40
    count = [2, 3, 0, 2]
    # volume per bin = count * pi / 6 * mid³, mids 5 15 25 35
    vol = [2 * 125, 3 * 3375, 0, 2 * 42875]  # x pi / 6; total 96125
    expected = {
        "left": [0, 10, 20, 30],
        "right": [10, 20, 30, 40],
        "mid": [5, 15, 25, 35],
        "width": [10, 10, 10, 10],
        "count": count,
        "cumulative": [2, 5, 5, 7],
        "count_norm": [2 / 7, 3 / 7, 0, 2 / 7],
        "cumulative_norm": [2 / 7, 5 / 7, 5 / 7, 1],
        "volume": [v * math.pi / 6 for v in vol],
        "volume_norm": [v / 96125 for v in vol],
        "volume_cumulative_norm": [
            250 / 96125,
            10375 / 96125,
            10375 / 96125,
            1,
        ],
    }
    for column, values_expected in expected.items():
        assert list(dist[column]) == pytest.approx(
            values_expected, rel=RTOL, abs=1e-15
        ), column


# Diameters in the summary are compared in px; a real pixel size checks
# that they are converted and the counts and ratios are not.
SUMMARY_RUNS = [
    pytest.param(c, px, id=f"{c.name}-px{px:g}")
    for c in SUMMARY
    for px in (1.0, REAL_PIXEL_SIZE)
]


@pytest.mark.parametrize(("case", "pixel_size"), SUMMARY_RUNS)
def test_summary(
    request: pytest.FixtureRequest,
    adapter: str,
    case: Case,
    pixel_size: float,
    tmp_path: Path,
):
    skip_unless_supported(adapter, "summary")
    expect_known_failure(request, adapter, "summary", case, pixel_size)
    result = ADAPTERS[adapter](case, tmp_path, pixel_size=pixel_size)
    for name, value in case.summary.items():
        assert result.summary[name] == pytest.approx(
            value, rel=RTOL, abs=0, nan_ok=True
        ), f"{case.name}: {name}"
