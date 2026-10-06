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

import pytest

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
