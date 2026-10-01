"""Grouping of primary particles into agglomerates, and idj detection.

Synthetic cases from ``support.synthetic.cases`` with hand-worked
answers. They run against the legacy code now; Phase 2 adds the new
core's adapter to ``ADAPTERS`` and the same cases become the unit tests
of ``agglpy.group``.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from support.synthetic.cases import DOUBLETS, SINGLE, STRUCTURES, Case
from support.synthetic.legacy_adapter import run_legacy
from support.synthetic.result import Result
from support.synthetic.transforms import TRANSFORMS, Transform

Adapter = Callable[..., Result]

ADAPTERS: dict[str, Adapter] = {"legacy": run_legacy}

CASES = [SINGLE, *DOUBLETS, *STRUCTURES]

# Pixel size 1.0 keeps coordinates in px. The doublet series also runs
# with a realistic SEM pixel size in metres, because the legacy code
# scales by it before the contact test.
REAL_PIXEL_SIZE = 2.5e-9
RUNS = [
    *[pytest.param(c, 1.0, id=f"{c.name}-px1") for c in CASES],
    *[
        pytest.param(c, REAL_PIXEL_SIZE, id=f"{c.name}-px{REAL_PIXEL_SIZE}")
        for c in DOUBLETS
    ],
]

# The legacy code scales x, y and r to metres before the contact and idj
# tests, so an exact tangency in px can miss by one ulp: 14 * 2.5e-9 is
# 3.5e-08, but 10 * 2.5e-9 + 4 * 2.5e-9 is 3.4999999999999996e-08.
# Fixed by testing contacts in px in the new grouper (roadmap 2.3).
LEGACY_KNOWN_FAILURES = {
    ("grouping", "doublet_unequal_tangent", REAL_PIXEL_SIZE),
    ("idj", "doublet_internally_tangent", REAL_PIXEL_SIZE),
}


def expect_known_failure(
    request: pytest.FixtureRequest,
    adapter: str,
    check: str,
    case: Case,
    pixel_size: float,
) -> None:
    """Mark a legacy run as a strict xfail if it is a known legacy bug."""
    if (
        adapter == "legacy"
        and (check, case.name, pixel_size) in LEGACY_KNOWN_FAILURES
    ):
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="tangency lost by scaling to metres first (2.3)",
            )
        )


@pytest.fixture(params=list(ADAPTERS))
def adapter(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.mark.parametrize(("case", "pixel_size"), RUNS)
def test_grouping(
    request: pytest.FixtureRequest,
    adapter: str,
    case: Case,
    pixel_size: float,
    tmp_path: Path,
):
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
    result = ADAPTERS[adapter](transform(case), tmp_path)
    assert result.groups() == case.groups, case.note
    assert result.idj_keys() == case.idj, case.note
