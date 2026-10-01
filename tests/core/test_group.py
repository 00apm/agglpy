"""Grouping of primary particles into agglomerates, and idj detection.

Synthetic cases from ``support.synthetic.cases`` with hand-worked answers. They
run against the legacy code now; Phase 2 adds the new core's adapter to
``ADAPTERS`` and the same cases become the unit tests of
``agglpy.group``.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from support.synthetic.cases import DOUBLETS, SINGLE, Case
from support.synthetic.legacy_adapter import run_legacy
from support.synthetic.result import Result

Adapter = Callable[..., Result]

ADAPTERS: dict[str, Adapter] = {"legacy": run_legacy}

GROUPING_CASES = [SINGLE, *DOUBLETS]

# 1.0: coordinates stay in px. 2.5e-9: a realistic SEM pixel size in
# metres; the legacy code scales by it before the contact test.
PIXEL_SIZES = [1.0, 2.5e-9]

# The legacy code scales x, y and r to metres before the contact and idj
# tests, so an exact tangency in px can miss by one ulp: 14 * 2.5e-9 is
# 3.5e-08, but 10 * 2.5e-9 + 4 * 2.5e-9 is 3.4999999999999996e-08.
# Fixed by testing contacts in px in the new grouper (roadmap 2.3).
LEGACY_KNOWN_FAILURES = {
    ("grouping", "doublet_unequal_tangent", 2.5e-9),
    ("idj", "doublet_internally_tangent", 2.5e-9),
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


@pytest.mark.parametrize("pixel_size", PIXEL_SIZES, ids=lambda v: f"px{v:g}")
@pytest.mark.parametrize("case", GROUPING_CASES, ids=lambda c: c.name)
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


@pytest.mark.parametrize("pixel_size", PIXEL_SIZES, ids=lambda v: f"px{v:g}")
@pytest.mark.parametrize("case", GROUPING_CASES, ids=lambda c: c.name)
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
