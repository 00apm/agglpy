"""Population metrics (``agglpy.core.metrics``).

The 1.4 synthetic summary cases (``support.synthetic.cases.SUMMARY``)
run once per implementation; the unit tests below run the core on
tables written by hand, with the values worked out in the 2.5 plan.
"""

from pathlib import Path

import pytest

from support.synthetic.adapters import (
    ADAPTERS,
    REAL_PIXEL_SIZE,
    expect_known_failure,
    skip_unless_supported,
)
from support.synthetic.cases import SUMMARY, Case

RTOL = 1e-12

# Diameters in the summary are compared in px; a real pixel size checks
# that they are converted and the counts and ratios are not.
SUMMARY_RUNS = [
    pytest.param(c, px, id=f"{c.name}-px{px:g}")
    for c in SUMMARY
    for px in (1.0, REAL_PIXEL_SIZE)
]


@pytest.mark.parametrize(("case", "pixel_size"), SUMMARY_RUNS)
def test_summary_cases(
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
