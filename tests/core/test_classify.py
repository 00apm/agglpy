"""Classification of agglomerates and their particles.

Rule under test: an agglomerate is a collector when the diameter ratio
of its second largest to its largest particle is ``<= threshold``. The
largest particle is then the collector and all others attached2coll;
otherwise all members are similar. A single particle is separate.
Cases in ``support.synthetic.cases.CLASSIFICATION``; Phase 2 turns this
file into the unit tests of ``agglpy.classify``.
"""

from pathlib import Path

import pytest

from support.synthetic.adapters import (
    ADAPTERS,
    REAL_PIXEL_SIZE,
    expect_known_failure,
    skip_unless_supported,
)
from support.synthetic.cases import CLASSIFICATION, Case

# Cases with the ratio exactly at the threshold also run with a
# realistic pixel size: the legacy code computes the ratio in metres.
AT_THRESHOLD = [c for c in CLASSIFICATION if c.name.startswith("ratio_at")]
RUNS = [
    *[pytest.param(c, 1.0, id=f"{c.name}-px1") for c in CLASSIFICATION],
    *[
        pytest.param(c, REAL_PIXEL_SIZE, id=f"{c.name}-px{REAL_PIXEL_SIZE}")
        for c in AT_THRESHOLD
    ],
]


@pytest.mark.parametrize(("case", "pixel_size"), RUNS)
def test_particle_types(
    request: pytest.FixtureRequest,
    adapter: str,
    case: Case,
    pixel_size: float,
    tmp_path: Path,
):
    skip_unless_supported(adapter, "classify")
    expect_known_failure(request, adapter, "classify", case, pixel_size)
    result = ADAPTERS[adapter](case, tmp_path, pixel_size=pixel_size)
    assert result.particle_types() == dict(case.types), case.note


@pytest.mark.parametrize(("case", "pixel_size"), RUNS)
def test_agglomerate_types(
    request: pytest.FixtureRequest,
    adapter: str,
    case: Case,
    pixel_size: float,
    tmp_path: Path,
):
    skip_unless_supported(adapter, "classify")
    expect_known_failure(request, adapter, "classify", case, pixel_size)
    result = ADAPTERS[adapter](case, tmp_path, pixel_size=pixel_size)
    assert result.agglomerate_types() == case.agglomerate_types(), case.note
