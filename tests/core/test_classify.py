"""Classification of agglomerates and their particles.

Rule under test: an agglomerate is a collector when the diameter ratio
of its second largest to its largest particle is ``<= threshold``. The
largest particle is then the collector and all others attached2coll;
otherwise all members are similar. A single particle is separate.
Cases in ``support.synthetic.cases.CLASSIFICATION``. The library has no
classification: for the core, the rule is the recipe in
``examples/agglomerate_types.py`` applied to ``size_ratio``.
"""

from pathlib import Path

import pytest

from support.synthetic.adapters import (
    ADAPTERS,
    expect_known_failure,
    skip_unless_supported,
)
from support.synthetic.cases import CLASSIFICATION, Case


@pytest.mark.parametrize("case", CLASSIFICATION, ids=lambda c: c.name)
def test_particle_types(
    request: pytest.FixtureRequest, adapter: str, case: Case, tmp_path: Path
):
    skip_unless_supported(adapter, "classify")
    expect_known_failure(request, adapter, "classify", case)
    result = ADAPTERS[adapter](case, tmp_path)
    assert result.particle_types() == dict(case.types), case.note


@pytest.mark.parametrize("case", CLASSIFICATION, ids=lambda c: c.name)
def test_agglomerate_types(
    request: pytest.FixtureRequest, adapter: str, case: Case, tmp_path: Path
):
    skip_unless_supported(adapter, "classify")
    expect_known_failure(request, adapter, "classify", case)
    result = ADAPTERS[adapter](case, tmp_path)
    assert result.agglomerate_types() == case.agglomerate_types(), case.note
