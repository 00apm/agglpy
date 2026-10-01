"""Agglomerate properties: volume, volume-equivalent D, member
statistics and the dsom values.

Definitions and cases in ``support.synthetic.cases.PROPERTIES``, values
worked out by hand. Centre of mass and radius of gyration are not
checked here (their definition is an open question; the golden tests
keep today's values). Phase 2 turns this file into the unit tests of
``agglpy.properties``.
"""

import math
from pathlib import Path

import pytest

from support.synthetic.adapters import ADAPTERS, REAL_PIXEL_SIZE
from support.synthetic.cases import PROPERTIES, Case

# Small enough to see members that add 3.75e-10 of the volume (the
# polydisperse case); computing in metres and converting back costs only
# a few ulps.
RTOL = 1e-12

# The properties are compared in px; running with a real pixel size as
# well checks that every value is converted with the right power of it.
PIXEL_SIZES = [1.0, REAL_PIXEL_SIZE]


@pytest.mark.parametrize("pixel_size", PIXEL_SIZES, ids=lambda v: f"px{v:g}")
@pytest.mark.parametrize("case", PROPERTIES, ids=lambda c: c.name)
def test_agglomerate_properties(
    adapter: str, case: Case, pixel_size: float, tmp_path: Path
):
    result = ADAPTERS[adapter](case, tmp_path, pixel_size=pixel_size)
    actual = result.agglomerate_properties()
    for members, expected in case.properties.items():
        for name, value in expected.items():
            got = actual[members][name]
            if math.isnan(value):
                assert math.isnan(got), f"{case.name}: {name}"
            else:
                assert got == pytest.approx(value, rel=RTOL, abs=0), (
                    f"{case.name}: {name}"
                )
