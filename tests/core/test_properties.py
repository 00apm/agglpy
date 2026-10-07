"""Particle and agglomerate properties (``agglpy.core.properties``).

The 1.4 synthetic cases (``support.synthetic.cases.PROPERTIES``, values
worked out by hand) run once per implementation; the unit tests below
run the core directly, on cases worked out in the 2.4 plan.
"""

import math
import warnings
from pathlib import Path

import pandas as pd
import pytest

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.properties import particle_properties
from agglpy.errors import DuplicateParticlesWarning, ParticleTableError
from agglpy.tables import make_particles

from support.synthetic.adapters import (
    ADAPTERS,
    REAL_PIXEL_SIZE,
    skip_unless_supported,
)
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
    skip_unless_supported(adapter, "properties")
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


# ---------------------------------------------------------------------
# Unit tests of agglpy.core.properties (one run, no adapter)
# ---------------------------------------------------------------------


def _particles(x, y, r) -> pd.DataFrame:
    """Particles grouped into agglomerates; duplicates allowed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DuplicateParticlesWarning)
        return find_agglomerates(make_particles(x, y, r))


def sphere(r: float) -> float:
    return 4 / 3 * math.pi * r**3


def test_particle_properties_values():
    out = particle_properties(_particles([0, 100], [0, 0], [1, 2.5]))
    assert out["D"].tolist() == [2, 5]
    assert out["area"].tolist() == [math.pi, math.pi * 2.5**2]
    assert out["volume"].tolist() == [sphere(1), sphere(2.5)]


def test_particle_properties_keeps_columns_rows_and_index():
    table = _particles([0, 100, 200], [0, 0, 0], [1, 2, 3])
    table["composition"] = ["Fe", "Si", "Fe"]
    table.index = [30, 10, 20]
    before = table.copy()
    out = particle_properties(table)
    pd.testing.assert_frame_equal(table, before)  # input untouched
    assert list(out.columns) == [*table.columns, "D", "area", "volume"]
    assert list(out.index) == [30, 10, 20]
    pd.testing.assert_frame_equal(out[table.columns], table)


def test_particle_properties_rerun_replaces_its_columns():
    once = particle_properties(_particles([0], [0], [1]))
    once["r"] = 2.0  # e.g. a corrected radius
    again = particle_properties(once)
    assert list(again.columns) == list(once.columns)
    assert again.loc[0, "D"] == 4
    pd.testing.assert_frame_equal(particle_properties(again), again)


def test_particle_properties_needs_r():
    with pytest.raises(ParticleTableError, match="'r'"):
        particle_properties(pd.DataFrame({"x": [0.0]}))
