"""Particle and agglomerate properties (``agglpy.core.properties``).

The 1.4 synthetic cases (``support.synthetic.cases.PROPERTIES``, values
worked out by hand) run once per implementation; the unit tests below
run the core directly, on cases worked out in the 2.4 plan.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.properties import agglomerate_properties, particle_properties
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


def _one(x, y, r) -> pd.Series:
    """Properties of the only agglomerate of these circles."""
    table = agglomerate_properties(_particles(x, y, r))
    assert len(table) == 1
    return table.iloc[0]


# Spec example: A r=10, B r=4 touching A's outline, C r=2 inside A.
ABC = ([0, 14, 2], [0, 0, 0], [10, 4, 2])


def test_one_circle_sizes():
    p = _one([3], [4], [10])
    assert p["member_count"] == 1
    assert p["enclosed_count"] == 0
    assert p["volume"] == pytest.approx(sphere(10), rel=1e-12)
    assert p["D"] == pytest.approx(20, rel=1e-12)
    assert p["D_mean"] == 20
    assert math.isnan(p["D_std"])
    assert p["D_largest"] == 20
    assert math.isnan(p["size_ratio"])
    assert p["volume_with_hidden"] == p["volume"]
    assert p["D_with_hidden"] == p["D"]
    assert p["member_count_with_hidden"] == 1


def test_example_sizes():
    p = _one(*ABC)
    assert p["member_count"] == 3
    assert p["enclosed_count"] == 1
    assert p["volume"] == pytest.approx(sphere(10) + sphere(4) + sphere(2))
    assert p["volume"] == pytest.approx(4490.4, abs=0.1)
    assert p["D"] == pytest.approx(2 * 1072 ** (1 / 3))  # r³ sum
    assert p["D"] == pytest.approx(20.47, abs=0.01)
    assert p["D_mean"] == pytest.approx(32 / 3)  # (20 + 8 + 4) / 3
    # deviations 28/3, -8/3, -20/3 from the mean; n - 1 = 2
    assert p["D_std"] == pytest.approx(math.sqrt(1248 / 9 / 2))
    assert p["D_largest"] == 20
    assert p["size_ratio"] == 0.4  # 8 / 20
    assert p["member_count_with_hidden"] == 4
    assert p["volume_with_hidden"] == pytest.approx(p["volume"] + sphere(2))
    assert p["volume_with_hidden"] == pytest.approx(4523.9, abs=0.1)
    assert p["D_with_hidden"] == pytest.approx(2 * 1080 ** (1 / 3))


def test_equal_members_have_zero_std():
    assert _one([0, 20], [0, 0], [10, 10])["D_std"] == 0


def test_one_row_per_agglomerate_sorted_by_id():
    table = pd.DataFrame(
        {
            "x": [0.0, 100.0, 14.0],
            "y": [0.0, 0.0, 0.0],
            "r": [10.0, 3.0, 4.0],
            "agglomerate_id": [7, 3, 7],
            "enclosed": [False, False, False],
        },
        index=[5, 6, 7],
    )
    out = agglomerate_properties(table)
    assert out.columns[0] == "agglomerate_id"
    assert out["agglomerate_id"].tolist() == [3, 7]
    assert out["member_count"].tolist() == [1, 2]
    assert list(out.index) == [0, 1]


def test_agglomerate_properties_does_not_modify_its_input():
    table = _particles(*ABC)
    before = table.copy()
    agglomerate_properties(table)
    pd.testing.assert_frame_equal(table, before)


def test_agglomerate_properties_needs_its_columns():
    table = _particles(*ABC).drop(columns="enclosed")
    with pytest.raises(ParticleTableError, match="enclosed"):
        agglomerate_properties(table)


def test_empty_table_has_every_column_and_dtype():
    empty = agglomerate_properties(_particles([], [], []))
    one = agglomerate_properties(_particles([0], [0], [1]))
    assert len(empty) == 0
    pd.testing.assert_series_equal(empty.dtypes, one.dtypes)
    for name in (
        "agglomerate_id",
        "member_count",
        "enclosed_count",
        "member_count_with_hidden",
    ):
        assert empty[name].dtype == np.int64


def test_one_circle_centre_of_mass_and_rg():
    p = _one([3], [4], [10])
    assert (p["x_com"], p["y_com"]) == (3, 4)
    assert p["rg"] == pytest.approx(math.sqrt(3 / 5) * 10, rel=1e-12)


def test_two_equal_circles_centre_of_mass_and_rg():
    p = _one([0, 20], [0, 0], [10, 10])
    assert (p["x_com"], p["y_com"]) == (10, 0)
    # each sphere: d = 10 from the CoM, plus its own (3/5) r²
    assert p["rg"] == pytest.approx(math.sqrt(100 + 60), rel=1e-12)


def test_example_centre_of_mass_and_rg():
    p = _one(*ABC)
    xc = (14 * 64 + 2 * 8) / 1072  # masses r³ = 1000, 64, 8
    assert p["x_com"] == pytest.approx(xc, rel=1e-12)
    assert p["y_com"] == 0
    inertia = (
        1000 * (xc**2 + 3 / 5 * 100)
        + 64 * ((14 - xc) ** 2 + 3 / 5 * 16)
        + 8 * ((2 - xc) ** 2 + 3 / 5 * 4)
    )
    assert p["rg"] == pytest.approx(math.sqrt(inertia / 1072), rel=1e-12)
