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

from agglpy.core import properties
from agglpy.core.agglomerates import find_agglomerates, find_contacts
from agglpy.core.properties import (
    DIMENSIONS,
    agglomerate_properties,
    particle_properties,
)
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


def _bare(**columns) -> pd.DataFrame:
    """Two agglomerates (A + B touching, C alone), columns overridable."""
    table = {
        "x": [0.0, 14.0, 100.0],
        "y": [0.0, 0.0, 0.0],
        "r": [10.0, 4.0, 5.0],
        "agglomerate_id": [0, 0, 1],
        "enclosed": [False, False, False],
    }
    table.update(columns)
    return pd.DataFrame(table)


@pytest.mark.parametrize(
    "enclosed",
    [
        pd.Series([False, np.nan, False], dtype=object),  # NaN -> True
        ["False", "False", "True"],  # any non-empty string -> True
        [0, 1, 0],
        pd.Series([False, pd.NA, False], dtype="boolean"),
    ],
    ids=["nan", "strings", "ints", "missing"],
)
def test_agglomerate_properties_rejects_a_non_boolean_enclosed(enclosed):
    with pytest.raises(ParticleTableError, match="'enclosed'"):
        agglomerate_properties(_bare(enclosed=enclosed))


def test_agglomerate_properties_accepts_python_bools_in_an_object_column():
    enclosed = pd.Series([False, True, False], dtype=object)
    out = agglomerate_properties(_bare(enclosed=enclosed))
    assert out["enclosed_count"].tolist() == [1, 0]


@pytest.mark.parametrize(
    "ids", [[0.0, 0.5, 1.0], [0, np.nan, 1], ["a", "a", "b"]]
)
def test_agglomerate_properties_rejects_non_integer_ids(ids):
    with pytest.raises(ParticleTableError, match="'agglomerate_id'"):
        agglomerate_properties(_bare(agglomerate_id=ids))


def test_agglomerate_properties_accepts_integral_float_ids():
    out = agglomerate_properties(_bare(agglomerate_id=[0.0, 0.0, 1.0]))
    assert out["agglomerate_id"].tolist() == [0, 1]
    assert out["agglomerate_id"].dtype == np.int64


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("r", [10.0, np.nan, 5.0], "'r' must be finite"),
        ("r", [10.0, -4.0, 5.0], "'r' must be > 0"),
        ("r", [10.0, 0.0, 5.0], "'r' must be > 0"),
        ("x", [0.0, np.inf, 100.0], "'x' must be finite"),
        ("y", ["0", "a", "0"], "'y' must be numeric"),
    ],
)
def test_agglomerate_properties_rejects_bad_geometry(column, values, message):
    with pytest.raises(ParticleTableError, match=message):
        agglomerate_properties(_bare(**{column: values}))


@pytest.mark.parametrize(
    ("r", "message"),
    [([1.0, np.nan], "'r' must be finite"), ([1.0, -1.0], "'r' must be > 0")],
)
def test_particle_properties_rejects_bad_radii(r, message):
    with pytest.raises(ParticleTableError, match=message):
        particle_properties(pd.DataFrame({"r": r}))


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


def test_one_circle_area():
    p = _one([3], [4], [10])
    assert p["area"] == pytest.approx(100 * math.pi, rel=1e-12)
    assert p["D_pa"] == pytest.approx(20, rel=1e-12)


def test_tangent_circles_area_is_the_sum():
    p = _one([0, 14], [0, 0], [10, 4])
    assert p["area"] == pytest.approx(116 * math.pi, rel=1e-12)


def test_overlapping_circles_area_counts_the_lens_once():
    p = _one([0, 10], [0, 0], [10, 10])
    # lens of two circles r = 10 at distance 10
    lens = 2 * 100 * math.acos(10 / 20) - 5 * math.sqrt(400 - 100)
    assert p["area"] == pytest.approx(200 * math.pi - lens, rel=1e-12)


def test_enclosed_circle_adds_no_area():
    p = _one(*ABC)
    assert p["area"] == pytest.approx(116 * math.pi, rel=1e-12)
    assert p["D_pa"] == pytest.approx(math.sqrt(464), rel=1e-12)


def test_identical_circles_area_counts_once():
    p = _one([5, 5], [5, 5], [8, 8])
    assert p["area"] == pytest.approx(64 * math.pi, rel=1e-12)


def test_circle_covered_by_two_others_adds_no_area():
    # C lies inside the union of A and B, but inside neither alone.
    alone = _one([0, 12], [0, 0], [10, 10])
    p = _one([0, 12, 6], [0, 0, 0], [10, 10, 7.5])
    assert p["area"] == pytest.approx(alone["area"], rel=1e-12)


def test_area_does_not_depend_on_position():
    near = _one([0, 10, 3], [0, 0, 9], [10, 10, 4])
    far = _one([5000, 5010, 5003], [7000, 7000, 7009], [10, 10, 4])
    assert far["area"] == pytest.approx(near["area"], rel=1e-12)


def test_nearly_tangent_circles_area():
    # 1e-9 px of overlap: the arc half-angle is close to 0; rounding
    # must not give NaN or a jump.
    p = _one([0, 14 - 1e-9], [0, 0], [10, 4])
    assert p["area"] == pytest.approx(116 * math.pi, rel=1e-9)


@pytest.mark.parametrize("gap", [0.0, 1e-9])
def test_nearly_internally_tangent_circle_area(gap: float):
    # B (r = 4) touches A's outline from inside (gap 0: enclosed) or
    # pokes 1e-9 px out of it: no NaN, no jump from A's area.
    p = _one([0, 6 + gap], [0, 0], [10, 4])
    assert p["area"] == pytest.approx(100 * math.pi, rel=1e-9)


def test_contact_search_gets_image_positions(monkeypatch):
    # Review Focus 1: centres shifted to each agglomerate's centre of
    # mass would pile every agglomerate onto the origin. The area stays
    # right but the contact search gets ~1000x slower, so only a check
    # of the call itself catches it.
    seen = []

    def spy(particles: pd.DataFrame):
        seen.append(particles[["x", "y"]].to_numpy().copy())
        return find_contacts(particles)

    monkeypatch.setattr(properties, "find_contacts", spy)
    table = _particles([0, 14, 500], [0, 0, 300], [10, 4, 5])
    agglomerate_properties(table)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], table[["x", "y"]].to_numpy())


def _pixel_area(x, y, r, h: float) -> float:
    """Union area counted on a grid of spacing h (cell centres)."""
    x, y, r = map(np.asarray, (x, y, r))
    gx = np.arange((x - r).min() + h / 2, (x + r).max(), h)
    gy = np.arange((y - r).min() + h / 2, (y + r).max(), h)
    px, py = np.meshgrid(gx, gy)
    inside = np.zeros(px.shape, dtype=bool)
    for xi, yi, ri in zip(x, y, r, strict=True):
        inside |= (px - xi) ** 2 + (py - yi) ** 2 <= ri * ri
    return float(inside.sum()) * h * h


@pytest.mark.parametrize("seed", range(5))
def test_area_matches_a_pixel_count(seed: int):
    rng = np.random.default_rng(seed)
    x, y, r = (
        rng.uniform(0, 30, 12),
        rng.uniform(0, 30, 12),
        rng.uniform(2, 8, 12),
    )
    # circles that overlap share an agglomerate, so areas add up
    total = agglomerate_properties(_particles(x, y, r))["area"].sum()
    assert total == pytest.approx(_pixel_area(x, y, r, 0.05), rel=1e-3)


def test_area_of_a_ring_leaves_out_the_hole():
    angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    x, y, r = 50 * np.cos(angle), 50 * np.sin(angle), np.full(24, 8.0)
    p = _one(x, y, r)
    assert p["area"] == pytest.approx(_pixel_area(x, y, r, 0.05), rel=1e-3)


def test_one_circle_feret():
    p = _one([3], [4], [10])
    assert (p["D_feret_x"], p["D_feret_y"], p["D_feret_max"]) == (20, 20, 20)


def test_example_feret():
    p = _one(*ABC)
    assert (p["D_feret_x"], p["D_feret_y"], p["D_feret_max"]) == (28, 20, 28)


def test_diagonal_pair_feret():
    p = _one([0, 6], [0, 8], [10, 10])  # d = 10
    assert p["D_feret_x"] == 26  # 16 - (-10)
    assert p["D_feret_y"] == 28  # 18 - (-10)
    assert p["D_feret_max"] == 30  # d + r + r


def test_long_chain_feret_max():
    # 100 members: the large-agglomerate path (pruning)
    n = 100
    p = _one(2 * np.arange(n), np.zeros(n), np.ones(n))
    assert p["member_count"] == n
    assert p["D_feret_max"] == 2 * n
    assert (p["D_feret_x"], p["D_feret_y"]) == (2 * n, 2)


def _brute_feret_max(table: pd.DataFrame) -> list[float]:
    out = []
    for _, members in table.groupby("agglomerate_id"):
        x, y, r = (members[c].to_numpy() for c in ("x", "y", "r"))
        span = np.hypot(x[:, None] - x, y[:, None] - y) + r[:, None] + r
        out.append(float(span.max()))
    return out


@pytest.mark.parametrize("seed", range(3))
def test_feret_max_matches_brute_force(seed: int):
    rng = np.random.default_rng(seed)
    n = 1500
    side = 7 * math.sqrt(n)  # dense: many small, some large agglomerates
    table = _particles(
        rng.uniform(0, side, n),
        rng.uniform(0, side, n),
        rng.lognormal(np.log(8), 0.35, n) / 2,
    )
    out = agglomerate_properties(table)
    assert out["member_count"].max() > 64, "needs a large agglomerate"
    expected = _brute_feret_max(table)
    assert out["D_feret_max"].tolist() == pytest.approx(expected, rel=1e-12)


def test_columns_in_spec_order():
    out = agglomerate_properties(_particles([0], [0], [1]))
    assert list(out.columns) == ["agglomerate_id", *DIMENSIONS]


def test_dimensions_cover_exactly_the_property_columns():
    table = _particles(*ABC)
    added = set(particle_properties(table).columns) - set(table.columns)
    agglomerate = set(agglomerate_properties(table).columns)
    assert set(DIMENSIONS) == added | (agglomerate - {"agglomerate_id"})


def test_dimensions_are_read_only():
    with pytest.raises(TypeError):
        DIMENSIONS["D"] = 2  # type: ignore[index]
