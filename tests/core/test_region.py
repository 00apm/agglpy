"""The analysed region (``agglpy.core.region``)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from agglpy.core.region import (
    Bounds,
    outside_region,
    region_bounds,
    region_slices,
)
from agglpy.errors import ParamsError, ParticleTableError
from agglpy.params import RegionParams
from agglpy.tables import make_particles


def test_small_example():
    # 10 x 8 image, a quarter cut off: 6 rows kept
    bounds = region_bounds(RegionParams(bottom_fraction=0.25), 10, 8)
    assert bounds == Bounds(-0.5, -0.5, 9.5, 5.5)
    assert bounds.area == 60
    assert region_slices(bounds) == (slice(0, 6), slice(0, 10))


def test_d7_017_numbers():
    # 3072 x 2304 Zeiss image, the usual 0.1: 0.9 * 2304 = 2073.6 -> 2074
    bounds = region_bounds(RegionParams(bottom_fraction=0.1), 3072, 2304)
    assert bounds == Bounds(-0.5, -0.5, 3071.5, 2073.5)
    assert bounds.area == 6_371_328
    assert region_slices(bounds) == (slice(0, 2074), slice(0, 3072))


@pytest.mark.parametrize(
    ("fraction", "height", "rows"),
    [
        (0.8, 10, 2),  # floats give 1.9999999999999996
        (0.55, 1920, 864),  # floats give 863.9999999999999
        (0.34, 25, 17),  # 16.5 rounds up; floats give 16.499999999999996
        (0.5, 5, 3),  # 2.5 rounds up, not to the even 2
        (0.3, 10, 7),
    ],
)
def test_rows_round_to_the_nearest_whole_row(fraction, height, rows):
    bounds = region_bounds(RegionParams(bottom_fraction=fraction), 4, height)
    assert bounds.y_max == rows - 0.5


def test_no_cut_keeps_the_whole_image():
    bounds = region_bounds(RegionParams(), 7, 5)
    assert bounds == Bounds(-0.5, -0.5, 6.5, 4.5)
    assert bounds.area == 35
    image = np.arange(35, dtype=np.uint8).reshape(5, 7)
    rows, cols = region_slices(bounds)
    assert np.array_equal(image[rows, cols], image)


def test_the_cut_is_a_view_of_the_bottom_rows_removed():
    image = np.arange(80, dtype=np.uint8).reshape(8, 10)
    rows, cols = region_slices(
        region_bounds(RegionParams(bottom_fraction=0.25), 10, 8)
    )
    cut = image[rows, cols]
    assert np.shares_memory(cut, image)
    assert np.array_equal(cut, image[:6])


def test_a_cut_without_rows_is_rejected():
    # 0.05 * 5 = 0.25 rows round to none
    with pytest.raises(
        ParamsError, match=r"bottom_fraction=0.95 leaves 0 of 5 rows"
    ):
        region_bounds(RegionParams(bottom_fraction=0.95), 10, 5)


@pytest.mark.parametrize(
    ("width", "height", "name"),
    [
        (0, 5, "width"),
        (10, -1, "height"),
        (10.0, 5, "width"),
        (10, True, "height"),
    ],
)
def test_image_size_must_be_a_positive_integer(width, height, name):
    with pytest.raises(ParamsError, match=name):
        region_bounds(RegionParams(), width, height)


def test_size_from_numpy_integers():
    # image.shape and table values give NumPy integers
    bounds = region_bounds(RegionParams(), np.int64(4), np.int32(3))
    assert bounds == Bounds(-0.5, -0.5, 3.5, 2.5)
    assert type(bounds.x_max) is float


def test_bounds_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Bounds(0, 0, 1, 1).x_min = 2  # type: ignore[misc]


# --- outside_region ---------------------------------------------------

FULL = Bounds(-0.5, -0.5, 3071.5, 2303.5)  # 3072 x 2304, no cut
CUT = Bounds(-0.5, -0.5, 3071.5, 2073.5)  # bottom_fraction 0.1


def test_outside_region_only_where_nothing_overlaps():
    t = make_particles(
        [1500.0, 10.0, -8.0, 3200.0],
        [1000.0, 900.0, 900.0, 900.0],
        [20.0, 25.0, 25.0, 25.0],
    )
    # normal, cut by the edge, centre outside but reaching in, outside
    mask = outside_region(t, FULL)
    assert mask.tolist() == [False, False, False, True]
    assert mask.dtype == bool


def test_touching_the_region_is_not_outside():
    # right edge at 3071.5: x - r = 3071.5 touches, 0.1 px further is
    # outside; the corner (3071.5, 2303.5) lies hypot(3, 4) = 5 = r
    # from the centre: touches
    t = make_particles(
        [3096.5, 3096.6, 3074.5], [900.0, 1500.0, 2307.5], [25.0, 25.0, 5.0]
    )
    assert outside_region(t, FULL).tolist() == [False, True, False]


def test_circles_on_the_data_bar_are_outside_the_cut_region():
    # bottom edge at 2073.5: 2200 lies 126.5 below (> 20), 2085 only
    # 11.5 (reaches in), 2093.5 exactly 20 (touches)
    t = make_particles(
        [1500.0, 1500.0, 1500.0], [2200.0, 2085.0, 2093.5], [20.0] * 3
    )
    assert outside_region(t, CUT).tolist() == [True, False, False]
    assert outside_region(t, FULL).tolist() == [False, False, False]


def test_outside_region_mask_follows_the_table_index():
    t = make_particles([10.0, 5000.0], [10.0, 10.0], [5.0, 5.0])
    t.index = [7, 3]
    mask = outside_region(t, Bounds(-0.5, -0.5, 99.5, 99.5))
    assert mask.index.tolist() == [7, 3]
    assert t.loc[~mask, "id"].tolist() == [1]


def test_outside_region_of_an_empty_table():
    mask = outside_region(make_particles([], [], []), FULL)
    assert len(mask) == 0 and mask.dtype == bool


def test_outside_region_names_missing_columns():
    with pytest.raises(ParticleTableError, match="'r'"):
        outside_region(pd.DataFrame({"x": [1.0], "y": [1.0]}), FULL)
