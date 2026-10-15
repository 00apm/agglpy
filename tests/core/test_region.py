"""The analysed region (``agglpy.core.region``)."""

import dataclasses

import numpy as np
import pytest

from agglpy.core.region import Bounds, region_bounds, region_slices
from agglpy.errors import ParamsError
from agglpy.params import RegionParams


def test_small_example():
    # 10 x 8 image, a quarter cut off: 6 rows kept
    bounds = region_bounds(RegionParams(bottom_fraction=0.25), 10, 8)
    assert bounds == Bounds(-0.5, -0.5, 9.5, 5.5)
    assert bounds.area == 60
    assert region_slices(bounds) == (slice(0, 6), slice(0, 10))


def test_d7_017_numbers():
    # 3072 x 2304 Zeiss image, the usual 0.1: int(0.9 * 2304) = 2073
    bounds = region_bounds(RegionParams(bottom_fraction=0.1), 3072, 2304)
    assert bounds == Bounds(-0.5, -0.5, 3071.5, 2072.5)
    assert bounds.area == 6_368_256
    assert region_slices(bounds) == (slice(0, 2073), slice(0, 3072))


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
    with pytest.raises(
        ParamsError, match=r"bottom_fraction=0.9 leaves 0 of 5 rows"
    ):
        region_bounds(RegionParams(bottom_fraction=0.9), 10, 5)


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
