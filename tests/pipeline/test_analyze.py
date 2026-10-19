"""Analysis of one image's particles (``agglpy.pipeline.analyze``)."""

import dataclasses
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from agglpy.core.region import Bounds
from agglpy.errors import (
    ParamsError,
    ParticlesOutsideRegionWarning,
    ParticleTableError,
)
from agglpy.params import RegionParams
from agglpy.pipeline import ImageResult, analyze
from agglpy.tables import make_particles

# Image 100 x 80 px, a quarter cut off: rows 0..59 kept, so the region
# is [-0.5, 99.5] x [-0.5, 59.5], area 100 * 60 = 6000 px².
REGION = RegionParams(bottom_fraction=0.25)
SIZE = {"width": 100, "height": 80}


def _table() -> pd.DataFrame:
    # 1 inside; 2 in contact with 1 (8 <= 5 + 4); 3 cut by the right
    # edge (97 + 4 >= 99.5); 4 on the data bar, 10.5 below the edge
    # (> 5): dropped; 5 2.5 below the edge, reaches in: kept, cut;
    # 4 and 5 touch, so dropping 4 also splits that agglomerate
    return make_particles(
        [20.0, 28.0, 97.0, 50.0, 50.0],
        [20.0, 20.0, 30.0, 70.0, 62.0],
        [5.0, 4.0, 4.0, 5.0, 5.0],
    )


def _all_outside() -> pd.DataFrame:
    # 12 small circles on the data bar, ids 1..12
    k = np.arange(12)
    return make_particles(np.full(12, 50.0), 70.0 + 3 * k, np.ones(12))


def _analyze_quietly(table: pd.DataFrame, **kwargs: object) -> ImageResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ParticlesOutsideRegionWarning)
        return analyze(table, **kwargs)  # type: ignore[arg-type]


def test_without_a_size_nothing_is_dropped_or_flagged():
    res = analyze(_table())
    assert len(res.particles) == 5
    # 1 + 2, 3, and 4 + 5 (8 apart <= 5 + 5)
    assert len(res.agglomerates) == 3
    assert res.outside_ids == ()
    assert "touches_edge" not in res.particles.columns
    assert "touches_edge" not in res.agglomerates.columns
    assert res.fov_area is None and res.bounds is None
    assert res.width is None and res.height is None
    assert res.region == RegionParams()


def test_with_a_size_the_outside_circle_is_dropped_and_warned():
    with pytest.warns(ParticlesOutsideRegionWarning) as record:
        res = analyze(_table(), **SIZE, region=REGION)
    assert len(record) == 1
    assert str(record[0].message) == (
        "1 of 5 particles lie entirely outside the analysed region "
        "(x -0.5 to 99.5, y -0.5 to 59.5 px) and were left out: ids 4; "
        "widen the region if they are real"
    )
    assert res.outside_ids == (4,)
    assert res.particles["id"].tolist() == [1, 2, 3, 5]
    assert res.particles["touches_edge"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert len(res.agglomerates) == 3
    assert res.agglomerates["touches_edge"].sum() == 2
    assert res.fov_area == 6000.0
    assert res.bounds == Bounds(-0.5, -0.5, 99.5, 59.5)
    assert res.width == 100 and res.height == 80
    assert res.region == REGION


def test_the_warning_points_at_the_caller():
    with pytest.warns(ParticlesOutsideRegionWarning) as record:
        analyze(_table(), **SIZE, region=REGION)
    assert record[0].filename == __file__


def test_nothing_outside_gives_no_warning():
    table = _table()
    table = table[table["id"] != 4]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = analyze(table, **SIZE, region=REGION)
    assert res.outside_ids == ()


def test_the_warning_lists_ten_ids_and_counts_the_rest():
    with pytest.warns(ParticlesOutsideRegionWarning) as record:
        analyze(_all_outside(), **SIZE, region=REGION)
    message = str(record[0].message)
    assert "12 of 12 particles" in message
    assert "ids 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 and 2 more;" in message


def test_all_dropped_gives_empty_tables_with_their_columns():
    full = _analyze_quietly(_table(), **SIZE, region=REGION)
    res = _analyze_quietly(_all_outside(), **SIZE, region=REGION)
    assert res.particles.empty and res.agglomerates.empty
    assert res.particles.columns.tolist() == full.particles.columns.tolist()
    assert (
        res.agglomerates.columns.tolist() == full.agglomerates.columns.tolist()
    )
    assert res.outside_ids == tuple(range(1, 13))
    assert res.fov_area == 6000.0


def test_zero_particles():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = analyze(make_particles([], [], []), **SIZE)
    assert res.particles.empty and res.agglomerates.empty
    assert "touches_edge" in res.particles.columns
    assert res.outside_ids == ()
    assert res.fov_area == 8000.0


def test_input_table_is_not_modified():
    table = _table()
    table.index = [10, 11, 12, 13, 14]
    copy = table.copy()
    _analyze_quietly(table, **SIZE, region=REGION)
    pd.testing.assert_frame_equal(table, copy)


def test_outside_ids_are_ids_not_index_labels():
    table = _table()
    table.index = [40, 30, 20, 10, 0]
    res = _analyze_quietly(table, **SIZE, region=REGION)
    assert res.outside_ids == (4,)


def test_reanalysis_is_stable():
    res = _analyze_quietly(_table(), **SIZE, region=REGION)
    again = analyze(res.particles, **SIZE, region=REGION)
    pd.testing.assert_frame_equal(again.particles, res.particles)
    pd.testing.assert_frame_equal(again.agglomerates, res.agglomerates)


def test_reanalysis_without_a_size_drops_touches_edge():
    # a touches_edge from an earlier run with a size would be stale
    res = _analyze_quietly(_table(), **SIZE, region=REGION)
    again = analyze(res.particles)
    assert "touches_edge" not in again.particles.columns
    assert "touches_edge" not in again.agglomerates.columns


def test_numpy_integer_sizes_are_accepted():
    res = analyze(_table(), width=np.int64(100), height=np.int64(80))
    assert res.width == 100 and type(res.width) is int
    assert res.height == 80 and type(res.height) is int


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"width": 100}, "both width and height"),
        ({"height": 80}, "both width and height"),
        ({"region": RegionParams(bottom_fraction=0.1)}, "region needs"),
        ({"width": True, "height": 80}, "width"),
        ({"width": 100.0, "height": 80}, "width"),
        ({"width": 0, "height": 80}, "width"),
    ],
)
def test_bad_sizes_raise(kwargs, match):
    with pytest.raises(ParamsError, match=match):
        analyze(_table(), **kwargs)


def test_a_bad_table_raises():
    with pytest.raises(ParticleTableError):
        analyze(pd.DataFrame({"x": [1.0], "y": [1.0]}))


def test_result_survives_a_pickle_round_trip():
    res = _analyze_quietly(_table(), **SIZE, region=REGION)
    back = pickle.loads(pickle.dumps(res))
    pd.testing.assert_frame_equal(back.particles, res.particles)
    pd.testing.assert_frame_equal(back.agglomerates, res.agglomerates)
    for field in ("outside_ids", "width", "height", "region", "bounds"):
        assert getattr(back, field) == getattr(res, field)
    assert back.fov_area == res.fov_area


def test_result_is_frozen():
    res = analyze(_table())
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.fov_area = 1.0  # type: ignore[misc]
