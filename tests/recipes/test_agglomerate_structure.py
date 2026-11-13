"""The structure recipe (``examples/agglomerate_structure.py``)."""

import math
import runpy

import numpy as np
import pandas as pd
import pytest

from examples.agglomerate_structure import (
    branch_statistics,
    dangling_end_statistics,
    fill,
    relative_lengths,
    volume_share_beyond,
)
from support.paths import EXAMPLES_DIR


def _brush() -> pd.DataFrame:
    """Agglomerate 0 of image a: a base, 10 single particles on it
    (tips of depth 1) and 2 dendrites of 5 (tips of depth 5); agglomerate
    1: a pair (one tip of depth 1). Every particle has D = 2, volume 1.
    """
    depth = [0] + [1] * 10 + [1, 2, 3, 4, 5] * 2 + [0, 1]
    tip = [False] + [True] * 10 + [False, False, False, False, True] * 2
    tip += [False, True]
    n = len(depth)
    return pd.DataFrame(
        {
            "image": "a",
            "id": range(n),
            "agglomerate_id": [0] * (n - 2) + [1, 1],
            "branch_depth": depth,
            "branch_length": [2.0 * d for d in depth],
            "tip": tip,
            "volume": 1.0,
        }
    )


def test_branch_statistics_with_and_without_the_filter():
    table = _brush()
    every = branch_statistics(table, min_depth=0).loc[("a", 0)]
    assert every["branches"] == 12
    assert every["branch_depth_mean"] == pytest.approx(20 / 12)
    assert every["branch_depth_median"] == 1
    deep = branch_statistics(table).loc[("a", 0)]
    assert deep["branches"] == 2
    assert (deep["branch_depth_mean"], deep["branch_depth_median"]) == (5, 5)
    assert deep["branch_length_mean"] == 10
    assert deep["branch_length_median"] == 10


def test_no_deep_branch_gives_nan_not_zero():
    pair = branch_statistics(_brush()).loc[("a", 1)]
    assert pair["branches"] == 0
    assert math.isnan(pair["branch_depth_mean"])
    assert math.isnan(pair["branch_length_median"])


def test_dangling_end_statistics():
    table = pd.DataFrame(
        {
            "image": "a",
            "id": range(7),
            "agglomerate_id": [0] * 6 + [1],
            "segment_type": ["core", "dangling end", "dangling end"]
            + ["dangling end"] * 3
            + ["no core"],
            "segment_id": [0, 1, 1, 2, 2, 2, 3],
            "dangling_end_depth": [np.nan, 1, 2, 1, 2, 3, np.nan],
            "dangling_end_length": [np.nan, 2, 4, 2, 4, 6, np.nan],
        }
    )
    ends = dangling_end_statistics(table, min_depth=0).loc[("a", 0)]
    assert ends["dangling_ends"] == 2
    assert ends["dangling_end_depth_mean"] == 2.5
    assert ends["dangling_end_length_median"] == 5
    deep = dangling_end_statistics(table, min_depth=3).loc[("a", 0)]
    assert deep["dangling_ends"] == 1
    assert deep["dangling_end_depth_mean"] == 3
    none = dangling_end_statistics(table).loc[("a", 1)]
    assert none["dangling_ends"] == 0
    assert math.isnan(none["dangling_end_depth_mean"])


def test_volume_share_beyond():
    share = volume_share_beyond(_brush())
    # depth >= 2: 4 particles of each dendrite, of 21
    assert share[("a", 0)] == pytest.approx(8 / 21)
    assert share[("a", 1)] == 0


def test_relative_lengths_and_fill():
    table = pd.DataFrame(
        {
            "longest_chain_length": [6.0, 2.0],
            "branch_length_max": [4.0, 0.0],
            "D_largest": [2.0, 2.0],
            "area": [3.0, math.pi],
            "D_feret_max": [6.0, 2.0],
        }
    )
    ratios = relative_lengths(table)
    assert ratios.columns.tolist() == [
        "longest_chain_per_D_largest",
        "branch_length_max_per_D_largest",
    ]
    assert ratios.iloc[0].tolist() == [3, 2]
    # one circle fills its Feret circle
    assert fill(table).tolist() == pytest.approx([3 / (math.pi * 9), 1])


def test_recipe_example_runs_top_to_bottom():
    names = runpy.run_path(
        str(EXAMPLES_DIR / "agglomerate_structure.py"), run_name="__main__"
    )
    branches = names["branches"]
    assert branches.loc[("img1", 0)].tolist()[:3] == [2, 3, 3]
    assert branches.loc[("img1", 1), "branches"] == 0
    assert names["share"][("img1", 0)] == pytest.approx(4 / 11)
