"""The agglomerate-types recipe (``examples/agglomerate_types.py``)."""

import math
import runpy

import pandas as pd
import pytest

from examples.agglomerate_types import agglomerate_types, type_counts
from support.paths import EXAMPLES_DIR


@pytest.mark.parametrize(
    ("member_count", "size_ratio", "threshold", "expected"),
    [
        (1, math.nan, 0.5, "separate"),
        (2, 0.5, 0.5, "collector"),  # at the threshold: collector
        (3, 0.4, 0.5, "collector"),
        (2, 0.6, 0.5, "similar"),
        (2, 0.4, 0.0, "similar"),  # threshold 0: never a collector
        (2, 1.0, 0.8, "similar"),  # equal sizes
    ],
)
def test_rule(member_count, size_ratio, threshold, expected):
    table = pd.DataFrame(
        {"member_count": [member_count], "size_ratio": [size_ratio]}
    )
    assert agglomerate_types(table, threshold).tolist() == [expected]


def test_type_counts_per_image():
    table = pd.DataFrame(
        {
            "image": ["a", "a", "a", "b"],
            "member_count": [1, 2, 2, 1],
            "size_ratio": [math.nan, 0.3, 0.9, math.nan],
        }
    )
    counts = type_counts(table, threshold=0.5)
    assert counts.loc["a"].to_dict() == {
        "collector": 1,
        "similar": 1,
        "separate": 1,
    }
    # a type missing in an image is a zero, not a missing column
    assert counts.loc["b"].to_dict() == {
        "collector": 0,
        "similar": 0,
        "separate": 1,
    }


def test_recipe_example_runs_top_to_bottom():
    # Runs the file as a script would (all cells in order) and checks
    # the table its last cell shows.
    names = runpy.run_path(
        str(EXAMPLES_DIR / "agglomerate_types.py"), run_name="__main__"
    )
    assert names["counts"].to_dict(orient="index") == {
        "img1": {"collector": 1, "similar": 1, "separate": 1},
        "img2": {"collector": 0, "similar": 0, "separate": 2},
    }
