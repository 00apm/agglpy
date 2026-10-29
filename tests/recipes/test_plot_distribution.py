"""The distribution plot recipe (``examples/plot_distribution.py``)."""

import runpy

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from agglpy.core.distributions import distribution, size_classes

from examples.plot_distribution import plot_distribution
from support.paths import EXAMPLES_DIR

EDGES = size_classes(1, 100, count=4, scale="log")


def _table() -> pd.DataFrame:
    images = pd.DataFrame({"image": ["a", "b"], "series": ["x", "y"]})
    values = pd.DataFrame(
        {"image": ["a", "a", "a", "b", "b"], "D": [2, 5, 40, 3, 30]}
    )
    return distribution(images, values, "D", EDGES, by="series")


def test_bars_have_true_widths_and_groups() -> None:
    table = _table()
    ax = Figure().add_subplot()
    plot_distribution(table, ax=ax, by="series")
    widths = [bar.get_width() for bar in ax.patches]
    assert len(widths) == 2 * (len(EDGES) - 1)  # two groups
    np.testing.assert_allclose(widths, table["width"])
    assert ax.get_xscale() == "log"
    # bar height: the fraction per decade, so areas compare on a log axis
    heights = [bar.get_height() for bar in ax.patches]
    np.testing.assert_allclose(heights, table["log_density"])
    (twin,) = (a for a in ax.figure.axes if a is not ax)
    # one cumulative curve per group, from 0 to 1
    assert [line.get_ydata()[[0, -1]].tolist() for line in twin.lines] == [
        [0, 1],
        [0, 1],
    ]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["x", "y"]


def test_recipe_example_runs_top_to_bottom() -> None:
    names = runpy.run_path(
        str(EXAMPLES_DIR / "plot_distribution.py"), run_name="__main__"
    )
    assert isinstance(names["fig"], Figure)
    assert set(names["dist"]["series"]) == {"low flow", "high flow"}
