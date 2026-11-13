"""draw.color_scale: one colour scale over the values of many images."""

import matplotlib
import pandas as pd
import pytest
from matplotlib.colors import LogNorm

from agglpy import draw
from agglpy.errors import ParamsError, TableError

TAB10 = matplotlib.colormaps["tab10"]


def drawn(column: str, values: list, scale: dict) -> draw.Drawing:
    """Two separate particles with ``values``, drawn with ``scale``."""
    p = pd.DataFrame(
        {"x": [30.0, 70.0], "y": [40.0, 40.0], "r": [10.0, 10.0]}
    ).assign(**{column: values})
    _, ax = draw.figure((80, 100))
    return draw.particles(ax, p, color_by=column, **scale)


def test_numbers_and_labels_keep_the_scale_of_all_images():
    numbers = pd.Series([1.0, 9.0, 4.0], name="n")
    scale = draw.color_scale(numbers)
    assert set(scale) == {"cmap", "norm"}
    assert (scale["norm"].vmin, scale["norm"].vmax) == (1.0, 9.0)
    # an image whose own values are 2 and 3 keeps the range 1 ... 9
    bar = drawn("n", [2.0, 3.0], scale).colorbar
    assert (bar.norm.vmin, bar.norm.vmax) == (1.0, 9.0)
    assert draw.color_scale(numbers, vmin=0)["norm"].vmin == 0
    log = LogNorm()
    assert draw.color_scale(numbers, norm=log)["norm"].vmax == 9.0
    assert log.vmax is None  # the caller's norm is not filled

    labels = pd.Series(["b", "a", "c"], name="k")
    colors = draw.color_scale(labels)["colors"]
    assert list(colors) == ["a", "b", "c"]
    # an image holding only "a" lists a, b, c in the shared colours
    legend = drawn("k", ["a", "a"], {"colors": colors}).legend
    assert [t.get_text() for t in legend.get_texts()] == ["a", "b", "c"]
    assert colors["c"] == TAB10(2)


@pytest.mark.parametrize(
    "values, options, error, match",
    [
        (
            [1.0, 2.0],
            {"colors": {1.0: "red"}},
            ParamsError,
            "colors= is for label columns",
        ),
        (
            ["a", "b"],
            {"cmap": "viridis"},
            ParamsError,
            "are for numeric columns",
        ),
        (
            [1.0, 2.0],
            {"norm": LogNorm(), "vmin": 1.0},
            ParamsError,
            "pass either norm= or vmin=",
        ),
        (
            [f"l{i}" for i in range(11)],
            {},
            TableError,
            "more than the 10 default colours",
        ),
    ],
)
def test_options_are_checked_as_by_the_draw_functions(
    values, options, error, match
):
    with pytest.raises(error, match=match):
        draw.color_scale(pd.Series(values, name="v"), **options)
