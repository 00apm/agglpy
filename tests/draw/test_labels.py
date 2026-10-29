"""Label columns (text, category, bool): one colour per label, a legend."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

from agglpy import draw
from agglpy.errors import ParamsError, TableError

from support.render import render, rgba

TAB10 = matplotlib.colormaps["tab10"]


def table(labels) -> pd.DataFrame:
    """Three separate particles with the given labels."""
    return pd.DataFrame(
        {
            "x": [30.0, 70.0, 50.0],
            "y": [40.0, 40.0, 90.0],
            "r": [10.0, 10.0, 5.0],
            "type": labels,
        }
    )


def drawn(p: pd.DataFrame, **options) -> tuple[np.ndarray, draw.Drawing]:
    fig, ax = draw.figure((120, 100))
    drawing = draw.particles(ax, p, color_by="type", alpha=1.0, **options)
    return render(fig), drawing


def legend_labels(drawing: draw.Drawing) -> list[str]:
    return [t.get_text() for t in drawing.legend.get_texts()]


def test_sorted_labels_get_the_tab10_colours():
    image, drawing = drawn(table(["similar", "collector", "similar"]))
    np.testing.assert_array_equal(image[40, 30], rgba(TAB10(1)))  # similar
    np.testing.assert_array_equal(image[40, 70], rgba(TAB10(0)))  # collector
    assert legend_labels(drawing) == ["collector", "similar"]
    assert drawing.legend.get_title().get_text() == "type"


def test_category_colours_do_not_depend_on_the_labels_present():
    kinds = ["collector", "similar", "separate"]
    one = table(pd.Categorical(["separate"] * 3, categories=kinds))
    two = table(
        pd.Categorical(["similar", "separate", "similar"], categories=kinds)
    )
    image_one, drawing_one = drawn(one)
    image_two, drawing_two = drawn(two)
    # "separate" is the third category in both images
    np.testing.assert_array_equal(image_one[40, 30], rgba(TAB10(2)))
    np.testing.assert_array_equal(image_two[40, 70], rgba(TAB10(2)))
    # identical legends, absent categories included
    assert legend_labels(drawing_one) == legend_labels(drawing_two) == kinds


def test_colors_map_labels_to_colours():
    colors = {"a": "red", "b": "blue", "c": "green"}
    image, drawing = drawn(table(["a", "b", "a"]), colors=colors)
    np.testing.assert_array_equal(image[40, 30], [255, 0, 0, 255])
    np.testing.assert_array_equal(image[40, 70], [0, 0, 255, 255])
    assert legend_labels(drawing) == ["a", "b", "c"]  # every key


def test_bool_column_is_labels():
    image, drawing = drawn(table([True, False, True]))
    np.testing.assert_array_equal(image[40, 70], rgba(TAB10(0)))  # False
    assert legend_labels(drawing) == ["False", "True"]


def test_a_missing_label_is_drawn_as_others():
    image, _ = drawn(table(["a", None, "a"]), linewidth=6)
    np.testing.assert_array_equal(image[40, 70], [255, 255, 255, 255])
    assert (image[40, 80] != 255).any()  # grey outline


def test_a_label_without_a_colour_raises():
    with pytest.raises(TableError, match="'b'"):
        drawn(table(["a", "b", "a"]), colors={"a": "red"})


def test_more_than_ten_labels_need_colors():
    p = pd.DataFrame(
        {
            "x": np.arange(11) * 9.0 + 5,
            "y": np.full(11, 50.0),
            "r": np.full(11, 3.0),
            "type": [f"t{i}" for i in range(11)],
        }
    )
    with pytest.raises(TableError, match="colors="):
        drawn(p)


def test_numeric_options_do_not_fit_labels():
    with pytest.raises(ParamsError, match="colors="):
        drawn(table(["a", "b", "a"]), vmax=3)


def test_legend_without_strip_goes_into_the_axes():
    _fig, ax = draw.figure((120, 100), legend=False)
    drawing = draw.particles(ax, table(["a", "b", "a"]), color_by="type")
    assert drawing.legend is not None and drawing.legend.axes is ax


def test_legend_text_is_white_on_a_dark_background():
    _fig, ax = draw.figure((120, 100), background="black")
    drawing = draw.particles(ax, table(["a", "b", "a"]), color_by="type")
    for text in [drawing.legend.get_title(), *drawing.legend.get_texts()]:
        assert matplotlib.colors.to_rgba(text.get_color()) == (1, 1, 1, 1)


def test_an_empty_selection_has_no_legend():
    p = table(pd.Categorical(["a", "b", "a"], categories=["a", "b"]))
    _, drawing = drawn(p, select=p["x"] > 1000)
    assert drawing.legend is None


def test_bool_colours_do_not_depend_on_the_values_present():
    image_one, drawing_one = drawn(table([True, True, True]))
    image_two, drawing_two = drawn(table([False, True, True]))
    # True keeps the second colour also where no False is drawn
    np.testing.assert_array_equal(image_one[40, 30], rgba(TAB10(1)))
    np.testing.assert_array_equal(image_two[40, 70], rgba(TAB10(1)))
    assert legend_labels(drawing_one) == legend_labels(drawing_two)
    assert legend_labels(drawing_one) == ["False", "True"]


def test_legend_from_colours():
    colours = {"new (3)": "lime", "gone (1)": "red", "same (40)": "grey"}
    _, ax = draw.figure((200, 300))
    key = draw.legend(ax, colours, title="circles")
    assert key.axes.get_label() == "agglpy legend"
    assert [t.get_text() for t in key.get_texts()] == list(colours)
    assert [
        matplotlib.colors.to_rgba(h.get_color()) for h in key.legend_handles
    ] == [matplotlib.colors.to_rgba(c) for c in colours.values()]
    assert key.get_title().get_text() == "circles"
    # the size of the label legends of particles: 10 % of the strip
    assert key.get_texts()[0].get_fontsize() == pytest.approx(3.6)
    # without a strip: in the axes, no title
    _, ax = draw.figure((200, 300), legend=False)
    key = draw.legend(ax, {"edges": "white"})
    assert key.axes is ax
    assert key.get_title().get_text() == ""
