"""agglpy.draw.particles: one circle per particle, coloured by a column.

Each test draws on a white 100 x 120 px figure and checks pixels of the
saved PNG at circle centres (row = y, column = x).
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import LogNorm, Normalize

from agglpy import draw
from agglpy.errors import ParamsError, TableError

from support.render import render, rgba

WHITE = [255, 255, 255, 255]
GREY = rgba(matplotlib.colors.to_rgba("grey"))
VIRIDIS = matplotlib.colormaps["viridis"]


def table(**extra) -> pd.DataFrame:
    """Three separate particles; D = 20, 20, 10 px."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x": [30.0, 70.0, 50.0],
            "y": [40.0, 40.0, 90.0],
            "r": [10.0, 10.0, 5.0],
            "D": [20.0, 20.0, 10.0],
            **extra,
        }
    )


def drawn(p: pd.DataFrame, **options) -> tuple[np.ndarray, draw.Drawing]:
    fig, ax = draw.figure((120, 100), legend=False)
    options = {"alpha": 1.0, "legend": False, **options}
    drawing = draw.particles(ax, p, **options)
    return render(fig), drawing


def test_centre_colour_is_the_colormap_of_the_value():
    image, _ = drawn(table(), color_by="D", cmap="viridis")
    # default range: the selected values, 10 ... 20
    np.testing.assert_array_equal(image[40, 30], rgba(VIRIDIS(1.0)))
    np.testing.assert_array_equal(image[90, 50], rgba(VIRIDIS(0.0)))
    np.testing.assert_array_equal(image[5, 5], WHITE)


def test_vmin_vmax_and_norm_set_the_range():
    image, _ = drawn(table(), color_by="D", cmap="viridis", vmin=0, vmax=40)
    np.testing.assert_array_equal(image[40, 30], rgba(VIRIDIS(0.5)))
    norm = LogNorm(5, 20)
    image, _ = drawn(table(), color_by="D", cmap="viridis", norm=norm)
    np.testing.assert_array_equal(image[90, 50], rgba(VIRIDIS(norm(10.0))))


def test_an_unscaled_norm_is_not_modified():
    # Review Focus: matplotlib scales an unset norm in place, which
    # would carry the first image's range over to the next one.
    norm = Normalize()
    drawn(table(), color_by="D", norm=norm)
    assert (norm.vmin, norm.vmax) == (None, None)


def test_fill_alpha_and_opaque_outline():
    image, _ = drawn(
        table(), color_by="D", cmap="viridis", alpha=0.5, linewidth=4
    )
    colour = np.asarray(VIRIDIS(1.0))
    mixed = colour[:3] * 0.5 + 0.5  # half over white
    np.testing.assert_allclose(image[40, 30, :3], mixed * 255, atol=2)
    # the outline at the circle's right edge is opaque
    np.testing.assert_array_equal(image[40, 40], rgba(VIRIDIS(1.0)))


def test_unselected_particles_are_grey_outlines():
    p = table()
    image, drawing = drawn(p, color_by="D", select=p["D"] > 15, linewidth=6)
    np.testing.assert_array_equal(image[90, 50], WHITE)  # not filled
    np.testing.assert_array_equal(image[90, 55], GREY)  # its outline
    assert drawing.others is not None


def test_others_none_hides_them():
    p = table()
    image, drawing = drawn(
        p, color_by="D", select=p["D"] > 15, others=None, linewidth=6
    )
    np.testing.assert_array_equal(image[90, 55], WHITE)
    assert drawing.others is None


def test_others_take_any_colour():
    p = table()
    image, _ = drawn(
        p, color_by="D", select=p["D"] > 15, others="red", linewidth=6
    )
    np.testing.assert_array_equal(image[90, 55], [255, 0, 0, 255])


def test_the_range_comes_from_the_selected_values():
    p = table(D=[20.0, 30.0, 10.0])
    image, _ = drawn(p, color_by="D", cmap="viridis", select=p["D"] > 15)
    np.testing.assert_array_equal(image[40, 30], rgba(VIRIDIS(0.0)))
    np.testing.assert_array_equal(image[40, 70], rgba(VIRIDIS(1.0)))


def test_a_particle_without_a_value_is_drawn_as_others():
    image, _ = drawn(table(D=[20.0, np.nan, 10.0]), color_by="D", linewidth=6)
    np.testing.assert_array_equal(image[40, 70], WHITE)
    np.testing.assert_array_equal(image[40, 80], GREY)


def test_largest_first_keeps_an_enclosed_particle_visible():
    # B (r 4) lies inside A (r 15) and comes first in the table.
    p = pd.DataFrame(
        {"x": [50.0, 50.0], "y": [60.0, 60.0], "r": [4.0, 15.0]}
    ).assign(v=[0.0, 1.0])
    image, _ = drawn(p, color_by="v", cmap="viridis")
    np.testing.assert_array_equal(image[60, 50], rgba(VIRIDIS(0.0)))


def test_the_image_is_drawn_underneath_as_it_is():
    image = np.full((120, 100), 100, dtype=np.uint8)
    image[0, 0] = 101  # no contrast stretch: 100 stays 100
    rendered, _ = drawn(table(), color_by="D", image=image)
    np.testing.assert_array_equal(rendered[5, 5], [100, 100, 100, 255])
    np.testing.assert_array_equal(rendered[0, 0], [101, 101, 101, 255])


@pytest.mark.parametrize(
    ("background", "expected"),
    [("black", 80), ("white", 233)],  # 0.4 * 200 + 0.6 * (0 or 255)
)
def test_image_alpha_dims_the_image_towards_the_background(
    background, expected
):
    image = np.full((120, 100), 200, dtype=np.uint8)
    fig, ax = draw.figure((120, 100), legend=False, background=background)
    draw.particles(
        ax, table(), color_by="D", image=image, image_alpha=0.4, legend=False
    )
    rendered = render(fig)
    np.testing.assert_allclose(rendered[5, 5, :3], expected, atol=1)
    assert rendered[5, 5, 3] == 255


def test_a_colorbar_goes_into_the_strip():
    fig, ax = draw.figure((120, 100))
    drawing = draw.particles(ax, table(), color_by="D")
    assert drawing.colorbar is not None
    assert drawing.colorbar.ax.yaxis.label.get_text() == "D"
    assert render(fig).shape == (120, 112, 4)  # image area unchanged


def test_legend_text_sets_the_colorbar_title():
    _fig, ax = draw.figure((120, 100))
    drawing = draw.particles(ax, table(), color_by="D", legend="D [px]")
    assert drawing.colorbar.ax.yaxis.label.get_text() == "D [px]"


@pytest.mark.parametrize(
    ("background", "text"),
    [
        ("white", "black"),
        ("black", "white"),
        ("navy", "white"),
        (None, "black"),
    ],
)
def test_legend_text_follows_the_background(background, text):
    _fig, ax = draw.figure((120, 100), background=background)
    drawing = draw.particles(ax, table(), color_by="D")
    label = drawing.colorbar.ax.yaxis.label
    assert matplotlib.colors.to_rgba(
        label.get_color()
    ) == matplotlib.colors.to_rgba(text)


def test_without_strip_the_colorbar_goes_next_to_the_axes():
    _fig, ax = draw.figure((120, 100), legend=False)
    drawing = draw.particles(ax, table(), color_by="D")
    assert drawing.colorbar is not None


def test_legend_false_adds_nothing():
    _fig, ax = draw.figure((120, 100))
    drawing = draw.particles(ax, table(), color_by="D", legend=False)
    assert (drawing.colorbar, drawing.legend) == (None, None)


def test_an_empty_selection_draws_only_others():
    p = table()
    _fig, ax = draw.figure((120, 100))
    drawing = draw.particles(ax, p, color_by="D", select=p["D"] > 100)
    assert drawing.fills is None and drawing.others is not None
    assert (drawing.colorbar, drawing.legend) == (None, None)


def test_one_value_draws_without_warnings():
    # Review Focus: vmin == vmax (one particle, or all equal).
    fig, ax = draw.figure((120, 100))
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        drawing = draw.particles(ax, table(D=[5.0, 5.0, 5.0]), color_by="D")
        render(fig)
    assert drawing.colorbar is not None


def test_own_axes_fit_the_circles_y_down():
    fig, ax = plt.subplots()
    try:
        draw.particles(ax, table(), color_by="D", legend=False)
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        assert left <= 20 and right >= 80
        assert bottom >= 95 and top <= 30  # y down: bottom > top
    finally:
        plt.close(fig)


def test_the_table_is_not_modified():
    p = table()
    before = p.copy()
    drawn(p, color_by="D", select=p["D"] > 15)
    pd.testing.assert_frame_equal(p, before)


def test_unknown_column_raises():
    with pytest.raises(TableError, match="'volume'"):
        drawn(table(), color_by="volume")


def test_missing_geometry_raises():
    with pytest.raises(TableError, match="'r'"):
        drawn(table().drop(columns="r"), color_by="D")


@pytest.mark.parametrize(
    "select",
    [
        pd.Series([True, False]),  # too short
        pd.Series([True, False, True], index=[5, 6, 7]),  # other rows
        pd.Series([1, 0, 1]),  # not boolean
        np.array([True, False, True]),  # not a Series
    ],
)
def test_select_must_be_a_boolean_series_on_the_index(select):
    with pytest.raises(TableError, match="select"):
        drawn(table(), color_by="D", select=select)


def test_select_is_matched_by_index_not_position():
    p = table().set_axis([10, 11, 12])
    select = pd.Series([False, True, True], index=[12, 10, 11])
    image, _ = drawn(p, color_by="D", select=select)
    np.testing.assert_array_equal(image[90, 50], WHITE)  # 12: not selected
    assert (image[40, 30] != WHITE).any()  # 10: selected


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"colors": {"a": "red"}}, "colors="),
        ({"norm": Normalize(), "vmax": 3}, "norm="),
        ({"alpha": 1.5}, "alpha"),
        ({"image_alpha": -0.1}, "image_alpha"),
        ({"linewidth": -1}, "linewidth"),
    ],
)
def test_options_that_do_not_fit_raise(options, message):
    with pytest.raises(ParamsError, match=message):
        drawn(table(), color_by="D", **options)


def test_drawing_shows_in_jupyter():
    _fig, ax = draw.figure((120, 100))
    drawing = draw.particles(ax, table(), color_by="D")
    assert drawing._repr_png_()[:8] == b"\x89PNG\r\n\x1a\n"


def test_drawing_on_pyplot_axes_leaves_showing_to_pyplot():
    fig, ax = plt.subplots()
    try:
        drawing = draw.particles(ax, table(), color_by="D")
        assert drawing._repr_png_() is None
    finally:
        plt.close(fig)


def test_select_on_a_table_with_a_repeated_index():
    # e.g. the tables of two images joined with pd.concat
    p = pd.concat([table().iloc[:2], table().iloc[2:]]).set_axis([0, 1, 0])
    image, _ = drawn(p, color_by="D", select=p["D"] > 15)
    np.testing.assert_array_equal(image[90, 50], WHITE)
    assert (image[40, 30] != WHITE).any()


def test_transparent_outside_the_circles():
    fig, ax = draw.figure((120, 100), background=None, legend=False)
    draw.particles(ax, table(), color_by="D", alpha=1.0, legend=False)
    image = render(fig)
    assert image[5, 5, 3] == 0  # background: transparent
    assert image[40, 30, 3] == 255  # an opaque circle
