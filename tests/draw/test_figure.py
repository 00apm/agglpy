"""agglpy.draw.figure: one image pixel per file pixel."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from agglpy import draw
from agglpy.errors import ParamsError

from support.render import render


@pytest.mark.parametrize(
    # 29, 57 and 3441 px came out 1 px short at 100 dpi (matplotlib 3.8)
    "shape",
    [(2304, 3072), (2073, 3072), (1, 1), (29, 57), (5, 3441)],
)
def test_without_legend_the_file_has_the_image_size(shape):
    fig, _ = draw.figure(shape, legend=False)
    assert render(fig).shape == (*shape, 4)


def test_the_legend_strip_is_12_percent_of_the_width():
    fig, ax = draw.figure((2304, 3072))
    assert render(fig).shape == (2304, 3072 + 369, 4)  # round(0.12 * 3072)
    # the image area itself keeps its size
    box = ax.get_window_extent()
    assert (box.width, box.height) == pytest.approx((3072, 2304))


def test_a_colour_image_shape_is_accepted():
    fig, _ = draw.figure((20, 30, 3), legend=False)
    assert render(fig).shape == (20, 30, 4)


def test_axes_use_pixel_centres_and_y_down():
    _, ax = draw.figure((20, 30))
    assert ax.get_xlim() == (-0.5, 29.5)
    assert ax.get_ylim() == (19.5, -0.5)  # y down
    assert not ax.axison  # no ticks, no frame


def test_background_colour_fills_the_figure():
    fig, _ = draw.figure((20, 30), background="black")
    image = render(fig)
    assert (image == [0, 0, 0, 255]).all()


def test_no_background_is_transparent():
    fig, _ = draw.figure((20, 30), background=None, legend=False)
    assert (render(fig)[..., 3] == 0).all()


def test_no_pyplot_figure_is_made():
    plt.close("all")
    for _ in range(25):  # pyplot would warn above 20 open figures
        draw.figure((20, 30))
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "shape", [(0, 10), (10,), (10.5, 20), (True, 20), "ab", None]
)
def test_shape_must_start_with_two_positive_integers(shape):
    with pytest.raises(ParamsError, match="shape"):
        draw.figure(shape)


def test_shape_from_numpy_integers():
    image = np.zeros((7, 9), dtype=np.uint8)
    fig, _ = draw.figure(image.shape, legend=False)
    assert render(fig).shape == (7, 9, 4)


def test_image_at_its_origin():
    # a region 10 rows high, drawn where it was cut: rows 10-19
    fig, ax = draw.figure((20, 30), legend=False, background="black")
    shown = draw.image(ax, np.full((10, 30), 255, np.uint8), origin=(0, 10))
    pixels = render(fig)
    assert (pixels[10:, :, :3] == 255).all()
    assert (pixels[:10, :, :3] == 0).all()
    # the axes keep the whole image's limits
    assert ax.get_xlim() == (-0.5, 29.5)
    assert ax.get_ylim() == (19.5, -0.5)
    assert shown.get_alpha() == 1.0
    # uint8 grey as it is: 100 stays 100, no contrast stretch
    fig, ax = draw.figure((20, 30), legend=False)
    draw.image(ax, np.full((20, 30), 100, np.uint8), alpha=1.0)
    assert (render(fig)[..., :3] == 100).all()
    for alpha in (-0.1, 1.5, True):
        with pytest.raises(ParamsError, match="alpha must be within 0-1"):
            draw.image(ax, np.zeros((2, 2), np.uint8), alpha=alpha)


def test_figure_pair_two_linked_panels():
    fig, left, right = draw.figure_pair((20, 30), legend=False)
    assert render(fig).shape == (20, 60, 4)
    assert left.get_shared_x_axes().joined(left, right)
    assert left.get_shared_y_axes().joined(left, right)
    for ax in (left, right):
        assert ax.get_xlim() == (-0.5, 29.5)
        assert ax.get_ylim() == (19.5, -0.5)
        assert not ax.axison
        box = ax.get_window_extent()
        assert (box.width, box.height) == pytest.approx((30, 20))
    # zooming one panel zooms the other
    right.set_xlim(0, 10)
    assert left.get_xlim() == (0, 10)
    # the legend strip right of the second panel, sized by one panel
    fig, left, right = draw.figure_pair((20, 30), background="black")
    assert render(fig).shape == (20, 64, 4)  # 60 + round(0.12 * 30)
    assert right.get_window_extent().x0 == pytest.approx(30)


def test_origin_shows_an_area_of_the_image_in_its_px():
    fig, ax = draw.figure((144, 200), legend=False, origin=(10, 20))
    assert ax.get_xlim() == (9.5, 209.5)
    assert ax.get_ylim() == (163.5, 19.5)
    image = (np.arange(200 * 300).reshape(200, 300) % 256).astype(np.uint8)
    draw.image(ax, image)
    shown = render(fig)
    assert shown.shape[:2] == (144, 200)
    np.testing.assert_array_equal(shown[:, :, 0], image[20:164, 10:210])


def test_origin_on_a_pair():
    _fig, left, right = draw.figure_pair((144, 200), origin=(0, 16))
    assert left.get_ylim() == right.get_ylim() == (159.5, 15.5)


@pytest.mark.parametrize("origin", [(0,), (0, np.nan), "ab", (0, True)])
def test_origin_must_be_two_finite_numbers(origin):
    with pytest.raises(ParamsError, match="origin"):
        draw.figure((10, 10), origin=origin)
