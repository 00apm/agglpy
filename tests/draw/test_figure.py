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
