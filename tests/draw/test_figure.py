"""agglpy.draw.figure: one image pixel per file pixel."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from agglpy import draw
from agglpy.errors import ParamsError

from support.render import render

ONE_PARTICLE = pd.DataFrame(
    {"id": [1], "x": [10.0], "y": [8.0], "r": [4.0], "D": [8.0]}
)


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


def test_figures_show_in_jupyter():
    # Jupyter's inline backend shows a figure made without pyplot only
    # once something has switched the backend on; a figure that shows
    # itself needs nothing
    for fig in (draw.figure((20, 30))[0], draw.figure_pair((20, 30))[0]):
        assert fig._repr_png_()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_show_through_ipython():
    formatters = pytest.importorskip("IPython.core.formatters")
    fig, _ = draw.figure((20, 30))
    assert formatters.PNGFormatter()(fig)[:8] == b"\x89PNG\r\n\x1a\n"


def test_without_widget_backend_figures_show_as_png(monkeypatch):
    monkeypatch.setattr(
        "matplotlib.get_backend",
        lambda: "module://matplotlib_inline.backend_inline",
    )
    fig, ax = draw.figure((20, 30))
    drawing = draw.particles(ax, ONE_PARTICLE, color_by="D")
    # no bundle: IPython then shows the PNG
    assert fig._repr_mimebundle_() is None
    assert drawing._repr_mimebundle_() is None


def widget_formatter():
    """IPython's formatter with the printers %matplotlib widget adds."""
    formatters = pytest.importorskip("IPython.core.formatters")
    backend = matplotlib.get_backend()
    pytest.importorskip("ipympl.backend_nbagg")
    matplotlib.use(backend)  # importing ipympl switches to its backend

    def reshow(fig):  # as IPython's: only figures with a pyplot manager
        if fig.canvas.manager is None:
            raise NotImplementedError

    formatter = formatters.DisplayFormatter()
    formatter.ipython_display_formatter.for_type(Figure, reshow)
    formatter.formatters["image/png"].for_type(Figure, lambda fig: "png")
    return formatter


# ipympl builds its toolbar with an argument traitlets deprecates
@pytest.mark.filterwarnings("ignore:Passing unrecognized:DeprecationWarning")
def test_widget_backend_shows_a_zoomable_canvas(monkeypatch):
    formatter = widget_formatter()
    monkeypatch.setattr("matplotlib.get_backend", lambda: "widget")
    plt.close("all")
    fig, ax = draw.figure((20, 30))
    drawing = draw.particles(ax, ONE_PARTICLE, color_by="D")
    first, _ = formatter.format(fig)
    again, _ = formatter.format(drawing)  # the same figure, its canvas
    widget = "application/vnd.jupyter.widget-view+json"
    assert first[widget]["model_id"] == again[widget]["model_id"]
    assert first[widget]["model_id"] == fig.canvas.model_id
    assert plt.get_fignums() == []  # not held by pyplot


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
