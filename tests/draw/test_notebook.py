"""agglpy._notebook: agglpy's figures shown in Jupyter."""

import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from agglpy import _notebook, draw

ONE_PARTICLE = pd.DataFrame(
    {"id": [1], "x": [10.0], "y": [8.0], "r": [4.0], "D": [8.0]}
)


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
IPYMPL_TOOLBAR = "ignore:Passing unrecognized:DeprecationWarning"


@pytest.mark.filterwarnings(IPYMPL_TOOLBAR)
def test_widget_backend_shows_a_zoomable_canvas(monkeypatch):
    formatter = widget_formatter()
    monkeypatch.setattr("matplotlib.get_backend", lambda: "widget")
    plt.close("all")
    fig, ax = draw.figure((20, 30))
    drawing = draw.particles(ax, ONE_PARTICLE, color_by="D")
    for shown in (fig, drawing):
        data, _ = formatter.format(shown)
        assert "application/vnd.jupyter.widget-view+json" in data
    assert plt.get_fignums() == []  # not held by pyplot


def fitted(shape, color_by="D"):
    """A figure with a colorbar on the widget canvas, its probe."""
    widget_formatter()  # skips without IPython / ipympl
    pytest.importorskip("anywidget")
    fig, ax = draw.figure(shape)
    drawing = draw.particles(ax, ONE_PARTICLE, color_by=color_by)
    box = _notebook.widget(fig)
    probe, canvas = box.children
    assert canvas is fig.canvas
    return fig, drawing, probe


def css_width(fig):
    return fig.get_figwidth() * 72  # 1 CSS px per figure px at 72 dpi


@pytest.mark.filterwarnings(IPYMPL_TOOLBAR)
def test_widget_opens_at_a_moderate_width():
    # before the browser has measured the cell
    fig, _, _ = fitted((200, 2000))
    assert css_width(fig) == pytest.approx(_notebook.FALLBACK_WIDTH)
    # a small figure keeps its size
    fig, _, _ = fitted((20, 30))
    assert css_width(fig) == pytest.approx(30 + 4)  # with the strip


@pytest.mark.filterwarnings(IPYMPL_TOOLBAR)
def test_widget_fits_the_cell_width_fonts_with_it():
    fig, drawing, probe = fitted((200, 2000))
    label = drawing.colorbar.ax.yaxis.label
    ticks = drawing.colorbar.ax.yaxis.get_major_ticks()[0].label1
    full = 2000 * 1.12
    probe.width = 560  # the browser reports the cell width
    assert css_width(fig) == pytest.approx(560)
    assert fig.get_figheight() / fig.get_figwidth() == pytest.approx(
        200 / full
    )
    scale = 560 / full
    full_size = 0.1 * 0.12 * 2000  # the strip's text size at full size
    assert label.get_fontsize() == pytest.approx(full_size * scale)
    assert ticks.get_fontsize() == pytest.approx(full_size * scale)
    # never above full size: zooming shows the detail
    probe.width = 5000
    assert css_width(fig) == pytest.approx(full)
    assert label.get_fontsize() == pytest.approx(full_size)


@pytest.mark.filterwarnings(IPYMPL_TOOLBAR)
def test_refitting_keeps_the_zoom():
    fig, _, probe = fitted((200, 2000))
    ax = fig.axes[0]
    ax.set_xlim(100, 300)
    ax.set_ylim(150, 50)
    probe.width = 700
    assert ax.get_xlim() == (100, 300)
    assert ax.get_ylim() == (150, 50)


@pytest.mark.filterwarnings(IPYMPL_TOOLBAR)
def test_without_anywidget_the_canvas_alone(monkeypatch, request):
    widget_formatter()
    monkeypatch.setitem(sys.modules, "anywidget", None)  # not installed
    _notebook._probe_class.cache_clear()  # made with anywidget before
    request.addfinalizer(_notebook._probe_class.cache_clear)
    fig, _ = draw.figure((200, 2000))
    assert _notebook.widget(fig) is fig.canvas
    assert css_width(fig) == pytest.approx(_notebook.FALLBACK_WIDTH)
