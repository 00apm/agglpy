"""agglpy._notebook: agglpy's figures shown in Jupyter."""

import io
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from agglpy import _notebook, draw

# ipympl builds its toolbar with an argument traitlets deprecates
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing unrecognized:DeprecationWarning"
)

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


@pytest.fixture(autouse=True)
def _close_widgets():
    """Widgets stay registered until closed, with their figures."""
    yield
    widgets = sys.modules.get("ipywidgets")
    if widgets is not None:
        widgets.Widget.close_all()


def widget_formatter():
    """IPython's formatter with the printers %matplotlib widget adds."""
    formatters = pytest.importorskip("IPython.core.formatters")
    # read without resolving: get_backend could pick a GUI backend
    backend = dict.__getitem__(matplotlib.rcParams, "backend")
    pytest.importorskip("ipympl.backend_nbagg")
    # importing ipympl switches to its backend
    if isinstance(backend, str):
        matplotlib.use(backend)
    else:  # none chosen yet
        dict.__setitem__(matplotlib.rcParams, "backend", backend)

    def reshow(fig):  # as IPython's: only figures with a pyplot manager
        if fig.canvas.manager is None:
            raise NotImplementedError

    formatter = formatters.DisplayFormatter()
    formatter.ipython_display_formatter.for_type(Figure, reshow)
    formatter.formatters["image/png"].for_type(Figure, lambda fig: "png")
    return formatter


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


def test_widget_opens_at_a_moderate_width():
    # before the browser has measured the cell
    fig, _, _ = fitted((200, 2000))
    assert css_width(fig) == pytest.approx(_notebook.FALLBACK_WIDTH)
    # a small figure keeps its size
    fig, _, _ = fitted((20, 30))
    assert css_width(fig) == pytest.approx(30 + 4)  # with the strip


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


def test_refitting_keeps_the_zoom():
    fig, _, probe = fitted((200, 2000))
    ax = fig.axes[0]
    ax.set_xlim(100, 300)
    ax.set_ylim(150, 50)
    probe.width = 700
    assert ax.get_xlim() == (100, 300)
    assert ax.get_ylim() == (150, 50)


def test_without_anywidget_the_canvas_alone(monkeypatch, request):
    widget_formatter()
    monkeypatch.setitem(sys.modules, "anywidget", None)  # not installed
    _notebook._probe_class.cache_clear()  # made with anywidget before
    request.addfinalizer(_notebook._probe_class.cache_clear)
    fig, _ = draw.figure((200, 2000))
    assert _notebook.widget(fig) is fig.canvas
    assert css_width(fig) == pytest.approx(_notebook.FALLBACK_WIDTH)


def test_outside_jupyter_no_backend_is_chosen(monkeypatch):
    # e.g. terminal IPython: asking matplotlib would pick a GUI backend
    def chosen():
        raise AssertionError("the backend was resolved")

    monkeypatch.setattr("matplotlib.get_backend", chosen)
    monkeypatch.delitem(sys.modules, "ipympl.backend_nbagg", raising=False)
    fig, _ = draw.figure((20, 30))
    assert fig._repr_mimebundle_() is None


def test_showing_again_reuses_the_widget():
    fig, _, _ = fitted((200, 2000))
    assert _notebook.widget(fig) is _notebook.widget(fig)


def test_saving_a_shown_figure_gives_the_full_size():
    fig, drawing, probe = fitted((200, 2000))
    label = drawing.colorbar.ax.yaxis.label
    probe.width = 560
    fitted_size = label.get_fontsize()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    assert plt.imread(buffer).shape[:2] == (200, 2000 + 240)
    # the widget keeps its fitted size
    assert css_width(fig) == pytest.approx(560)
    assert label.get_fontsize() == pytest.approx(fitted_size)


def test_texts_added_after_the_first_fit_scale_too():
    fig, _, probe = fitted((200, 2000))
    text = fig.text(0.5, 0.5, "added", fontsize=20)  # sized for full size
    probe.width = 1120  # half of 2240
    assert text.get_fontsize() == pytest.approx(10)


def test_a_drawing_shown_on_the_widget_is_still_a_png_later():
    _, drawing, _ = fitted((20, 30))  # the canvas has a manager now
    assert drawing._repr_png_()[:8] == b"\x89PNG\r\n\x1a\n"


def test_drawing_on_pyplot_axes_leaves_showing_to_pyplot():
    fig, ax = plt.subplots()
    try:
        drawing = draw.particles(ax, ONE_PARTICLE, color_by="D")
        assert drawing._repr_png_() is None
        assert drawing._repr_mimebundle_() is None
    finally:
        plt.close(fig)
