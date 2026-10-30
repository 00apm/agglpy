"""agglpy._notebook: agglpy's figures shown in Jupyter."""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from agglpy import draw

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
