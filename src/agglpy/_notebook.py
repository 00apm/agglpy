"""Showing agglpy's figures in Jupyter (part of the draw edge).

Figures are made without pyplot, so Jupyter's matplotlib backends do
not show them by themselves: a ``ShownFigure`` as the last expression
of a cell is a PNG under the inline backend and an ipympl canvas (zoom
and pan) under ``%matplotlib widget``. ipympl stays the user's optional
install: it is imported only when its backend is on.
"""

import io
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure

# Names of the ipympl backend (``%matplotlib widget``), as ipympl lists them.
_WIDGET_BACKENDS = ("widget", "ipympl", "module://ipympl.backend_nbagg")


class ShownFigure(Figure):
    """A figure that shows itself as the last expression of a cell.

    Jupyter's inline backend shows a figure made without pyplot only
    after ``%matplotlib inline`` or a first pyplot plot; under
    ``%matplotlib widget`` (ipympl) it is shown on a canvas that zooms
    and pans, pyplot still not involved.
    """

    def _repr_png_(self) -> bytes:
        return png(self)

    def _repr_mimebundle_(self, **kwargs: Any) -> dict[str, Any] | None:
        # The widget backend registers display printers for Figure that
        # only redraw pyplot figures; IPython asks for this bundle
        # before those printers' PNG. Called by IPython only, after
        # ipykernel has set the backend, so get_backend imports no
        # pyplot.
        if mpl.get_backend() not in _WIDGET_BACKENDS:
            return None  # IPython: the PNG
        from ipympl.backend_nbagg import Canvas, FigureManager

        canvas: Any = self.canvas
        if not isinstance(canvas, Canvas):
            canvas = Canvas(self)  # becomes self.canvas
            FigureManager(canvas, 0)  # the toolbar
            canvas.header_visible = False  # "Figure 0" says nothing
        bundle: dict[str, Any] = canvas._repr_mimebundle_(**kwargs)
        return bundle


def png(figure: Figure) -> bytes:
    """The figure as PNG bytes, at its own dpi."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()
