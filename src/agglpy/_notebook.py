"""Showing agglpy's figures in Jupyter (part of the draw edge).

Figures are made without pyplot, so Jupyter's matplotlib backends do
not show them by themselves: a ``ShownFigure`` as the last expression
of a cell is a PNG under the inline backend and an ipympl canvas (zoom
and pan) under ``%matplotlib widget``. ipympl stays the user's optional
install: it is imported only when its backend is on.

The canvas fits the cell: a full-size figure (one figure pixel per
image pixel) would be wider than the notebook. A small widget measures
the cell's width in the browser (anywidget) and the figure is resized
to it, never above its full size; zooming in then shows the detail.
"""

import functools
import io
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal
from weakref import WeakKeyDictionary

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.axis import Axis, Tick
from matplotlib.figure import Figure
from matplotlib.text import Text

# Names of the ipympl backend (``%matplotlib widget``), as ipympl lists them.
_WIDGET_BACKENDS = ("widget", "ipympl", "module://ipympl.backend_nbagg")
# Width in CSS px a canvas opens at before the browser has measured the
# cell, and keeps without anywidget.
FALLBACK_WIDTH = 800

# Measures its own width (the cell's: it spans the output) and sends it
# once the size has settled, e.g. after the window was resized.
_PROBE_ESM = """
function render({ model, el }) {
  el.style.width = "100%";
  el.style.height = "0px";
  let timer;
  const observer = new ResizeObserver(([entry]) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const width = Math.floor(entry.contentRect.width);
      if (width > 0 && width !== model.get("width")) {
        model.set("width", width);
        model.save_changes();
      }
    }, 150);
  });
  observer.observe(el);
  return () => {
    clearTimeout(timer);
    observer.disconnect();
  };
}
export default { render };
"""


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
        # before those printers' PNG. Without ipympl imported its
        # backend is not on, and get_backend would choose one (a GUI
        # backend in terminal IPython).
        if (
            "ipympl.backend_nbagg" not in sys.modules
            or mpl.get_backend() not in _WIDGET_BACKENDS
        ):
            return None  # IPython: the PNG
        # IPython adds the backend's PNG to the bundle, for viewers
        # without widgets
        bundle: dict[str, Any] = widget(self)._repr_mimebundle_(**kwargs)
        return bundle

    def savefig(self, *args: Any, **kwargs: Any) -> None:
        """As ``Figure.savefig``, at full size after a fit to a cell."""
        fit = getattr(self.canvas, "_agglpy_fit", None)
        if fit is None:
            super().savefig(*args, **kwargs)
            return
        with fit.full_size():
            super().savefig(*args, **kwargs)


def widget(figure: Figure) -> Any:
    """The figure on an ipympl canvas that fits the cell's width.

    A box with the width probe and the canvas; the canvas alone, at
    ``FALLBACK_WIDTH``, without anywidget. Made once per figure and
    kept, so every display shows the same zoom.
    """
    from ipympl.backend_nbagg import Canvas, FigureManager

    canvas: Any = figure.canvas
    if isinstance(canvas, Canvas):
        return canvas._agglpy_widget
    canvas = Canvas(figure)  # becomes figure.canvas
    FigureManager(canvas, 0)  # the toolbar
    canvas.header_visible = False  # "Figure 0" says nothing
    canvas.toolbar_position = "top"  # the cell's width is the figure's
    canvas.resizable = False  # a drag would not scale the fonts
    fit = canvas._agglpy_fit = _Fit(figure)
    fit(FALLBACK_WIDTH)
    canvas._agglpy_widget = canvas
    try:
        probe = _probe_class()()
    except ImportError:
        return canvas
    from ipywidgets import VBox

    probe.observe(lambda change: fit(change["new"]), names="width")
    canvas._agglpy_widget = VBox([probe, canvas])
    return canvas._agglpy_widget


_Which = Literal["major", "minor"]
_WHICH: tuple[_Which, ...] = ("major", "minor")


class _Fit:
    """Resizes a figure to a width in CSS px, its fonts with it.

    Never above the full size the figure was made at. Fonts are sized
    in points, which a smaller figure does not shrink: each text and
    tick label is scaled from its full size, a text added later taken
    at full size when first seen. Lines keep their width, so zoomed in
    they stay sharp.
    """

    def __init__(self, figure: Figure) -> None:
        self._figure = figure
        width, height = figure.get_size_inches()
        self._size = (float(width), float(height))
        # CSS px per inch: the dpi before any HiDPI scaling by the
        # browser's device pixel ratio
        self._dpi = figure.dpi
        self._scale = 1.0
        # full sizes, by text and by (axis, which); weak: a removed
        # artist is not kept alive
        self._texts: WeakKeyDictionary[Text, float] = WeakKeyDictionary()
        self._ticks: WeakKeyDictionary[Axis, dict[_Which, float]] = (
            WeakKeyDictionary()
        )

    def __call__(self, width: int) -> None:
        if width <= 0:
            return
        self._apply(min(1.0, width / (self._size[0] * self._dpi)))
        # forward: the canvas in the browser takes the new size
        self._figure.set_size_inches(self._scaled(self._scale), forward=True)
        self._figure.canvas.draw_idle()

    @contextmanager
    def full_size(self) -> Iterator[None]:
        """The figure at full size inside the block, then fitted again."""
        scale = self._scale
        self._apply(1.0)
        self._figure.set_size_inches(self._size, forward=False)
        try:
            yield
        finally:
            self._apply(scale)
            self._figure.set_size_inches(self._scaled(scale), forward=False)

    def _scaled(self, scale: float) -> tuple[float, float]:
        return self._size[0] * scale, self._size[1] * scale

    def _apply(self, scale: float) -> None:
        """Scale every text and tick label from its full size."""
        # tick labels go by the axis' tick params: matplotlib makes new
        # ones (e.g. on zoom) at the size set there
        labels = {
            label
            for tick in self._figure.findobj(Tick)
            for label in (tick.label1, tick.label2)
        }
        for text in self._figure.findobj(Text):
            if text in labels:
                continue
            full = self._texts.setdefault(text, float(text.get_fontsize()))
            text.set_fontsize(full * scale)
        # inset axes too (the colorbar's), which figure.axes leaves out
        for ax in self._figure.findobj(Axes):
            for axis in (ax.xaxis, ax.yaxis):
                sizes = self._ticks.setdefault(axis, {})
                for which in _WHICH:
                    ticks = axis.get_ticklabels(which=which)
                    if which not in sizes and ticks:
                        sizes[which] = float(ticks[0].get_fontsize())
                    if which in sizes:
                        axis.set_tick_params(
                            which=which, labelsize=sizes[which] * scale
                        )
        self._scale = scale


@functools.cache
def _probe_class() -> type:
    """The width probe's widget class; ImportError without anywidget."""
    import anywidget
    import traitlets

    class WidthProbe(anywidget.AnyWidget):  # type: ignore[misc]
        _esm = _PROBE_ESM
        width = traitlets.Int(0).tag(sync=True)

    return WidthProbe


def png(figure: Figure) -> bytes:
    """The figure as PNG bytes, at its own dpi."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()
