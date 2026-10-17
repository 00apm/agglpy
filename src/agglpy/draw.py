"""Drawing the detection: particles and agglomerates (the draw edge).

Static figures with matplotlib, in px: ``x, y, r`` of the particle
table and the image share pixel coordinates, pixel centres at integers,
y pointing down. ``figure`` makes a figure whose image area has exactly
the image's size in file pixels; ``particles`` colours each particle by
a column of the particle table, ``agglomerates`` each agglomerate by a
column of the agglomerate table. Both take the whole tables and a
``select=`` mask; unselected particles are drawn as thin outlines.

Figures are made without pyplot: nothing is kept in a global registry,
so a loop over many images holds no figures in memory. In Jupyter, a
``Drawing`` as the last expression of a cell shows its figure.
"""

import numbers
from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.typing import ColorType

from agglpy.errors import ParamsError

# File pixels per inch of the figures from ``figure``: at 72, one point
# is one pixel (linewidth=2 draws 2 px lines), and width / 72 * 72 gives
# back every width exactly (matplotlib < 3.9 cuts the size to whole px).
_DPI = 72
# Width of the legend strip, as a fraction of the image width.
_STRIP = 0.12
# Label of the strip's axes, by which the draw functions find it.
_STRIP_LABEL = "agglpy legend"


def figure(
    shape: Sequence[int],
    *,
    legend: bool = True,
    background: ColorType | None = "white",
) -> tuple[Figure, Axes]:
    """Make a figure with one image pixel per file pixel.

    The axes cover exactly ``width x height`` pixels of the saved file
    (``fig.savefig``), pixel centres at integer coordinates, y down, no
    ticks or frame: with ``legend=False`` the file has the image's size
    and lies pixel for pixel over the source image.

    Args:
        shape: The image's ``(height, width)``, e.g. ``image.shape``; a
            third entry (colour channels) is ignored.
        legend: Leave a strip to the right of the image for the
            colorbar or legend of the draw functions.
        background: Any matplotlib colour; None for a transparent
            figure (a PNG keeps the transparency). An image drawn with
            ``image=`` covers it.

    Returns:
        The figure and its image axes. The figure is not managed by
        pyplot: save it with ``fig.savefig``; in Jupyter it shows as the
        last expression of a cell or with ``display(fig)``.

    Raises:
        ParamsError: If ``shape`` does not start with two integers
            >= 1.
    """
    height, width = _shape(shape)
    strip = round(_STRIP * width) if legend else 0
    face = "none" if background is None else background
    fig = Figure(
        figsize=((width + strip) / _DPI, height / _DPI),
        dpi=_DPI,
        facecolor=face,
    )
    total = width + strip
    ax = fig.add_axes((0.0, 0.0, width / total, 1.0))
    ax.set_facecolor(face)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if legend:
        side = fig.add_axes(
            (width / total, 0.0, strip / total, 1.0), label=_STRIP_LABEL
        )
        side.set_axis_off()
    return fig, ax


# ---------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------


def _shape(shape: Sequence[int]) -> tuple[int, int]:
    try:
        height, width = tuple(shape)[:2]
    except (TypeError, ValueError):  # not a sequence, or too short
        height = width = 0
    if not all(
        isinstance(v, numbers.Integral) and not isinstance(v, bool) and v >= 1
        for v in (height, width)
    ):
        raise ParamsError(
            f"shape must start with (height, width) >= 1, got {shape!r}"
        )
    return int(height), int(width)
