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

import copy
import io
import numbers
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import Collection, EllipseCollection
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Patch
from matplotlib.typing import ColorType
from numpy.typing import NDArray

from agglpy.errors import ParamsError, TableError

# File pixels per inch of the figures from ``figure``: at 72, one point
# is one pixel (linewidth=2 draws 2 px lines), and width / 72 * 72 gives
# back every width exactly (matplotlib < 3.9 cuts the size to whole px).
_DPI = 72
# Width of the legend strip, as a fraction of the image width.
_STRIP = 0.12
# Legend text height as a fraction of the strip width.
_TEXT = 0.1
# Label of the strip's axes, by which the draw functions find it.
_STRIP_LABEL = "agglpy legend"
# Default colours of label columns (10 colours).
_PALETTE = "tab10"


@dataclass(frozen=True)
class Drawing:
    """What a draw function added to the axes, for restyling or layout.

    Attributes:
        fills: The coloured shapes: the filled particle circles with
            their outlines (``particles``), or one filled shape per
            agglomerate (``agglomerates``).
        outlines: The agglomerates' outer outlines; None for
            ``particles``.
        others: The thin outlines of the unselected particles; None
            when there are none or ``others=None``.
        colorbar: The colorbar of a numeric ``color_by``, if drawn.
        legend: The legend of a label ``color_by``, if drawn.
    """

    fills: Collection | None
    outlines: Collection | None
    others: Collection | None
    colorbar: Colorbar | None
    legend: Legend | None

    def _figure(self) -> Figure | None:
        for artist in (self.fills, self.outlines, self.others):
            if artist is not None:
                figure = artist.get_figure()
                if isinstance(figure, Figure):
                    return figure
        return None

    def _repr_png_(self) -> bytes | None:
        # pyplot figures have a manager and pyplot shows them itself;
        # showing them here too would give the image twice.
        figure = self._figure()
        if figure is None or figure.canvas.manager is not None:
            return None
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        return buffer.getvalue()


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


def particles(
    ax: Axes,
    particles: pd.DataFrame,
    *,
    color_by: Hashable,
    select: pd.Series | None = None,
    image: NDArray[Any] | None = None,
    others: ColorType | None = "grey",
    cmap: str | Colormap | None = None,
    norm: Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colors: Mapping[Any, ColorType] | None = None,
    alpha: float = 0.5,
    linewidth: float = 1.0,
    legend: bool | str = True,
) -> Drawing:
    """Draw each particle as a circle coloured by a particle column.

    Each selected particle is a filled circle (``alpha``) with an opaque
    outline in the same colour, largest first: a small particle on a
    larger one stays visible.

    Args:
        ax: The axes to draw into, e.g. from ``figure``.
        particles: The image's particle table, in px (``x, y, r``).
        color_by: The column that gives the colours: numbers through a
            colormap, or labels (text, ``category``, bool) one colour
            each.
        select: A boolean Series on the particle table's index; None
            selects all. Unselected particles are drawn as ``others``.
        image: The image to draw underneath (a 2-D grey ``uint8``
            image is shown 0 = black to 255 = white, without contrast
            stretching; RGB / RGBA arrays as they are).
        others: Colour of the thin outlines of unselected particles,
            and of particles without a value; None hides them.
        cmap: Colormap of a numeric column (a name or a ``Colormap``);
            matplotlib's default when None.
        norm: A ``Normalize`` of a numeric column, e.g. ``LogNorm``;
            not modified. Not together with ``vmin`` / ``vmax``.
        vmin: Value of the colormap's lower end; the smallest selected
            value when None.
        vmax: Value of the colormap's upper end; the largest selected
            value when None.
        colors: ``{label: colour}`` for a label column; it must give
            every drawn label. Without it, the ``tab10`` colours go to
            the categories of a ``category`` column, else to the sorted
            labels.
        alpha: Opacity of the fill, 0 (outlines only) to 1.
        linewidth: Width of the outlines in points, others half of it;
            on a ``figure`` one point is one pixel of the file.
        legend: Add a colorbar (numbers) or a legend (labels): into
            the strip of ``figure`` if there is one. True titles it with
            the column name, a string sets the title, False adds none.
            Without a strip, a colorbar takes its room from the image:
            on ``figure(..., legend=False)`` pass False here too.

    Returns:
        The ``Drawing`` with the added artists.

    Raises:
        TableError: If ``x, y, r`` or ``color_by`` is missing or ``x, y,
            r`` is not finite with ``r > 0``; ``select`` is not a
            boolean Series on the table's index; a label has no colour
            (more than 10 labels need ``colors=``).
        ParamsError: If an option does not fit the column (``colors``
            for numbers, ``cmap`` / ``norm`` / ``vmin`` / ``vmax`` for
            labels), ``norm`` comes with ``vmin`` / ``vmax``, ``alpha``
            is not within 0-1 or ``linewidth`` is negative.
    """
    x, y, r = _geometry(particles)
    values = _column(particles, color_by)
    chosen = _selection(particles, select)
    style = _Style(alpha, linewidth)
    palette = _palette(values[chosen], cmap, norm, vmin, vmax, colors)
    # NaN has no colour: drawn as others
    colored = chosen & palette.has_colour(values)
    _draw_image(ax, image)
    others_artist = _others(ax, x, y, r, ~colored, others, style)
    # largest first: a smaller circle lands on top of a larger one
    order = np.argsort(-r[colored], kind="stable")
    rgba = palette.rgba(values[colored])[order]
    fills = None
    if colored.any():
        fills = EllipseCollection(
            2 * r[colored][order],
            2 * r[colored][order],
            np.zeros(int(colored.sum())),
            units="xy",
            offsets=np.column_stack([x[colored], y[colored]])[order],
            offset_transform=ax.transData,
            facecolors=_with_alpha(rgba, style.alpha),
            edgecolors=rgba,
            linewidths=style.linewidth,
        )
        ax.add_collection(fills, autolim=False)
    _fit(ax, x, y, r, image)
    bar, key = (
        _legend(ax, palette, color_by, legend, style)
        if colored.any()
        else (None, None)
    )
    return Drawing(fills, None, others_artist, bar, key)


# ---------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------


class _Palette:
    """Colours of one ``color_by`` column: a colormap or label colours."""

    def __init__(
        self,
        scale: mpl.cm.ScalarMappable | None,
        labels: dict[Any, ColorType] | None,
    ) -> None:
        self.scale = scale
        self.labels = labels

    def has_colour(self, values: pd.Series) -> NDArray[np.bool_]:
        if self.labels is not None:
            return values.isin(list(self.labels)).to_numpy(dtype=bool)
        return values.notna().to_numpy(dtype=bool)

    def rgba(self, values: pd.Series) -> NDArray[np.float64]:
        if self.labels is not None:
            return np.array(
                [to_rgba(self.labels[v]) for v in values], dtype=np.float64
            ).reshape(-1, 4)
        assert self.scale is not None
        return np.asarray(
            self.scale.to_rgba(values.to_numpy(dtype=np.float64)),
            dtype=np.float64,
        )


def _palette(
    values: pd.Series,
    cmap: str | Colormap | None,
    norm: Normalize | None,
    vmin: float | None,
    vmax: float | None,
    colors: Mapping[Any, ColorType] | None,
) -> _Palette:
    """Colours for the selected ``values`` of one call."""
    numeric = pd.api.types.is_numeric_dtype(
        values
    ) and not pd.api.types.is_bool_dtype(values)
    if numeric:
        if colors is not None:
            raise ParamsError(
                f"colors= is for label columns; {values.name!r} holds "
                "numbers (use cmap=)"
            )
        if norm is not None and (vmin is not None or vmax is not None):
            raise ParamsError("pass either norm= or vmin= / vmax=")
        # A copy: matplotlib fills an unset range in place, which would
        # carry this image's range over to the next call.
        scale_norm = (
            copy.copy(norm) if norm is not None else Normalize(vmin, vmax)
        )
        present = values.dropna().to_numpy(dtype=np.float64)
        if len(present):
            scale_norm.autoscale_None(present)
        scale = mpl.cm.ScalarMappable(
            scale_norm, mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        )
        scale.set_array(present)
        return _Palette(scale, None)

    if (
        cmap is not None
        or norm is not None
        or vmin is not None
        or (vmax is not None)
    ):
        raise ParamsError(
            f"cmap=, norm=, vmin= and vmax= are for numeric columns; "
            f"{values.name!r} holds labels (use colors=)"
        )
    found = values.dropna().unique().tolist()
    if colors is not None:
        missing = [v for v in found if v not in colors]
        if missing:
            raise TableError(f"colors= gives no colour for {_listed(missing)}")
        return _Palette(None, dict(colors))
    # category: colours by category order, the same in every image
    if isinstance(values.dtype, pd.CategoricalDtype):
        labels = list(values.cat.categories)
    else:
        labels = _sorted(found)
    palette = mpl.colormaps[_PALETTE]
    if len(labels) > palette.N:
        raise TableError(
            f"{len(labels)} labels in {values.name!r}, more than the "
            f"{palette.N} default colours: pass colors={{label: colour}}"
        )
    return _Palette(
        None, {label: palette(i) for i, label in enumerate(labels)}
    )


def _sorted(labels: list[Any]) -> list[Any]:
    try:
        return sorted(labels)
    except TypeError:  # mixed types, e.g. 1 and "a"
        return sorted(labels, key=str)


def _listed(labels: list[Any]) -> str:
    return ", ".join(repr(v) for v in labels)


def _with_alpha(
    rgba: NDArray[np.float64], alpha: float
) -> NDArray[np.float64]:
    faded = rgba.copy()
    faded[:, 3] = faded[:, 3] * alpha
    return faded


# ---------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------


def _others(
    ax: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    r: NDArray[np.float64],
    mask: NDArray[np.bool_],
    colour: ColorType | None,
    style: "_Style",
) -> Collection | None:
    if colour is None or not mask.any():
        return None
    outlines = EllipseCollection(
        2 * r[mask],
        2 * r[mask],
        np.zeros(int(mask.sum())),
        units="xy",
        offsets=np.column_stack([x[mask], y[mask]]),
        offset_transform=ax.transData,
        facecolors="none",
        edgecolors=[to_rgba(colour)],
        linewidths=style.linewidth / 2,
    )
    ax.add_collection(outlines, autolim=False)
    return outlines


def _draw_image(ax: Axes, image: NDArray[Any] | None) -> None:
    if image is None:
        return
    array = np.asarray(image)
    if array.ndim == 2:
        # uint8 as it is (0 black, 255 white), no contrast stretch
        limits = (0, 255) if array.dtype == np.uint8 else (None, None)
        ax.imshow(
            array,
            cmap="gray",
            vmin=limits[0],
            vmax=limits[1],
            interpolation="nearest",
        )
    else:
        ax.imshow(array, interpolation="nearest")


def _fit(
    ax: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    r: NDArray[np.float64],
    image: NDArray[Any] | None,
) -> None:
    """Fit axes without set limits to the circles; y down, equal aspect."""
    if (
        image is None
        and len(r)
        and (ax.get_autoscalex_on() or ax.get_autoscaley_on())
    ):
        ax.update_datalim(
            [
                (float((x - r).min()), float((y - r).min())),
                (float((x + r).max()), float((y + r).max())),
            ]
        )
        ax.autoscale_view()
    ax.set_aspect("equal")
    # pixel coordinates: y down, never mirrored against the image
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


# ---------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Style:
    alpha: float
    linewidth: float

    def __post_init__(self) -> None:
        if isinstance(self.alpha, bool) or not 0 <= self.alpha <= 1:
            raise ParamsError(f"alpha must be within 0-1, got {self.alpha!r}")
        if isinstance(self.linewidth, bool) or not self.linewidth >= 0:
            raise ParamsError(
                f"linewidth must be >= 0, got {self.linewidth!r}"
            )


def _legend(
    ax: Axes,
    palette: _Palette,
    color_by: Hashable,
    legend: bool | str,
    style: _Style,
) -> tuple[Colorbar | None, Legend | None]:
    if legend is False:
        return None, None
    title = str(color_by) if legend is True else str(legend)
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    strip = _strip(fig)
    text = _text_colour(fig)
    if palette.scale is not None:
        if strip is None:
            bar = fig.colorbar(palette.scale, ax=ax, label=title)
            return bar, None
        size = _font_size(fig, strip)
        cax = strip.inset_axes((0.1, 0.1, 0.15, 0.8))
        bar = fig.colorbar(palette.scale, cax=cax)
        bar.set_label(title, color=text, fontsize=size)
        bar.ax.tick_params(colors=text, labelsize=size)
        bar.outline.set_edgecolor(text)
        return bar, None
    assert palette.labels is not None
    handles = [
        Patch(
            facecolor=_with_alpha(np.array([to_rgba(c)]), style.alpha)[0],
            edgecolor=c,
            label=str(label),
        )
        for label, c in palette.labels.items()
    ]
    if strip is None:
        return None, ax.legend(handles=handles, title=title)
    size = _font_size(fig, strip)
    key = strip.legend(
        handles=handles,
        title=title,
        loc="center left",
        frameon=False,
        fontsize=size,
        title_fontsize=size,
        labelcolor=text,
    )
    key.get_title().set_color(text)
    return None, key


def _strip(fig: Figure) -> Axes | None:
    for axes in fig.axes:
        if axes.get_label() == _STRIP_LABEL:
            return axes
    return None


def _font_size(fig: Figure, strip: Axes) -> float:
    # In points: the strip's width in file pixels x _TEXT, at 72 pt/in.
    width = strip.get_position().width * fig.get_figwidth() * fig.dpi
    return _TEXT * width * 72 / fig.dpi


def _text_colour(fig: Figure) -> str:
    red, green, blue, opacity = to_rgba(fig.get_facecolor())
    # Transparent: the background underneath is unknown, keep black.
    if opacity == 0:
        return "black"
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.5 else "black"


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


def _geometry(
    table: pd.DataFrame,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    missing = [c for c in ("x", "y", "r") if c not in table.columns]
    if missing:
        raise TableError(f"the particle table has no columns {missing}")
    arrays = []
    for column in ("x", "y", "r"):
        try:
            values = pd.to_numeric(table[column]).to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise TableError(f"column {column!r} must be numeric") from exc
        if not np.isfinite(values).all():
            raise TableError(f"column {column!r} must be finite")
        arrays.append(values)
    x, y, r = arrays
    if (r <= 0).any():
        raise TableError("column 'r' must be > 0")
    return x, y, r


def _column(table: pd.DataFrame, column: Hashable) -> pd.Series:
    if column not in table.columns:
        raise TableError(f"color_by: no column {column!r} in the table")
    values: pd.Series = table[column]
    return values


def _selection(
    table: pd.DataFrame, select: pd.Series | None
) -> NDArray[np.bool_]:
    if select is None:
        return np.ones(len(table), dtype=bool)
    if isinstance(select, pd.Series) and pd.api.types.is_bool_dtype(select):
        if select.index.equals(table.index):
            return select.to_numpy(dtype=bool)
        # the same rows in another order
        if (
            len(select) == len(table)
            and not table.index.has_duplicates
            and not select.index.has_duplicates
            and select.index.isin(table.index).all()
        ):
            return select.reindex(table.index).to_numpy(dtype=bool)
    raise TableError(
        "select must be a boolean Series on the table's index, "
        "e.g. table['D'] > 5"
    )
