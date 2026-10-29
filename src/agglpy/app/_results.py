"""``Results``: what ``Collection.analyze`` found, as plain tables."""

import base64
import dataclasses
import html
import io
import warnings
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike

from agglpy import draw as _draw
from agglpy.app._files import replace_file
from agglpy.app._report import reemit
from agglpy.app._sources import PathLike
from agglpy.app._work import export_folder
from agglpy.core import distributions
from agglpy.core import metrics as _metrics
from agglpy.core.agglomerates import select_agglomerates
from agglpy.core.properties import DIMENSIONS
from agglpy.errors import (
    ImageError,
    OutputExistsError,
    ParamsError,
    TableError,
)
from agglpy.export import _check_replaceable, write_results
from agglpy.io import read_image, read_image_metadata

__all__ = ["Drawings", "Origin", "Results"]


@dataclass(frozen=True)
class Origin:
    """What a ``Results`` was made from, for its export record.

    Attributes:
        analyzed: When ``analyze()`` ran (ISO 8601).
        collection: The collection folder.
        sheet: The sheet's file name.
        images: The record entry of each image, as plain YAML data.
        files: The image files, resolved, for drawing them.
        excluded: The sheet rows with ``include`` false: ``image`` and
            ``note``.
        steps: The subset steps taken, in order.
        names: Every image of the sheet.
    """

    analyzed: str
    collection: Path
    sheet: str
    images: Mapping[str, Mapping[str, object]]
    files: Mapping[str, Path]
    excluded: tuple[Mapping[str, str], ...]
    steps: tuple[Mapping[str, object], ...]
    names: tuple[str, ...]


_TABLES = ("particles", "agglomerates")

_SI = {1: "m", 2: "m²", 3: "m³"}
_GEOMETRY = {"x": 1, "y": 1, "r": 1}
_IMAGES_UNITS = {
    "pixel_size": "m",
    "fov_area": "m²",
    "width": "px",
    "height": "px",
}


class Results:
    """The particles and agglomerates of a collection, in SI units.

    Lengths are in m, areas in m², volumes in m³; ``pixel_size`` is in
    m per pixel. For other units add a column, e.g.
    ``res.particles["d_um"] = res.particles.D * 1e6``.

    The statistics methods take the tables as they are and group
    images by columns of ``images`` (``by="series"``, ``by="image"``):
    a sheet column groups images directly.

    The tables are the stored frames, not copies: add your own
    columns to them, and the methods, ``draw`` and ``subset`` see
    them. Don't change rows or agglpy's columns by hand; take a
    ``subset`` to leave images or agglomerates out.

    With pandas:

    - each particle with its agglomerate's properties:
      ``res.particles.merge(res.agglomerates, on=["image",
      "agglomerate_id"], suffixes=("", "_agglomerate"))``;
    - user columns onto particle rows:
      ``res.particles.merge(res.images[["image", "series"]],
      on="image")``;
    - the share of agglomerates touching the edge, per image:
      ``res.agglomerates.groupby("image").touches_edge.mean()``;
    - leaving them out:
      ``res.subset(agglomerates=~res.agglomerates.touches_edge)``.

    Attributes:
        images: One row per analysed image: agglpy's columns, the
            user columns of the sheet, then ``note``.
        particles: One row per particle, keyed by ``(image, id)``.
        agglomerates: One row per agglomerate, keyed by ``(image,
            agglomerate_id)``. Ids are unique only within an image
            and change with a new detection or correction.
    """

    def __init__(
        self,
        images: pd.DataFrame,
        particles: pd.DataFrame,
        agglomerates: pd.DataFrame,
        origin: Origin,
    ) -> None:
        """Made by ``Collection.analyze`` and ``subset``; not called
        directly."""
        self.images = images
        self.particles = particles
        self.agglomerates = agglomerates
        self._origin = origin

    def __repr__(self) -> str:
        steps = len(self._origin.steps)
        return (
            f"Results({len(self.images)} images, {len(self.particles)} "
            f"particles, {len(self.agglomerates)} agglomerates; {steps} "
            f"subset {'step' if steps == 1 else 'steps'})"
        )

    def summary(
        self,
        by: str | Iterable[str] | None = None,
        *,
        metrics: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Built-in metrics of each group of images, pooled.

        As ``agglpy.core.metrics.summary`` on these tables.
        """
        return self._core(
            _metrics.summary,
            self.images,
            self.particles,
            self.agglomerates,
            by,
            metrics=metrics,
        )

    def summary_across_images(
        self,
        by: str | Iterable[str] | None = None,
        *,
        metrics: str | Iterable[str] | None = None,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """The built-in metrics per image, then mean ± CI over images.

        As ``agglpy.core.metrics.summary_across_images`` on these
        tables.
        """
        return self._core(
            _metrics.summary_across_images,
            self.images,
            self.particles,
            self.agglomerates,
            by,
            metrics=metrics,
            confidence=confidence,
        )

    def values_across_images(
        self,
        per_image: pd.DataFrame,
        by: str | Iterable[str] | None = None,
        *,
        keys: str | Iterable[str] = (),
        values: str | Iterable[str] | None = None,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """Mean, std, n and confidence interval of per-image values.

        As ``agglpy.core.metrics.values_across_images`` with
        ``images``.
        """
        return self._core(
            _metrics.values_across_images,
            self.images,
            per_image,
            by,
            keys=keys,
            values=values,
            confidence=confidence,
        )

    def distribution(
        self,
        table: str,
        column: str,
        edges: ArrayLike,
        by: str | Iterable[str] | None = None,
        *,
        closed: str = "right",
        weights: str | Iterable[str] = (),
    ) -> pd.DataFrame:
        """Distribution of ``column`` of a table over size classes.

        As ``agglpy.core.distributions.distribution``; ``table`` is
        ``"particles"`` or ``"agglomerates"``.

        Raises:
            ParamsError: If ``table`` is another name.
        """
        return self._core(
            distributions.distribution,
            self.images,
            self._table(table),
            column,
            edges,
            by,
            closed=closed,
            weights=weights,
        )

    def distribution_across_images(
        self,
        table: str,
        column: str,
        edges: ArrayLike,
        by: str | Iterable[str] | None = None,
        *,
        closed: str = "right",
        weights: str | Iterable[str] = (),
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """The fraction of each class per image, then mean ± CI.

        As ``agglpy.core.distributions.distribution_across_images``;
        ``table`` is ``"particles"`` or ``"agglomerates"``.

        Raises:
            ParamsError: If ``table`` is another name.
        """
        return self._core(
            distributions.distribution_across_images,
            self.images,
            self._table(table),
            column,
            edges,
            by,
            closed=closed,
            weights=weights,
            confidence=confidence,
        )

    def subset(
        self,
        *,
        images: pd.Series | str | Sequence[str] | None = None,
        agglomerates: pd.Series | None = None,
        label: str | None = None,
    ) -> "Results":
        """A part of the results, with this step kept for the export.

        ``images`` keeps those images with their particles and
        agglomerates; ``agglomerates`` keeps those agglomerates with
        their member particles, and every image (an image with nothing
        left counts as zeros). Both: one step, images first. The tables
        of the new results are renumbered from 0; ``self`` stays as it
        is.

        Args:
            images: A True / False Series on ``images``' index, e.g.
                ``res.images.series == "a"``, or a name or list of
                names (kept in the order of the results).
            agglomerates: A True / False Series on ``agglomerates``'
                index, e.g. ``~res.agglomerates.touches_edge``.
            label: Names the step in the export record.

        Raises:
            ParamsError: If neither is given, a name is not in these
                results, or no image is left.
            TableError: If a mask is not a True / False Series on its
                table's index.
        """
        if images is None and agglomerates is None:
            raise ParamsError("subset needs images= or agglomerates=")
        table, particles, aggls = (
            self.images,
            self.particles,
            self.agglomerates,
        )
        keep = None if images is None else self._image_mask(images)
        if agglomerates is not None:
            # the same rows as images first: both keep an intersection
            particles, aggls = select_agglomerates(
                particles, aggls, agglomerates
            )
        if keep is not None:
            if not keep.any():
                raise ParamsError("no image left")
            table = table[keep]
            particles = particles[particles.image.isin(table.image)]
            aggls = aggls[aggls.image.isin(table.image)]
        step = {
            "label": label,
            "images": _kept(table, self.images),
            "agglomerates": _kept(aggls, self.agglomerates),
            "particles": _kept(particles, self.particles),
        }
        names = set(table.image)
        origin = dataclasses.replace(
            self._origin,
            images={
                k: v for k, v in self._origin.images.items() if k in names
            },
            files={k: v for k, v in self._origin.files.items() if k in names},
            steps=(*self._origin.steps, step),
        )
        return Results(
            table.reset_index(drop=True),
            particles.reset_index(drop=True),
            aggls.reset_index(drop=True),
            origin,
        )

    def export(
        self,
        folder: PathLike | None = None,
        *,
        tables: Mapping[str, pd.DataFrame] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write the tables, their units and a record into ``folder``.

        Writes ``results.xlsx`` (the sheets ``images``,
        ``agglomerates``, ``particles``, the extra ``tables``, then
        ``units`` and ``info``), ``csv/`` (one CSV per table and
        ``units.csv``) and ``record.yml``: per image the particle set,
        pixel size, region and the detection or correction the numbers
        come from, the rows with ``include`` false and the subset
        steps. Drawings of the same results fit in a folder next to it
        (``Drawings.export``).

        Args:
            folder: The folder to write into; created if missing. A
                folder named for each export is the recommended way;
                None gives ``exports/results`` in the collection, where
                a repeated export needs ``overwrite=True``. A relative
                path is taken from the current working directory.
            tables: Extra tables made from these results, e.g.
                ``{"summary": res.summary(by="series")}``; written
                after the three, without units.
            overwrite: Replace an earlier export in ``folder``.

        Returns:
            The folder, absolute.

        Raises:
            ParamsError: If an extra table is called ``images``,
                ``agglomerates`` or ``particles``, or ``folder`` lies in
                the collection's work folder ``agglpy/``.
            OutputExistsError: If ``folder`` has an export and
                ``overwrite`` is False.
            OutputLockedError: If a file to replace is open in another
                program.
        """
        folder = export_folder(self._origin.collection, folder, "results")
        extra = dict(tables or {})
        for name in ("images", "agglomerates", "particles"):
            if name in extra:
                raise ParamsError(
                    f"tables: {name!r} is a table of the results; give "
                    "the extra table another name"
                )
        written = {
            "images": self.images,
            "agglomerates": self.agglomerates,
            "particles": self.particles,
        }
        origin = self._origin
        record = {
            "analysis": {
                "analyzed": origin.analyzed,
                "collection": origin.collection.as_posix(),
                "sheet": origin.sheet,
                "unit": "m",
            },
            "images": dict(origin.images),
            "excluded": list(origin.excluded),
            "subset": list(origin.steps),
        }
        units = {
            "images": {
                c: u for c, u in _IMAGES_UNITS.items() if c in self.images
            },
            "agglomerates": _si_units(self.agglomerates),
            "particles": _si_units(self.particles),
        }
        return write_results(
            folder,
            {**written, **extra},
            record=record,
            units=units,
            overwrite=overwrite,
        )

    def draw(
        self,
        images: str | Sequence[str] | None = None,
        what: Literal["particles", "agglomerates"] = "agglomerates",
        *,
        color_by: Hashable,
        select: pd.Series | None = None,
        image: bool = True,
        image_alpha: float = 1.0,
        background: ColorType | None = "white",
        others: ColorType | None = "grey",
        cmap: str | Colormap | None = None,
        norm: Normalize | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        colors: Mapping[Any, ColorType] | None = None,
        alpha: float = 0.5,
        linewidth: float = 1.0,
        legend: bool | str = True,
    ) -> "Drawings":
        """Draw the particles or agglomerates of each image, in px.

        One colour scale for all drawn images: numbers share one
        ``vmin`` / ``vmax`` over the selected values of every image
        (unless given), labels one label → colour mapping, so a series
        compares at a glance (draw one with ``res.subset(images=...)``
        first). The figures are made when looked at (``d[name]``) or
        exported, not held: a figure changed by hand is saved with
        ``fig.savefig``.

        Args:
            images: A name or list of names; None draws every image.
            what: ``"particles"`` (each particle coloured by a column
                of ``particles``) or ``"agglomerates"`` (each
                agglomerate by a column of ``agglomerates``).
            color_by: The column of that table that gives the colours.
                A column of ``images`` has to be merged onto the table
                first.
            select: A True / False Series on that table's index; the
                others are drawn as thin ``others`` outlines.
            image: Draw the image file underneath, read when a figure
                is made.
            image_alpha: Opacity of the image, 0 to 1: below 1 the
                background shows through, e.g. ``background="black",
                image_alpha=0.4`` darkens the image so the circles
                stand out.
            background: The figure's colour, under the image: any
                matplotlib colour (``"white"``, ``"#FF000033"``, ``(0,
                1, 0, 0.47)`` with values in 0-1) or None for a
                transparent figure.
            others, cmap, norm, vmin, vmax, colors, alpha, linewidth:
                As for ``agglpy.draw.particles``.
            legend: A colorbar or legend in a strip right of the image;
                a string titles it; False draws the figure without the
                strip, the image's size exactly.

        Returns:
            The ``Drawings``: image name → figure.

        Raises:
            ParamsError: If no image or a name not in these results is
                given, ``what`` is
                another name, ``color_by`` is a column of ``images``
                only, ``image`` is not True / False, ``alpha`` /
                ``image_alpha`` is not within 0-1, ``linewidth`` is
                negative, or an option does not fit the column.
            TableError: If ``color_by`` is not a column of the table or
                ``select`` not a True / False Series on its index.
        """
        names = (
            list(self.images.image)
            if images is None
            else list(self.images.image[self._image_mask(images)])
        )
        if not names:
            raise ParamsError("no image to draw")
        if what not in _TABLES:
            raise ParamsError(
                f"what must be 'particles' or 'agglomerates', got {what!r}"
            )
        table = self._table(what)
        if color_by not in table.columns:
            if color_by in self.images.columns:
                raise ParamsError(
                    f"{color_by!r} is a column of res.images: merge it onto "
                    f"res.{what} first (res.{what}.merge(res.images"
                    f'[["image", {color_by!r}]])), or draw a series with '
                    "res.subset(images=...)"
                )
            raise TableError(f"no column {color_by!r} in res.{what}")
        if select is None:
            chosen = pd.Series(True, index=table.index)
        elif (
            isinstance(select, pd.Series)
            and pd.api.types.is_bool_dtype(select)
            and select.index.equals(table.index)
        ):
            chosen = select
        else:
            raise TableError(
                f"select must be a True / False Series on res.{what}' "
                f"index, e.g. res.{what}.D > 1e-7"
            )
        if not isinstance(image, bool):
            raise ParamsError(
                "image must be True or False: res.draw reads the image "
                "file itself"
            )
        # the edge's own check, here so no figure is made first
        _draw._Style(alpha, linewidth, image_alpha)
        drawn = table.image.isin(names)
        scale = _draw.color_scale(
            table[color_by][drawn & chosen],
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            colors=colors,
        )
        return Drawings(
            {
                name: self._sheet(name, what, color_by, chosen)
                for name in names
            },
            image=image,
            background=background,
            collection=self._origin.collection,
            options={
                **scale,
                "image_alpha": image_alpha,
                "others": others,
                "alpha": alpha,
                "linewidth": linewidth,
                "legend": str(color_by) if legend is True else legend,
            },
        )

    def _sheet(
        self,
        name: str,
        what: str,
        color_by: Hashable,
        chosen: pd.Series,
    ) -> "_Sheet":
        """What ``Drawings`` needs of one image, copied, back in px."""
        row = self.images[self.images.image == name].iloc[0]
        size = float(row.pixel_size)
        mine = self.particles.image == name
        particles = self.particles.loc[mine, ["x", "y", "r"]] / size
        particles["agglomerate_id"] = self.particles.agglomerate_id[mine]
        agglomerates = None
        # under its own name: color_by="r" must not replace r in px
        if what == "particles":
            particles[_VALUES] = self.particles[color_by][mine]
        else:
            mine = self.agglomerates.image == name
            agglomerates = self.agglomerates.loc[mine, ["agglomerate_id"]]
            agglomerates[_VALUES] = self.agglomerates[color_by][mine]
        select = chosen[mine]
        return _Sheet(
            particles,
            agglomerates,
            select.copy(),
            int(row.width),
            int(row.height),
            self._origin.files[name],
        )

    def _image_mask(
        self, images: pd.Series | str | Sequence[str]
    ) -> pd.Series:
        """``images=`` of ``subset`` as a mask on ``self.images``."""
        if isinstance(images, pd.Series):
            if (
                not images.index.equals(self.images.index)
                or images.isna().any()
                or not pd.api.types.is_bool_dtype(images)
            ):
                raise TableError(
                    "images must be a True / False Series on res.images' "
                    "index, e.g. res.images.series == 'a'"
                )
            return images
        names = [images] if isinstance(images, str) else list(images)
        present = set(self.images.image)
        for name in names:
            if name in present:
                continue
            if name in self._origin.names:
                raise ParamsError(
                    f"{name!r} is not in these results (include false, or "
                    "left out by an earlier subset)"
                )
            raise ParamsError(f"no image {name!r} in the collection")
        return self.images.image.isin(names)

    def _table(self, name: str) -> pd.DataFrame:
        if name not in _TABLES:
            raise ParamsError(
                f"table must be 'particles' or 'agglomerates', got {name!r}"
            )
        table: pd.DataFrame = getattr(self, name)
        return table

    @staticmethod
    def _core(
        function: Callable[..., pd.DataFrame], *args: object, **kwargs: object
    ) -> pd.DataFrame:
        """Call a core function; its warnings point at the user's line."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = function(*args, **kwargs)
        # 4: past reemit, this helper and the public method
        reemit(caught, stacklevel=4)
        return out


# Figures a notebook shows inline; a whole collection would be tens of MB.
_THUMBNAILS = 4
# The colour column of the tables ``Drawings`` holds.
_VALUES = "agglpy color_by"


@dataclass(frozen=True)
class _Sheet:
    """One image of ``Drawings``: its rows in px and its file."""

    particles: pd.DataFrame
    agglomerates: pd.DataFrame | None
    select: pd.Series
    width: int
    height: int
    path: Path


class Drawings(Mapping[str, Figure]):
    """Figures of results, one per image, made when looked at.

    A read-only mapping image name → matplotlib ``Figure``, in the
    order of the results: ``d["D7-017"]``, ``list(d)``, ``len(d)``.
    Each access makes a new figure from the rows copied at
    ``res.draw``; with ``image=True`` the image file is read then.
    """

    def __init__(
        self,
        sheets: Mapping[str, _Sheet],
        *,
        image: bool,
        background: ColorType | None,
        collection: Path,
        options: Mapping[str, Any],
    ) -> None:
        """Made by ``Results.draw``; not called directly."""
        self._sheets = dict(sheets)
        self._image = image
        self._background = background
        # where exports go by default, and where they must not go
        self._collection = collection
        self._options = dict(options)

    def __getitem__(self, name: str) -> Figure:
        """A new figure of image ``name``.

        Raises:
            KeyError: If ``name`` was not drawn.
            FileNotFoundError: If the image file is gone
                (``image=True``).
            ImageError: If the image file no longer has the analysed
                size.
        """
        sheet = self._sheets[name]
        image = _image(sheet) if self._image else None
        fig, ax = _draw.figure(
            (sheet.height, sheet.width),
            legend=self._options["legend"] is not False,
            background=self._background,
        )
        if sheet.agglomerates is None:
            _draw.particles(
                ax,
                sheet.particles,
                color_by=_VALUES,
                select=sheet.select,
                image=image,
                **self._options,
            )
        else:
            _draw.agglomerates(
                ax,
                sheet.particles,
                sheet.agglomerates,
                color_by=_VALUES,
                select=sheet.select,
                image=image,
                **self._options,
            )
        return fig

    def __iter__(self) -> Iterator[str]:
        return iter(self._sheets)

    def __len__(self) -> int:
        return len(self._sheets)

    def __contains__(self, name: object) -> bool:
        # Mapping's own answers through d[name]: a figure and a file read
        return name in self._sheets

    def __repr__(self) -> str:
        names = list(self)
        shown = ", ".join(names[:_THUMBNAILS])
        more = ", ..." if len(names) > _THUMBNAILS else ""
        noun = "image" if len(names) == 1 else "images"
        return f"Drawings({len(names)} {noun}: {shown}{more})"

    def export(
        self, folder: PathLike | None = None, *, overwrite: bool = False
    ) -> list[Path]:
        """Write one ``<image>.png`` per image into ``folder``.

        Each PNG has 72 dpi and the figure's size: with
        ``legend=False`` it lies pixel for pixel over the image. No
        record is written: export next to ``res.export`` of the same
        results, whose record covers the drawings.

        Args:
            folder: The folder to write into; created with its parents.
                None gives ``exports/drawings`` in the collection, where
                a repeated export needs ``overwrite=True``. A relative
                path is taken from the current working directory.
            overwrite: Replace PNGs of these images already there.

        Returns:
            The PNG files, absolute, in the order of the results.

        Raises:
            ParamsError: If ``folder`` lies in the collection's work
                folder ``agglpy/``.
            OutputExistsError: If a PNG of a drawn image is there and
                ``overwrite`` is False; nothing is written.
            OutputLockedError: If a PNG to replace is open in another
                program; nothing is changed.
            FileNotFoundError: If an image file is gone (``image=True``);
                nothing is written.
            ImageError: If an image file no longer has the analysed
                size; nothing is written.
        """
        folder = export_folder(self._collection, folder, "drawings")
        paths = [folder / f"{name}.png" for name in self]
        existing = [p for p in paths if p.exists()]
        if existing and not overwrite:
            names = ", ".join(p.name for p in existing)
            raise OutputExistsError(
                f"{folder} already holds {names}; pass overwrite=True to "
                "replace them"
            )
        _check_replaceable(existing)
        if self._image:
            # headers only: a changed image stops it before any PNG
            for sheet in self._sheets.values():
                _check_file(sheet)
                meta = read_image_metadata(sheet.path)
                _check_size(sheet, meta.width, meta.height)
        folder.mkdir(parents=True, exist_ok=True)
        made: list[Path] = []
        try:
            for name, path in zip(self, paths, strict=True):
                fig = self[name]
                new = not path.exists()
                replace_file(
                    path, partial(fig.savefig, format="png", dpi="figure")
                )
                if new:
                    made.append(path)
        except BaseException:
            # the new files only: a replaced PNG can't be put back
            for path in made:
                path.unlink(missing_ok=True)
            raise
        return paths

    def _repr_html_(self) -> str:
        """The first figures for a notebook, and how to see the rest."""
        names = list(self)
        parts = []
        for name in names[:_THUMBNAILS]:
            buffer = io.BytesIO()
            self[name].savefig(buffer, format="png", dpi="figure")
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            parts.append(
                f"<p>{html.escape(name)}</p>"
                f'<img src="data:image/png;base64,{data}" '
                'style="max-width:100%">'
            )
        rest = names[_THUMBNAILS:]
        if rest:
            hint = html.escape(f'd["{rest[0]}"]', quote=False)
            parts.append(
                f"<p>and {len(rest)} more: {hint} or d.export(folder)</p>"
            )
        return "".join(parts)


def _image(sheet: _Sheet) -> Any:
    """The image file of ``sheet``, checked against the analysed size."""
    _check_file(sheet)
    image = read_image(sheet.path)
    height, width = image.shape
    _check_size(sheet, width, height)
    return image


def _check_file(sheet: _Sheet) -> None:
    if not sheet.path.is_file():
        raise FileNotFoundError(f"image file not found: {sheet.path}")


def _check_size(sheet: _Sheet, width: int, height: int) -> None:
    if (width, height) != (sheet.width, sheet.height):
        raise ImageError(
            f"{sheet.path.name} is now {width} x {height} px, analysed at "
            f"{sheet.width} x {sheet.height} px: run analyze() again"
        )


def _kept(table: pd.DataFrame, of: pd.DataFrame) -> dict[str, int]:
    return {"kept": len(table), "of": len(of)}


def _si_units(table: pd.DataFrame) -> dict[str, str]:
    """``m`` / ``m²`` / ``m³`` for agglpy's columns present in ``table``."""
    dimensions = {**_GEOMETRY, **DIMENSIONS}
    return {
        column: _SI[dimensions[column]]
        for column in table.columns
        if dimensions.get(column, 0) > 0
    }
