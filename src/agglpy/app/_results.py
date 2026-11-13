"""`Results`: what `Collection.analyze` found, as plain tables."""

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

import numpy as np
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
from agglpy.core.region import Bounds, region_bounds
from agglpy.core.structure import (
    AGGLOMERATE_COLUMNS,
    PARTICLE_COLUMNS,
    structure_properties,
)
from agglpy.errors import (
    ImageError,
    OutputExistsError,
    ParamsError,
    TableError,
)
from agglpy.export import _check_replaceable, write_results
from agglpy.io import read_image, read_image_metadata
from agglpy.params import RegionParams

__all__ = ["Drawings", "Origin", "Results"]


@dataclass(frozen=True)
class Origin:
    """What a `Results` was made from, for its export record.

    Attributes
    ----------
    analyzed : str
        When ``analyze()`` ran (ISO 8601).
    collection : Path
        The collection folder.
    sheet : str
        The sheet's file name.
    images : mapping
        The record entry of each image, as plain YAML data.
    files : mapping
        The image files, resolved, for drawing them.
    excluded : tuple of mapping
        The sheet rows with ``include`` false: ``image`` and
        ``note``.
    steps : tuple of mapping
        The subset steps taken, in order.
    names : tuple of str
        Every image of the sheet.
    structure : mapping, optional
        The settings of `Results.with_structure`, or None.
    """

    analyzed: str
    collection: Path
    sheet: str
    images: Mapping[str, Mapping[str, object]]
    files: Mapping[str, Path]
    excluded: tuple[Mapping[str, str], ...]
    steps: tuple[Mapping[str, object], ...]
    names: tuple[str, ...]
    structure: Mapping[str, object] | None = None


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

    Made by `Collection.analyze`, `subset` and `with_structure`; not
    created directly. Lengths are in m, areas in m², volumes in m³,
    ``pixel_size`` in m per pixel. The tables are the stored frames,
    not copies: columns added to them are seen by the methods, `draw`
    and `subset`; rows and agglpy's columns are not changed by hand
    (a `subset` leaves images or agglomerates out).

    Parameters
    ----------
    images, particles, agglomerates, contacts : DataFrame
        The tables, kept as the attributes of the same names.
    origin : Origin
        What the results were made from, for the export record.

    Attributes
    ----------
    images : DataFrame
        One row per analysed image: agglpy's columns, the
        user columns of the sheet, then ``note``.
    particles : DataFrame
        One row per particle, keyed by ``(image, id)``.
    agglomerates : DataFrame
        One row per agglomerate, keyed by ``(image,
        agglomerate_id)``. Ids are unique only within an image
        and change with a new detection or correction.
    contacts : DataFrame
        One row per possible contact: ``image``,
        ``agglomerate_id``, the particle ids ``id_a`` (the larger)
        and ``id_b``, ``inside`` (``id_b`` lies inside ``id_a``).

    See Also
    --------
    agglpy.Collection.analyze : Make the results of a collection.

    Notes
    -----
    See *Results* (``docs/results.md``), sections *Units and keys*,
    *Images table*, *Particle table*, *Agglomerate table* and
    *Contacts table*.
    """

    def __init__(
        self,
        images: pd.DataFrame,
        particles: pd.DataFrame,
        agglomerates: pd.DataFrame,
        contacts: pd.DataFrame,
        origin: Origin,
    ) -> None:
        self.images = images
        self.particles = particles
        self.agglomerates = agglomerates
        self.contacts = contacts
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
        """Compute the built-in metrics of each group of images, pooled.

        `agglpy.core.metrics.summary` on these tables.

        Parameters
        ----------
        by : str or iterable of str, optional
            ``"image"``, a column of `images` (a sheet column groups
            images directly) or a list of them; None gives one row.
        metrics : str or iterable of str, optional
            Family names; None gives every family, ``"structure"``
            only when the tables have its columns.

        Returns
        -------
        DataFrame
            One row per group: the `by` columns, ``n_images``, then
            the metrics.

        Raises
        ------
        TableError
            If ``"structure"`` is chosen on results without the
            structure columns (`with_structure` adds them), or a `by`
            column is missing or has the name of an output column.
        ParamsError
            If a family name is unknown.

        See Also
        --------
        agglpy.core.metrics.summary : The computation and the metrics.
        """
        metrics = self._check_structure(metrics)
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
        """Compute the metrics per image, then mean ± CI over images.

        `agglpy.core.metrics.summary_across_images` on these tables.
        Each image weighs equally.

        Parameters
        ----------
        by : str or iterable of str, optional
            As in `summary`.
        metrics : str or iterable of str, optional
            As in `summary`.
        confidence : float, default 0.95
            Level of the confidence interval, between 0 and 1.

        Returns
        -------
        DataFrame
            A long table: the `by` columns, ``metric``, ``mean``,
            ``std``, ``n``, ``ci_low``, ``ci_high``; one row per group
            and metric.

        Raises
        ------
        TableError
            If ``"structure"`` is chosen on results without the
            structure columns, or a `by` column is missing or has the
            name of an output column.
        ParamsError
            If a family name is unknown or `confidence` is not
            between 0 and 1.

        See Also
        --------
        agglpy.core.metrics.summary_across_images : The computation.
        """
        metrics = self._check_structure(metrics)
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
        """Compute mean, std, n and a confidence interval over images.

        `agglpy.core.metrics.values_across_images` with `images`.
        Each image weighs equally.

        Parameters
        ----------
        per_image : DataFrame
            ``image``, the `keys` columns and the value columns; one
            row per image (and key).
        by : str or iterable of str, optional
            As in `summary`.
        keys : str or iterable of str, default ()
            Columns that tell several values of one image apart
            (e.g. size class and basis); kept in the output.
        values : str or iterable of str, optional
            The value columns; None takes every other column of
            `per_image` not in `images`.
        confidence : float, default 0.95
            Level of the confidence interval, between 0 and 1.

        Returns
        -------
        DataFrame
            A long table: the `by` columns, the `keys`, ``metric``
            (the value column's name), ``mean``, ``std``, ``n``,
            ``ci_low``, ``ci_high``.

        Raises
        ------
        TableError
            If `per_image` lacks a column, names an unknown image,
            repeats an image (and key) or has a value column that is
            not numeric, or a `by` column is missing or has the name
            of an output column.
        ParamsError
            If `confidence` is not between 0 and 1.

        Warns
        -----
        MissingImagesWarning
            If images of `images` have no row in `per_image`.

        See Also
        --------
        agglpy.core.metrics.values_across_images : The computation.
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
        """Compute the distribution of a column over size classes.

        `agglpy.core.distributions.distribution` on `images` and one
        of the tables.

        Parameters
        ----------
        table : {"particles", "agglomerates"}
            The table that holds `column`.
        column : str
            The column to classify, e.g. ``"D"`` (m).
        edges : array_like
            Increasing class edges, in the unit of `column` (e.g.
            `agglpy.size_classes`).
        by : str or iterable of str, optional
            As in `summary`.
        closed : {"right", "left"}, default "right"
            Classes ``(a, b]`` or ``[a, b)``.
        weights : str or iterable of str, default ()
            Columns to weight by, e.g. ``"volume"``; each adds one
            ``basis``.

        Returns
        -------
        DataFrame
            A long table: the `by` columns, ``left, mid, right,
            width``, ``basis``, ``amount``, ``fraction``,
            ``cumulative``, ``density``, ``log_density``.

        Raises
        ------
        ParamsError
            If `table` is another name, `edges` or `closed` is
            invalid, or a weight column is called ``"number"``.
        TableError
            If a column is missing or not numeric, a weight is not
            finite on a counted value, or a `by` column has the name
            of an output column.

        Warns
        -----
        ValuesNotCountedWarning
            If values were left out (outside the classes or NaN).

        See Also
        --------
        agglpy.core.distributions.distribution : The computation and
            the output columns.
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
        """Compute each class's fraction per image, then mean ± CI.

        `agglpy.core.distributions.distribution_across_images` on
        `images` and one of the tables. An image without counted
        values is left out of the means and of ``n``.

        Parameters
        ----------
        table : {"particles", "agglomerates"}
            As in `distribution`.
        column : str
            As in `distribution`.
        edges : array_like
            As in `distribution`.
        by : str or iterable of str, optional
            As in `summary`.
        closed : {"right", "left"}, default "right"
            As in `distribution`.
        weights : str or iterable of str, default ()
            As in `distribution`.
        confidence : float, default 0.95
            Level of the confidence interval, between 0 and 1.

        Returns
        -------
        DataFrame
            A long table: the `by` columns, ``left, mid, right,
            width``, ``basis``, ``metric`` (``"fraction"``), ``mean``,
            ``std``, ``n``, ``ci_low``, ``ci_high``.

        Raises
        ------
        ParamsError
            If `table` is another name, `edges` or `closed` is
            invalid, a weight column is called ``"number"``, or
            `confidence` is not between 0 and 1.
        TableError
            If a column is missing or not numeric, a weight is not
            finite on a counted value, or a `by` column has the name
            of an output column.

        Warns
        -----
        ValuesNotCountedWarning
            If values were left out (outside the classes or NaN).

        See Also
        --------
        agglpy.core.distributions.distribution_across_images : The
            computation.
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
        """Return a part of the results, the step kept for the export.

        `images` keeps those images with their particles, agglomerates
        and contacts; `agglomerates` keeps those agglomerates with
        their member particles and contacts, and every image (an image
        with nothing left counts as zeros). Both: one step, images
        first. The results it is called on stay as they are.

        Parameters
        ----------
        images : Series of bool or str or sequence of str, optional
            A True / False Series on the index of `images`, e.g.
            ``res.images.series == "a"``, or a name or list of
            names (kept in the order of the results).
        agglomerates : Series of bool, optional
            A True / False Series on the index of `agglomerates`,
            e.g. ``~res.agglomerates.touches_edge``.
        label : str, optional
            Names the step in the export record.

        Returns
        -------
        Results
            New results, their tables renumbered from 0.

        Raises
        ------
        ParamsError
            If neither `images` nor `agglomerates` is given, a name
            is not in these results, or no image is left.
        TableError
            If a mask is not a True / False Series on its
            table's index.
        """
        if images is None and agglomerates is None:
            raise ParamsError("subset needs images= or agglomerates=")
        table, particles, aggls, contacts = (
            self.images,
            self.particles,
            self.agglomerates,
            self.contacts,
        )
        keep = None if images is None else self._image_mask(images)
        if agglomerates is not None:
            # the same rows as images first: both keep an intersection
            particles, aggls = select_agglomerates(
                particles, aggls, agglomerates
            )
            key = ["image", "agglomerate_id"]
            contacts = contacts[
                pd.MultiIndex.from_frame(contacts[key]).isin(
                    pd.MultiIndex.from_frame(aggls[key])
                )
            ]
        if keep is not None:
            if not keep.any():
                raise ParamsError("no image left")
            table = table[keep]
            particles = particles[particles.image.isin(table.image)]
            aggls = aggls[aggls.image.isin(table.image)]
            contacts = contacts[contacts.image.isin(table.image)]
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
            contacts.reset_index(drop=True),
            origin,
        )

    def with_structure(self, *, min_core_members: int = 7) -> "Results":
        """Return new results with the structure columns added.

        Particles get ``PARTICLE_COLUMNS`` and agglomerates
        ``AGGLOMERATE_COLUMNS`` of `agglpy.core.structure`, image by
        image; lengths are in m. Columns from an earlier call are
        replaced; the results it is called on stay as they are.

        Parameters
        ----------
        min_core_members : int, default 7
            Particles a group of triangles needs to be a core; 3 or
            more.

        Returns
        -------
        Results
            New results with the structure columns.

        Raises
        ------
        ParamsError
            If `min_core_members` is not an integer of 3 or more.

        See Also
        --------
        agglpy.core.structure.structure_properties : The computation.

        Notes
        -----
        `subset` keeps whole agglomerates, so the columns stay valid
        after it; ``segment_id`` then keeps its numbers, while `subset`
        first numbers them from 0. Draw the segments with
        ``draw(what="particles", color_by="segment_type")``. See
        *Structure of agglomerates* (``docs/agglomerate-structure.md``).
        """
        particles: list[pd.DataFrame] = []
        aggls: list[pd.DataFrame] = []
        p_rows = self.particles.groupby("image").indices
        a_rows = self.agglomerates.groupby("image").indices
        c_rows = self.contacts.groupby("image").indices
        none = np.empty(0, dtype=np.intp)
        for name in self.images.image:
            p, a = structure_properties(
                self.particles.iloc[p_rows.get(name, none)],
                self.agglomerates.iloc[a_rows.get(name, none)],
                self.contacts.iloc[c_rows.get(name, none)].drop(
                    columns="image"
                ),
                min_core_members=min_core_members,
            )
            particles.append(p)
            aggls.append(a)
        origin = dataclasses.replace(
            self._origin,
            structure={
                "min_core_members": int(min_core_members),
                "base": "largest member, ties: lower id",
            },
        )
        return Results(
            self.images.copy(),
            _in_order(particles, self.particles),
            _in_order(aggls, self.agglomerates),
            self.contacts.copy(),
            origin,
        )

    def export(
        self,
        folder: PathLike | None = None,
        *,
        tables: Mapping[str, pd.DataFrame] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write the tables, their units and a record into a folder.

        Writes ``results.xlsx`` (the sheets ``images``,
        ``agglomerates``, ``particles``, ``contacts``, the extra
        `tables`, then ``units`` and ``info``), ``csv/`` (one CSV per
        table and ``units.csv``) and ``record.yml``, the record of
        where the numbers come from.

        Parameters
        ----------
        folder : str or Path, optional
            The folder to write into; created if missing. A
            folder named for each export is the recommended way;
            None gives ``exports/results`` in the collection, where
            a repeated export needs ``overwrite=True``. A relative
            path is taken from the current working directory.
        tables : mapping of str to DataFrame, optional
            Extra tables made from these results, e.g.
            ``{"summary": res.summary(by="series")}``; written
            after the four, without units.
        overwrite : bool, default False
            Replace an earlier export in `folder`.

        Returns
        -------
        Path
            The folder, absolute.

        Raises
        ------
        ParamsError
            If an extra table is called ``images``,
            ``agglomerates``, ``particles`` or ``contacts``, or
            `folder` lies in the collection's work folder
            ``agglpy/``.
        OutputExistsError
            If `folder` has an export and `overwrite` is False.
        OutputLockedError
            If a file to replace is open in another program.

        See Also
        --------
        agglpy.export.write_results : The writer.
        Drawings.export : Drawings of the same results, for a folder
            next to this one.

        Notes
        -----
        See *Results* (``docs/results.md``), section *Export*.
        """
        folder = export_folder(self._origin.collection, folder, "results")
        extra = dict(tables or {})
        written = {
            "images": self.images,
            "agglomerates": self.agglomerates,
            "particles": self.particles,
            "contacts": self.contacts,
        }
        for name in written:
            if name in extra:
                raise ParamsError(
                    f"tables: {name!r} is a table of the results; give "
                    "the extra table another name"
                )
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
        if origin.structure is not None:
            record["structure"] = dict(origin.structure)
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
        full_image: bool = False,
    ) -> "Drawings":
        """Draw the particles or agglomerates of each image, in px.

        One colour scale for all drawn images: numbers share one
        `vmin` / `vmax` over the selected values of every image
        (unless given), labels one label → colour mapping, so a series
        compares at a glance (draw one with ``res.subset(images=...)``
        first).

        Parameters
        ----------
        images : str or sequence of str, optional
            A name or list of names; None draws every image.
        what : {"particles", "agglomerates"}, default "agglomerates"
            ``"particles"`` (each particle coloured by a column
            of `particles`) or ``"agglomerates"`` (each
            agglomerate by a column of `agglomerates`).
        color_by : Hashable
            The column of that table that gives the colours.
            A column of `images` has to be merged onto the table
            first.
        select : Series of bool, optional
            A True / False Series on that table's index; the
            others are drawn as thin `others` outlines.
        image : bool, default True
            Draw the image file underneath, read when a figure
            is made.
        image_alpha : float, default 1.0
            Opacity of the image, 0 to 1: below 1 the
            background shows through, e.g. ``background="black",
            image_alpha=0.4`` darkens the image so the circles
            stand out.
        background : color or None, default "white"
            The figure's colour, under the image: any
            matplotlib colour (``"white"``, ``"#FF000033"``, ``(0,
            1, 0, 0.47)`` with values in 0-1) or None for a
            transparent figure.
        others : color or None, default "grey"
            As in `agglpy.draw.particles`.
        cmap : str or Colormap, optional
            As in `agglpy.draw.particles`.
        norm : Normalize, optional
            As in `agglpy.draw.particles`.
        vmin : float, optional
            As in `agglpy.draw.particles`; the smallest selected
            value of all drawn images when None.
        vmax : float, optional
            As in `agglpy.draw.particles`; the largest selected
            value of all drawn images when None.
        colors : mapping, optional
            As in `agglpy.draw.particles`.
        alpha : float, default 0.5
            As in `agglpy.draw.particles`.
        linewidth : float, default 1.0
            As in `agglpy.draw.particles`.
        legend : bool or str, default True
            A colorbar or legend in a strip right of the image;
            a string titles it; False draws the figure without the
            strip, the image's size exactly.
        full_image : bool, default False
            Draw the whole image; False draws the analysed
            region (the region of the params ``analyze()`` used),
            circles cut at its edge.

        Returns
        -------
        Drawings
            The figures: image name → figure.

        Raises
        ------
        ParamsError
            If no image or a name not in these results is
            given, `what` is another name, `color_by` is a column
            of `images` only, `image` or `full_image` is not True /
            False, `alpha` / `image_alpha` is not within 0-1,
            `linewidth` is negative, or an option does not fit the
            column.
        TableError
            If `color_by` is not a column of the table or
            `select` not a True / False Series on its index.

        See Also
        --------
        agglpy.draw.particles : Draw one image's particles.
        agglpy.draw.agglomerates : Draw one image's agglomerates.

        Notes
        -----
        The figures are made when looked at (``d[name]``) or
        exported, not held: a figure changed by hand is saved with
        ``fig.savefig``. See *Results* (``docs/results.md``), section
        *Drawings*.
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
        if not isinstance(full_image, bool):
            raise ParamsError(
                f"full_image must be True or False, got {full_image!r}"
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
            full_image=full_image,
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
        """Copy what `Drawings` needs of one image, back in px."""
        row = self.images[self.images.image == name].iloc[0]
        size = float(row.pixel_size)
        mine = self.particles.image == name
        particles = self.particles.loc[mine, ["x", "y", "r"]] / size
        particles["id"] = self.particles.id[mine]
        particles["agglomerate_id"] = self.particles.agglomerate_id[mine]
        agglomerates = contacts = None
        # under its own name: color_by="r" must not replace r in px
        if what == "particles":
            particles[_VALUES] = self.particles[color_by][mine]
        else:
            mine = self.agglomerates.image == name
            agglomerates = self.agglomerates.loc[mine, ["agglomerate_id"]]
            agglomerates[_VALUES] = self.agglomerates[color_by][mine]
            contacts = self.contacts[self.contacts.image == name].drop(
                columns="image"
            )
        select = chosen[mine]
        width, height = int(row.width), int(row.height)
        region = self._origin.images[name]["region"]
        assert isinstance(region, Mapping)
        return _Sheet(
            particles,
            agglomerates,
            contacts,
            select.copy(),
            width,
            height,
            self._origin.files[name],
            region_bounds(RegionParams(**region), width, height),
        )

    def _image_mask(
        self, images: pd.Series | str | Sequence[str]
    ) -> pd.Series:
        """Turn ``images=`` of `subset` into a mask on `images`."""
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

    def _check_structure(
        self, metrics: str | Iterable[str] | None
    ) -> list[str] | None:
        """Read the family names once.

        Refuse the family ``"structure"`` on results without its
        columns.
        """
        if metrics is None:
            return None
        chosen = [metrics] if isinstance(metrics, str) else list(metrics)
        present = set(PARTICLE_COLUMNS) <= set(self.particles.columns) and set(
            AGGLOMERATE_COLUMNS
        ) <= set(self.agglomerates.columns)
        if "structure" in chosen and not present:
            raise TableError(
                "metrics 'structure' needs the structure columns: call "
                "res.with_structure() first"
            )
        return chosen

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
        """Call a core function, its warnings at the user's line."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = function(*args, **kwargs)
        # 4: past reemit, this helper and the public method
        reemit(caught, stacklevel=4)
        return out


# Figures a notebook shows inline; all of a collection: tens of MB.
_THUMBNAILS = 4
# The colour column of the tables ``Drawings`` holds.
_VALUES = "agglpy color_by"


@dataclass(frozen=True)
class _Sheet:
    """One image of ``Drawings``: its rows in px and its file."""

    particles: pd.DataFrame
    agglomerates: pd.DataFrame | None
    contacts: pd.DataFrame | None
    select: pd.Series
    width: int
    height: int
    path: Path
    bounds: Bounds


class Drawings(Mapping[str, Figure]):
    """Figures of results, one per image, made when looked at.

    Made by `Results.draw`; not created directly. A read-only mapping
    image name → matplotlib ``Figure``, in the order of the results:
    ``d["D7-017"]``, ``list(d)``, ``len(d)``. Each access makes a new
    figure from the rows copied at `Results.draw`; with ``image=True``
    the image file is read then.

    Parameters
    ----------
    sheets : mapping
        Per image name, its rows in px, its size, file and region.
    image : bool
        Draw the image file underneath.
    full_image : bool
        Draw the whole image instead of the analysed region.
    background : color or None
        The figure's colour; None for transparent.
    collection : Path
        The collection folder: the default export folder lies in
        it, and its work folder is refused.
    options : mapping
        The drawing options of `agglpy.draw.particles` /
        `agglpy.draw.agglomerates`, the colour scale included.

    See Also
    --------
    Results.draw : Make the drawings.
    """

    def __init__(
        self,
        sheets: Mapping[str, _Sheet],
        *,
        image: bool,
        full_image: bool,
        background: ColorType | None,
        collection: Path,
        options: Mapping[str, Any],
    ) -> None:
        self._sheets = dict(sheets)
        self._image = image
        self._full_image = full_image
        self._background = background
        # where exports go by default, and where they must not go
        self._collection = collection
        self._options = dict(options)

    def __getitem__(self, name: str) -> Figure:
        """Make a new figure of image `name`.

        Raises
        ------
        KeyError
            If `name` was not drawn.
        FileNotFoundError
            If the image file is gone
            (``image=True``).
        ImageError
            If the image file no longer has the analysed
            size.
        """
        sheet = self._sheets[name]
        image = _image(sheet) if self._image else None
        if self._full_image:
            shape, origin = (sheet.height, sheet.width), (0.0, 0.0)
        else:
            # the image is drawn whole underneath: the limits cut it
            b = sheet.bounds
            shape = (round(b.y_max - b.y_min), round(b.x_max - b.x_min))
            origin = (b.x_min + 0.5, b.y_min + 0.5)
        fig, ax = _draw.figure(
            shape,
            legend=self._options["legend"] is not False,
            background=self._background,
            origin=origin,
        )
        if sheet.agglomerates is None or sheet.contacts is None:
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
                sheet.contacts,
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
        # Mapping's default calls d[name]: a figure and a file read
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
        """Write one ``<image>.png`` per image into a folder.

        Each PNG has 72 dpi and the figure's size: with
        ``legend=False`` it lies pixel for pixel over the analysed
        region (over the image with ``full_image=True``). No record is
        written: export next to `Results.export` of the same
        results, whose record covers the drawings.

        Parameters
        ----------
        folder : str or Path, optional
            The folder to write into; created with its parents.
            None gives ``exports/drawings`` in the collection, where
            a repeated export needs ``overwrite=True``. A relative
            path is taken from the current working directory.
        overwrite : bool, default False
            Replace PNGs of these images already there.

        Returns
        -------
        list of Path
            The PNG files, absolute, in the order of the results.

        Raises
        ------
        ParamsError
            If `folder` lies in the collection's work
            folder ``agglpy/``.
        OutputExistsError
            If a PNG of a drawn image is there and
            `overwrite` is False; nothing is written.
        OutputLockedError
            If a PNG to replace is open in another
            program; nothing is changed.
        FileNotFoundError
            If an image file is gone (``image=True``);
            nothing is written.
        ImageError
            If an image file no longer has the analysed
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
        """Show the first figures in a notebook, and how to see more."""
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
    """Read the image file of `sheet`; check its size."""
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


def _in_order(parts: list[pd.DataFrame], of: pd.DataFrame) -> pd.DataFrame:
    """Join `parts` (row subsets of `of`) in the rows' order."""
    joined = pd.concat(parts)
    # The index of each part is the rows' index in ``of``.
    position = of.index.get_indexer(joined.index)
    return joined.iloc[np.argsort(position, kind="stable")]


def _kept(table: pd.DataFrame, of: pd.DataFrame) -> dict[str, int]:
    return {"kept": len(table), "of": len(of)}


def _si_units(table: pd.DataFrame) -> dict[str, str]:
    """Return the SI unit of each of agglpy's columns in `table`."""
    dimensions = {**_GEOMETRY, **DIMENSIONS}
    return {
        column: _SI[dimensions[column]]
        for column in table.columns
        if dimensions.get(column, 0) > 0
    }
