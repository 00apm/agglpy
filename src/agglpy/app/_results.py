"""``Results``: what ``Collection.analyze`` found, as plain tables."""

import dataclasses
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from numpy.typing import ArrayLike

from agglpy.app._report import reemit
from agglpy.app._sources import PathLike
from agglpy.core import distributions
from agglpy.core import metrics as _metrics
from agglpy.core.agglomerates import select_agglomerates
from agglpy.core.properties import DIMENSIONS
from agglpy.errors import ParamsError, TableError
from agglpy.export import write_results

__all__ = ["Origin", "Results"]


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
        folder: PathLike,
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
            folder: The folder to write into; created if missing.
            tables: Extra tables made from these results, e.g.
                ``{"summary": res.summary(by="series")}``; written
                after the three, without units.
            overwrite: Replace an earlier export in ``folder``.

        Returns:
            The folder.

        Raises:
            ParamsError: If an extra table is called ``images``,
                ``agglomerates`` or ``particles``.
            OutputExistsError: If ``folder`` has an export and
                ``overwrite`` is False.
            OutputLockedError: If a file to replace is open in another
                program.
        """
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
