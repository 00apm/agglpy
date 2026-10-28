"""``Results``: what ``Collection.analyze`` found, as plain tables."""

import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from numpy.typing import ArrayLike

from agglpy.app._report import reemit
from agglpy.core import distributions
from agglpy.core import metrics as _metrics
from agglpy.errors import ParamsError

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
