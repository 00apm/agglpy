"""``Results``: what ``Collection.analyze`` found, as plain tables."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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


class Results:
    """The particles and agglomerates of a collection, in SI units.

    Attributes:
        images: One row per analysed image.
        particles: One row per particle, keyed by ``(image, id)``.
        agglomerates: One row per agglomerate, keyed by ``(image,
            agglomerate_id)``.
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
