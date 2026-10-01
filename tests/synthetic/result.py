"""Neutral result format that every adapter returns.

Column names follow the planned particle table of the new core
(roadmap 2.1), so the Phase 2 adapter has almost nothing to convert.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Result:
    """Outcome of running one case through an implementation.

    Attributes:
        particles: One row per circle, indexed by the case key. Columns:
            ``x, y, r`` (px), ``agglomerate_id`` (any label; only equality
            within one result matters), ``type``, ``idj`` (bool).
    """

    particles: pd.DataFrame

    def groups(self) -> frozenset[frozenset[str]]:
        """Agglomerates as a partition of the case keys."""
        by_label = self.particles.groupby("agglomerate_id").groups
        return frozenset(frozenset(keys) for keys in by_label.values())

    def idj_keys(self) -> frozenset[str]:
        """Keys of the particles flagged as internally disjoint."""
        flagged = self.particles.index[self.particles["idj"].astype(bool)]
        return frozenset(flagged)
