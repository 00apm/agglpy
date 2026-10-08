"""Neutral result format that every adapter returns.

Column names follow the planned particle and agglomerate tables of the
new core (roadmap 2.1), so the Phase 2 adapter has almost nothing to
convert.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Result:
    """Outcome of running one case through an implementation.

    An adapter fills only what its supported checks read
    (``adapters.SUPPORTED_CHECKS``).

    Attributes:
        particles: One row per circle, indexed by the case key. Columns:
            ``x, y, r`` (px), ``agglomerate_id`` (any label; only equality
            within one result matters), ``type``, ``enclosed`` (bool).
        agglomerates: One row per agglomerate, indexed by
            ``agglomerate_id``. Columns: ``members`` (frozenset of keys),
            ``type``, and the properties in px / px³ named as in
            ``cases.PROPERTIES`` (``volume``, ``D``, ``member_count``,
            ``D_mean``, ``D_std``, ``enclosed_count`` and the
            ``*_with_hidden`` values).
        summary: Per-image summary metrics (names as in
            ``cases.SUMMARY``; diameters in px).
    """

    particles: pd.DataFrame
    agglomerates: pd.DataFrame
    summary: dict

    def groups(self) -> frozenset[frozenset[str]]:
        """Agglomerates as a partition of the case keys."""
        by_label = self.particles.groupby("agglomerate_id").groups
        return frozenset(frozenset(keys) for keys in by_label.values())

    def enclosed_keys(self) -> frozenset[str]:
        """Keys of the particles flagged as enclosed."""
        flagged = self.particles.index[self.particles["enclosed"].astype(bool)]
        return frozenset(flagged)

    def particle_types(self) -> dict[str, str]:
        """Particle type per key."""
        return self.particles["type"].to_dict()

    def agglomerate_types(self) -> dict[frozenset[str], str]:
        """Agglomerate type, keyed by the agglomerate's members."""
        return dict(
            zip(
                self.agglomerates["members"],
                self.agglomerates["type"],
                strict=True,
            )
        )

    def agglomerate_properties(self) -> dict[frozenset[str], dict]:
        """All agglomerate columns except ``members``, keyed by members."""
        table = self.agglomerates.set_index("members")
        return {members: row.to_dict() for members, row in table.iterrows()}
