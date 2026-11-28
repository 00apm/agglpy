"""Run a synthetic case through the core, answers keyed by circle name.

The cases name their circles (``"a"``, ``"b"``); the core numbers them.
`run_case` gives the core's answers by the case's names, so a test
compares them with the hand-worked ones directly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.metrics import summary
from agglpy.core.properties import agglomerate_properties, particle_properties

from .cases import Case

# Importing the recipe also runs its small example cells (percent
# format): a few circles, no output.
from examples.agglomerate_types import agglomerate_types


@dataclass(frozen=True)
class Result:
    """The core's answers for one case, in px.

    Attributes:
        particles: One row per circle, indexed by the case key:
            ``x, y, r``, ``agglomerate_id``, ``enclosed``.
        agglomerates: One row per agglomerate, indexed by
            ``agglomerate_id``: ``members`` (frozenset of keys),
            ``type`` (the recipe's) and the agglomerate properties.
        summary: The summary metrics of the case as one image.
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
        """Agglomerate columns except ``members``, keyed by members."""
        table = self.agglomerates.set_index("members")
        return {members: row.to_dict() for members, row in table.iterrows()}


def run_case(case: Case) -> Result:
    """Group the case's circles, measure them and summarise them."""
    keys = [c.key for c in case.circles]
    table = pd.DataFrame(
        {
            "id": range(1, len(keys) + 1),
            "x": [c.x for c in case.circles],
            "y": [c.y for c in case.circles],
            "r": [c.r for c in case.circles],
        }
    )
    found, contacts = find_agglomerates(table)  # rows in input order
    particles = pd.DataFrame(
        {
            "x": found["x"].to_numpy(),
            "y": found["y"].to_numpy(),
            "r": found["r"].to_numpy(),
            "agglomerate_id": found["agglomerate_id"].to_numpy(),
            "enclosed": found["enclosed"].to_numpy(),
        },
        index=pd.Index(keys),
    )
    agglomerates = agglomerate_properties(found, contacts)
    images = pd.DataFrame({"image": [case.name], "fov_area": [np.nan]})
    row = summary(
        images,
        particle_properties(found).assign(image=case.name),
        agglomerates.assign(image=case.name),
    ).iloc[0]
    agglomerates = agglomerates.set_index("agglomerate_id")
    members = particles.groupby("agglomerate_id").groups
    agglomerates.insert(
        0, "members", [frozenset(members[i]) for i in agglomerates.index]
    )
    agglomerates["type"] = agglomerate_types(agglomerates, case.threshold)
    return Result(particles, agglomerates, row.to_dict())
