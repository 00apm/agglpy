"""Run synthetic cases through the new core (``agglpy.core``).

The core works in px, so a case goes in as it is. The library has no
classification (D-042): agglomerate types come from the recipe in
``examples/agglomerate_types.py``. The adapter fills only what the
core can do so far (``adapters.SUPPORTED_CHECKS``); the summary comes
with 2.5.
"""

from pathlib import Path

import pandas as pd

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.properties import agglomerate_properties

from .cases import Case
from .result import Result

# Importing the recipe also runs its small example cells (percent
# format, D-045): a few circles, no output.
from examples.agglomerate_types import agglomerate_types


def run_core(
    case: Case,
    tmp_path: Path,
    pixel_size: float = 1.0,
) -> Result:
    """Run one case through the core and convert the outcome to a Result.

    Args:
        case: The synthetic case.
        tmp_path: Not used: the core writes no files.
        pixel_size: Not used: the core works in px; physical units
            come in where images are pooled (2.8).
    """
    keys = [c.key for c in case.circles]
    table = pd.DataFrame(
        {
            "id": range(1, len(keys) + 1),
            "x": [c.x for c in case.circles],
            "y": [c.y for c in case.circles],
            "r": [c.r for c in case.circles],
            "source": "synthetic",
        }
    )
    found = find_agglomerates(table)  # rows stay in input order
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
    agglomerates = agglomerate_properties(found).set_index("agglomerate_id")
    members = particles.groupby("agglomerate_id").groups
    agglomerates.insert(
        0, "members", [frozenset(members[i]) for i in agglomerates.index]
    )
    agglomerates["type"] = agglomerate_types(agglomerates, case.threshold)
    # Particle types as agglpy 0.4 gave them (tests only): in a collector
    # agglomerate the largest member is the collector and the others
    # are attached to it; otherwise a particle has its agglomerate's type.
    particle_type = particles["agglomerate_id"].map(agglomerates["type"])
    largest = particles.groupby("agglomerate_id")["r"].transform("max")
    attached = (particle_type == "collector") & (particles["r"] < largest)
    particles["type"] = particle_type.mask(attached, "attached2coll")
    return Result(particles=particles, agglomerates=agglomerates, summary={})
