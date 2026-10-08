"""Run synthetic cases through the new core (``agglpy.core``).

The core works in px, so a case goes in as it is. The library has no
classification (D-042): agglomerate types come from the recipe in
``examples/agglomerate_types.py``. The summary is computed the way the
application will: tables converted to physical units with the pixel
size, then ``summary``; its diameters are converted back to px for the
comparison.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.distributions import distribution, size_classes
from agglpy.core.metrics import summary
from agglpy.core.properties import (
    agglomerate_properties,
    particle_properties,
    to_physical,
)
from agglpy.errors import ValuesNotCountedWarning

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
        pixel_size: Converts the tables before the summary; the
            grouping and the properties stay in px.
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
    return Result(
        particles=particles,
        agglomerates=agglomerates,
        summary=_summary(found, case.name, pixel_size),
    )


# Summary columns in a length unit, converted back to px for the cases.
_DIAMETERS = (
    "particle_D_mean",
    "particle_D_std",
    "particle_D10",
    "particle_D50",
    "particle_D90",
    "particle_SMD",
    "aerosol_D_mean",
    "aerosol_D_std",
    "aerosol_D10",
    "aerosol_D50",
    "aerosol_D90",
)


def _summary(found: pd.DataFrame, image: str, pixel_size: float) -> dict:
    """One image's summary, as the application will compute it."""
    images = pd.DataFrame({"image": [image], "fov_area": [np.nan]})
    particles = to_physical(
        particle_properties(found).assign(image=image), pixel_size
    )
    agglomerates = to_physical(
        agglomerate_properties(found).assign(image=image), pixel_size
    )
    row = summary(images, particles, agglomerates).iloc[0].to_dict()
    for name in _DIAMETERS:
        row[name] /= pixel_size
    return row


def core_size_classes(start: float, end: float, **kwargs) -> np.ndarray:
    """``size_classes`` as it is (the 1.4 checks use its vocabulary)."""
    return size_classes(start, end, **kwargs)


def core_distribution(values: list[float], edges: np.ndarray) -> pd.DataFrame:
    """Number distribution of the values, as one image."""
    images = pd.DataFrame({"image": ["case"]})
    table = pd.DataFrame({"image": "case", "value": values})
    # The 1.4 case leaves a value out on purpose; the warning is checked
    # in the unit tests.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValuesNotCountedWarning)
        return distribution(images, table, "value", edges)
