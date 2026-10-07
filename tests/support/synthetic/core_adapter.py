"""Run synthetic cases through the new core (``core.agglomerates``, …).

The core works in px, so a case goes in as it is. The adapter fills
only what the core can do so far (``adapters.SUPPORTED_CHECKS``) and
grows with 2.4 (types, agglomerate properties) and 2.5 (summary).
"""

from pathlib import Path

import pandas as pd

from agglpy.core.agglomerates import find_agglomerates

from .cases import Case
from .result import Result


def run_core(
    case: Case,
    tmp_path: Path,
    pixel_size: float = 1.0,
) -> Result:
    """Run one case through the core and convert the outcome to a Result.

    Args:
        case: The synthetic case.
        tmp_path: Not used: the core writes no files.
        pixel_size: Not used yet: the core works in px; physical units
            come in at the output (2.5).
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
    grouped = find_agglomerates(table)  # rows stay in input order
    particles = pd.DataFrame(
        {
            "x": grouped["x"].to_numpy(),
            "y": grouped["y"].to_numpy(),
            "r": grouped["r"].to_numpy(),
            "agglomerate_id": grouped["agglomerate_id"].to_numpy(),
            # Result keeps the case vocabulary (idj) until 2.4.
            "idj": grouped["enclosed"].to_numpy(),
        },
        index=pd.Index(keys),
    )
    return Result(particles=particles, agglomerates=pd.DataFrame(), summary={})
