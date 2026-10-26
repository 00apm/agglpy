"""The tables ``detect`` and ``import_particles`` return."""

from collections.abc import Mapping, Sequence

import pandas as pd

__all__ = ["REPORT_COLUMNS", "report"]

REPORT_COLUMNS = ("image", "outcome", "particle_count", "message", "warnings")


def report(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """One row per image; ``particle_count`` empty unless written."""
    table = pd.DataFrame(list(rows), columns=list(REPORT_COLUMNS))
    return table.astype(
        {
            "image": object,
            "outcome": object,
            "particle_count": "Int64",
            "message": object,
            "warnings": object,
        }
    )
