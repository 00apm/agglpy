"""Properties of particles and agglomerates, in px.

``particle_properties`` adds each particle's size to the particle
table; ``agglomerate_properties`` builds the agglomerate table, one row
per ``agglomerate_id``. Both take the table from
``agglomerates.find_agglomerates`` and never modify it.

Every value describes the circle model: each particle is a sphere
whose projection is its detected circle. Enclosed particles are members
everywhere: they count in every sum, count and statistic; only the
``*_with_hidden`` values count them a second time
(``docs/methodology.md``, section 7).
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.errors import ParticleTableError


def particle_properties(table: pd.DataFrame) -> pd.DataFrame:
    """Add each particle's size, as a sphere, to the particle table.

    Adds ``D = 2r`` (px), ``area = πr²`` (px², the circle) and
    ``volume = (4/3)πr³`` (px³, the sphere).

    Args:
        table: A particle table with at least the column ``r``.

    Returns:
        A copy with the three columns, replacing any from an earlier
        run; every other column, the row order and the index are kept.

    Raises:
        ParticleTableError: If ``r`` is missing.
    """
    _require(table, ("r",))
    out = table.copy()
    r = out["r"].to_numpy(dtype=np.float64)
    out["D"] = 2 * r
    out["area"] = np.pi * r**2
    out["volume"] = _sphere_volume(r)
    return out


def _sphere_volume(r: NDArray[np.float64]) -> NDArray[np.float64]:
    return 4 / 3 * np.pi * r**3


def _require(table: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ParticleTableError(f"missing columns: {missing}")
