"""The particle table: a plain DataFrame with a checked schema.

One row per detected circle, in px:

========  =======  ==================================================
column    dtype    meaning
========  =======  ==================================================
id        int64    particle id, unique within one image
x, y      float64  centre (px), finite
r         float64  radius (px), finite and > 0
source    object   where the table came from: "hct", "manual", …
========  =======  ==================================================

``source`` is known for certain when a table is read. Which circles a
manual correction kept, edited or added is not stored: ImageJ
renumbers circles, so it can only be inferred by matching geometry
(D-036).

Analysis adds columns later (agglomerate id, type, enclosed flag).
An empty table is valid: zero particles is a result (D-022).
"""

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from agglpy.errors import DuplicateParticlesWarning, ParticleTableError

PARTICLE_COLUMNS: tuple[str, ...] = ("id", "x", "y", "r", "source")
_FLOAT_COLUMNS = ("x", "y", "r")
_MAX_LISTED_DUPLICATES = 10


def make_particles(
    x: ArrayLike,
    y: ArrayLike,
    r: ArrayLike,
    *,
    source: str = "hct",
    ids: ArrayLike | None = None,
) -> pd.DataFrame:
    """Build a validated particle table from coordinate arrays.

    Args:
        x: Centre x coordinates in px.
        y: Centre y coordinates in px.
        r: Radii in px.
        source: Value of the ``source`` column for every row.
        ids: Particle ids; ``1..n`` if omitted.

    Returns:
        A new particle table (see the module docstring).

    Raises:
        ParticleTableError: If the values break the schema.
    """
    x_arr = np.asarray(x, dtype=float)
    n = len(x_arr)
    id_arr = np.arange(1, n + 1) if ids is None else np.asarray(ids)
    table = pd.DataFrame(
        {
            "id": id_arr,
            "x": x_arr,
            "y": np.asarray(y, dtype=float),
            "r": np.asarray(r, dtype=float),
            "source": [source] * n,
        }
    )
    return validate_particles(table)


def validate_particles(
    table: pd.DataFrame, *, warn_duplicates: bool = True
) -> pd.DataFrame:
    """Check a particle table and return a clean copy.

    The copy has the schema columns first, with their dtypes, then any
    extra columns in their original order, and a fresh ``RangeIndex``.
    The input is not modified.

    Args:
        table: A table with at least the columns in ``PARTICLE_COLUMNS``.
        warn_duplicates: Warn about near-identical circles
            (see ``find_duplicates``).

    Returns:
        The validated copy.

    Raises:
        ParticleTableError: If a column is missing, not numeric where
            it must be, not finite, ``r <= 0``, ids not integral or not
            unique, or ``source`` has missing values.
    """
    missing = [c for c in PARTICLE_COLUMNS if c not in table.columns]
    if missing:
        raise ParticleTableError(f"missing columns: {missing}")

    extra = [c for c in table.columns if c not in PARTICLE_COLUMNS]
    out = table.loc[:, [*PARTICLE_COLUMNS, *extra]].reset_index(drop=True)

    for column in _FLOAT_COLUMNS:
        out[column] = _as_float(out[column], column)
        if not np.isfinite(out[column].to_numpy()).all():
            raise ParticleTableError(f"column {column!r} must be finite")
    if (out["r"] <= 0).any():
        raise ParticleTableError("column 'r' must be > 0")

    out["id"] = _as_int(out["id"])
    duplicated = out.loc[out["id"].duplicated(), "id"].unique().tolist()
    if duplicated:
        raise ParticleTableError(
            f"column 'id' must be unique, repeated: {sorted(duplicated)}"
        )

    if out["source"].isna().any():
        raise ParticleTableError("column 'source' has missing values")
    out["source"] = out["source"].astype(str).astype(object)

    if warn_duplicates:
        pairs = find_duplicates(out)
        if pairs:
            listed = pairs[:_MAX_LISTED_DUPLICATES]
            more = len(pairs) - len(listed)
            suffix = f" and {more} more" if more else ""
            warnings.warn(
                f"{len(pairs)} pairs of near-identical circles (ids): "
                f"{', '.join(map(str, listed))}{suffix}",
                DuplicateParticlesWarning,
                stacklevel=2,
            )
    return out


def find_duplicates(
    table: pd.DataFrame,
    *,
    xy_tol: float = 2.0,
    r_rel_tol: float = 0.1,
) -> list[tuple[int, int]]:
    """Find pairs of near-identical circles.

    Two circles are duplicates when their centres are at most
    ``xy_tol`` px apart and their radii differ by at most
    ``r_rel_tol`` times the larger radius. A small circle inside a
    large one is not a duplicate. The result is an indicator: it
    depends on the tolerances, whose defaults are first guesses until
    they are calibrated on corrected images.

    Args:
        table: A particle table (columns ``id, x, y, r``).
        xy_tol: Largest centre distance in px (inclusive).
        r_rel_tol: Largest relative radius difference (inclusive).

    Returns:
        Pairs of ids ``(smaller id, larger id)``, sorted.
    """
    if len(table) < 2:
        return []
    xy = table[["x", "y"]].to_numpy(dtype=float)
    r = table["r"].to_numpy(dtype=float)
    ids = table["id"].to_numpy()
    candidates = cKDTree(xy).query_pairs(xy_tol, output_type="ndarray")
    if len(candidates) == 0:
        return []
    i, j = candidates[:, 0], candidates[:, 1]
    close = np.abs(r[i] - r[j]) <= r_rel_tol * np.maximum(r[i], r[j])
    pairs = {
        (int(min(a, b)), int(max(a, b)))
        for a, b in zip(ids[i[close]], ids[j[close]], strict=True)
    }
    return sorted(pairs)


def _as_float(values: pd.Series, column: str) -> pd.Series:
    try:
        return pd.to_numeric(values, errors="raise").astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise ParticleTableError(
            f"column {column!r} must be numeric: {exc}"
        ) from exc


def _as_int(values: pd.Series) -> pd.Series:
    numbers = _as_float(values, "id")
    array = numbers.to_numpy()
    if not np.isfinite(array).all() or (array != np.round(array)).any():
        raise ParticleTableError("column 'id' must hold integers")
    return numbers.astype(np.int64)
