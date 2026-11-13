"""The particle table: a plain DataFrame with a checked schema.

One row per circle: ``id`` (int64, unique within one image), centre
``x``, ``y`` and radius ``r`` (float64, finite, ``r > 0``), in px of
the raw image with pixel centres at integers. Analysis adds columns;
an empty table is valid. See *Results* (``docs/results.md``), section
*Particle table*, and *Collection* (``docs/collection.md``), section
*Coordinates*.
"""

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from agglpy.errors import (
    DuplicateParticlesWarning,
    ParticleTableError,
)

PARTICLE_COLUMNS: tuple[str, ...] = ("id", "x", "y", "r")
"""The schema columns of the particle table, in order."""
_FLOAT_COLUMNS = ("x", "y", "r")
_MAX_LISTED_DUPLICATES = 10


def make_particles(
    x: ArrayLike,
    y: ArrayLike,
    r: ArrayLike,
    *,
    ids: ArrayLike | None = None,
) -> pd.DataFrame:
    """Build a validated particle table from coordinate arrays.

    Parameters
    ----------
    x : array_like
        Centre x coordinates in px, 1-D.
    y : array_like
        Centre y coordinates in px, as long as `x`.
    r : array_like
        Radii in px, > 0, as long as `x`.
    ids : array_like, optional
        Particle ids, unique integers; ``1..n`` if omitted.

    Returns
    -------
    DataFrame
        A new particle table: ``id, x, y, r``.

    Raises
    ------
    ParticleTableError
        If the values break the schema.
    """
    try:
        columns = {
            name: np.asarray(values, dtype=float)
            for name, values in (("x", x), ("y", y), ("r", r))
        }
    except (ValueError, TypeError) as exc:
        raise ParticleTableError(f"x, y and r must be numeric: {exc}") from exc
    n = len(columns["x"]) if columns["x"].ndim == 1 else -1
    columns["id"] = np.arange(1, n + 1) if ids is None else np.asarray(ids)
    shapes = {name: values.shape for name, values in columns.items()}
    if any(shape != (n,) for shape in shapes.values()):
        raise ParticleTableError(
            f"x, y, r and ids must be 1-D and of equal length, got shapes "
            f"{shapes}"
        )
    table = pd.DataFrame(
        {
            "id": columns["id"],
            "x": columns["x"],
            "y": columns["y"],
            "r": columns["r"],
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

    Parameters
    ----------
    table : DataFrame
        A table with at least the columns in `PARTICLE_COLUMNS`.
    warn_duplicates : bool, default True
        Warn about near-identical circles (see `find_duplicates`).

    Returns
    -------
    DataFrame
        The validated copy.

    Raises
    ------
    ParticleTableError
        If a column is missing, not numeric where it must be, not
        finite, ``r <= 0``, or ids not integral or not unique.

    Warns
    -----
    DuplicateParticlesWarning
        If `warn_duplicates` is True and near-identical circles are
        found.
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

    if warn_duplicates:
        pairs = find_duplicates(out)
        if pairs:
            warnings.warn(
                _duplicates_message(pairs),
                DuplicateParticlesWarning,
                stacklevel=2,
            )
    return out


def _duplicates_message(pairs: list[tuple[int, int]]) -> str:
    """Return the duplicate warning's text; io prefixes the file."""
    listed = pairs[:_MAX_LISTED_DUPLICATES]
    more = len(pairs) - len(listed)
    suffix = f" and {more} more" if more else ""
    return (
        f"{len(pairs)} pairs of near-identical circles (ids): "
        f"{', '.join(map(str, listed))}{suffix}"
    )


# Shared with the matching of a correction to the detection it started
# from: a circle counts as kept under the same tolerances.
_DUPLICATE_XY_TOL = 2.0
_DUPLICATE_R_REL_TOL = 0.1


def find_duplicates(
    table: pd.DataFrame,
    *,
    xy_tol: float = _DUPLICATE_XY_TOL,
    r_rel_tol: float = _DUPLICATE_R_REL_TOL,
) -> list[tuple[int, int]]:
    """Find pairs of near-identical circles.

    Two circles are duplicates when their centres are at most `xy_tol`
    px apart and their radii differ by at most `r_rel_tol` times the
    larger radius. A small circle inside a large one is not a
    duplicate.

    Parameters
    ----------
    table : DataFrame
        A particle table (columns ``id, x, y, r``, px).
    xy_tol : float, default 2.0
        Largest centre distance in px (inclusive).
    r_rel_tol : float, default 0.1
        Largest relative radius difference (inclusive).

    Returns
    -------
    list of tuple of int
        Pairs of ids ``(smaller id, larger id)``, sorted.

    Notes
    -----
    The result is an indicator: it depends on the tolerances, whose
    defaults are first guesses until they are calibrated on corrected
    images.
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
