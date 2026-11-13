"""Population metrics: pooled per group of images, or across images.

The functions are unit-agnostic: they assume one common unit for all
rows. Pooling px tables of images with different pixel sizes gives
wrong sizes; convert each image with
`agglpy.core.properties.to_physical` first. All metrics use the visible
values only, never the ``*_with_hidden`` properties.

See *Results* (``docs/results.md``), section *Metrics*.
"""

import warnings
from collections.abc import Iterable
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from agglpy.core._images import (
    Grouping,
    check_images,
    group_images,
    image_positions,
    names,
    numbers,
    require,
    with_labels,
)
from agglpy.core.structure import SEGMENT_TYPES
from agglpy.errors import MissingImagesWarning, ParamsError, TableError
from agglpy.params import _number

# Built-in metrics by family, in the column order of ``summary``.
_FAMILIES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "counts": (
            "N_primary",
            "N_aerosol",
            "N_pp1",
            "N_ppA",
            "N_aggl",
            "N_merges",
        ),
        "ratios": (
            "Ra",
            "agglomerated_fraction",
            "merge_ratio",
            "n_ppA",
            "n_ppP",
        ),
        "contacts": ("N_contacts", "N_loops", "coordination_mean"),
        "structure": (
            "N_cores",
            "N_small_cores",
            "N_links",
            "N_dangling_ends",
            *(f"{s.replace(' ', '_')}_fraction" for s in SEGMENT_TYPES),
        ),
        "per_area": (
            "coverage",
            "N_primary_per_area",
            "N_aerosol_per_area",
            "N_aggl_per_area",
        ),
        "particle_size": (
            "particle_D_mean",
            "particle_D_std",
            "particle_D10",
            "particle_D50",
            "particle_D90",
            "particle_SMD",
        ),
        "aerosol_size": (
            "aerosol_D_mean",
            "aerosol_D_std",
            "aerosol_D10",
            "aerosol_D50",
            "aerosol_D90",
        ),
        "member_count": (
            "aerosol_member_count_std",
            "aerosol_member_count_q10",
            "aerosol_member_count_q50",
            "aerosol_member_count_q90",
        ),
    }
)


# The columns of structure.structure_properties the family reads.
_STRUCTURE_COLUMNS = {
    "particles": ["segment_type"],
    "agglomerates": [
        "core_count",
        "small_core_member_count",
        "link_count",
        "dangling_end_count",
    ],
}

_QUANTILES = [0.1, 0.5, 0.9]
_MAX_LISTED = 10


def summary(
    images: pd.DataFrame,
    particles: pd.DataFrame,
    agglomerates: pd.DataFrame,
    by: str | Iterable[str] | None = None,
    *,
    metrics: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute the built-in metrics of each group of images, pooled.

    Counts are summed over the images of a group, ratios come from the
    summed counts (never means of per-image ratios) and size
    descriptors from the pooled rows.

    Parameters
    ----------
    images : DataFrame
        Images table with ``image`` (unique) and, for ``per_area``,
        ``fov_area`` in the tables' unit squared; other columns can
        form groups.
    particles : DataFrame
        Particle table with ``image``; ``D`` for ``particle_size``,
        ``segment_type`` for ``structure``.
    agglomerates : DataFrame
        Agglomerate table with ``image`` and ``member_count``; ``D``
        for ``aerosol_size``, ``area`` for ``per_area``,
        ``contact_count`` for ``contacts``, the structure counts for
        ``structure``.
    by : str or list of str, optional
        None (all images, one row), ``"image"``, an images-table column
        or a list of them: one row per existing combination; an empty
        cell is its own group.
    metrics : str or list of str, optional
        Families, in this column order: ``"counts"``, ``"ratios"``,
        ``"contacts"``, ``"structure"``, ``"per_area"``,
        ``"particle_size"``, ``"aerosol_size"``, ``"member_count"``.
        None gives all, ``"structure"`` only when the tables have its
        columns (`agglpy.core.structure.structure_properties`).

    Returns
    -------
    DataFrame
        One row per group: the `by` columns, ``n_images`` (blank
        images included), then the metrics.

    Raises
    ------
    TableError
        If a table lacks a column or names an unknown image, if the
        particle and agglomerate tables do not match (in each image the
        agglomerates' ``member_count`` must add up to the particle
        rows; filter both with
        `agglpy.core.agglomerates.select_agglomerates`), or if a `by`
        column has the name of an output column.
    ParamsError
        If a family name is unknown, or `by` is not column names or
        repeats one.

    See Also
    --------
    summary_across_images : The same metrics per image, then their
        mean over the images.

    Notes
    -----
    No particles give zero counts and NaN ratios and sizes; particles
    without agglomerates give ``n_ppA`` NaN and ``Ra`` 0. Divisions
    follow numpy (0 / 0 is NaN). See *Results* (``docs/results.md``),
    section *Metrics*.
    """
    families = _families(metrics, particles, agglomerates)
    check_images(images, fov_area="per_area" in families)
    groups = group_images(images, by)
    require(particles, ["image"], "particles")
    require(agglomerates, ["image", "member_count"], "agglomerates")
    if "particle_size" in families:
        require(particles, ["D"], "particles")
    if "aerosol_size" in families:
        require(agglomerates, ["D"], "agglomerates")
    if "per_area" in families:
        require(agglomerates, ["area"], "agglomerates")
    if "contacts" in families:
        require(agglomerates, ["contact_count"], "agglomerates")
    if "structure" in families:
        require(particles, _STRUCTURE_COLUMNS["particles"], "particles")
        require(
            agglomerates, _STRUCTURE_COLUMNS["agglomerates"], "agglomerates"
        )
    p_image = image_positions(images, particles, "particles")
    a_image = image_positions(images, agglomerates, "agglomerates")
    members = numbers(agglomerates, "member_count", "agglomerates")
    _check_members(images, p_image, a_image, members)
    # Group of each particle and each agglomerate, via its image.
    p_group = groups.codes[p_image]
    a_group = groups.codes[a_image]
    k = groups.k

    n_primary = np.bincount(p_group, minlength=k)
    n_aerosol = np.bincount(a_group, minlength=k)
    n_pp1 = np.bincount(a_group[members == 1], minlength=k)
    counts = {
        "N_primary": n_primary,
        "N_aerosol": n_aerosol,
        "N_pp1": n_pp1,
        "N_ppA": n_primary - n_pp1,
        "N_aggl": n_aerosol - n_pp1,
        "N_merges": n_primary - n_aerosol,
    }
    columns: dict[str, NDArray[np.generic]] = {
        "n_images": np.bincount(groups.codes, minlength=k)
    }
    # 0 / 0 -> NaN is intended (an image without particles).
    with np.errstate(divide="ignore", invalid="ignore"):
        if "counts" in families:
            columns.update(counts)
        if "ratios" in families:
            columns.update(_ratios(counts))
        if "contacts" in families:
            columns.update(_contacts(agglomerates, a_group, k, counts))
        if "structure" in families:
            columns.update(
                _structure(particles, agglomerates, p_group, a_group, k)
            )
        if "per_area" in families:
            columns.update(
                _per_area(images, groups, agglomerates, a_group, counts)
            )
    if "particle_size" in families:
        d = numbers(particles, "D", "particles")
        columns.update(_size("particle_D", d, p_group, k))
        d2 = np.bincount(p_group, weights=d * d, minlength=k)
        d3 = np.bincount(p_group, weights=d * d * d, minlength=k)
        with np.errstate(divide="ignore", invalid="ignore"):
            columns["particle_SMD"] = d3 / d2
    if "aerosol_size" in families:
        d = numbers(agglomerates, "D", "agglomerates")
        columns.update(_size("aerosol_D", d, a_group, k))
    if "member_count" in families:
        described = _describe(members, a_group, k)
        columns["aerosol_member_count_std"] = described["std"].to_numpy()
        for q in _QUANTILES:
            name = f"aerosol_member_count_q{round(q * 100)}"
            columns[name] = described[q].to_numpy()
    return with_labels(groups.labels, pd.DataFrame(columns))


def summary_across_images(
    images: pd.DataFrame,
    particles: pd.DataFrame,
    agglomerates: pd.DataFrame,
    by: str | Iterable[str] | None = None,
    *,
    metrics: str | Iterable[str] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute each metric per image, then its mean ± CI over images.

    ``summary(..., by="image")``, then `values_across_images` over the
    groups of `by`. Each image weighs equally: this describes the
    typical image, not the pooled population (`summary`).

    Parameters
    ----------
    images : DataFrame
        As in `summary`.
    particles : DataFrame
        As in `summary`.
    agglomerates : DataFrame
        As in `summary`.
    by : str or list of str, optional
        As in `summary`.
    metrics : str or list of str, optional
        As in `summary`.
    confidence : float, default 0.95
        Level of the confidence interval, between 0 and 1.

    Returns
    -------
    DataFrame
        A long table: the `by` columns, ``metric``, ``mean``, ``std``,
        ``n``, ``ci_low``, ``ci_high``; one row per group and metric.

    Raises
    ------
    TableError
        If a table is invalid, as in `summary`.
    ParamsError
        If a family name, `by` or `confidence` is invalid.

    See Also
    --------
    summary : The pooled metrics.
    values_across_images : The same for any per-image values.
    """
    per_image = summary(
        images, particles, agglomerates, by="image", metrics=metrics
    ).drop(columns="n_images")
    values = [c for c in per_image.columns if c != "image"]
    return values_across_images(
        images, per_image, by, values=values, confidence=confidence
    )


def values_across_images(
    images: pd.DataFrame,
    per_image: pd.DataFrame,
    by: str | Iterable[str] | None = None,
    *,
    keys: str | Iterable[str] = (),
    values: str | Iterable[str] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute mean, std, n and confidence interval of per-image values.

    For each group of images and each value column of `per_image`:
    ``mean``, sample ``std``, ``n`` (images with a value) and the
    t-based interval ``mean ± t(q, n - 1) · std / √n`` with
    ``q = (1 + confidence) / 2``. Each image weighs equally ("the
    typical image"); a pooled value of the whole population is
    `summary`.

    Parameters
    ----------
    images : DataFrame
        Images table with ``image`` (unique); the groups come from it.
    per_image : DataFrame
        Table with ``image``, the `keys` columns and the value columns;
        one row per image (and key).
    by : str or list of str, optional
        As in `summary`.
    keys : str or list of str, default ()
        Columns that tell several values of one image apart (e.g. size
        class and basis); they are kept in the output.
    values : str or list of str, optional
        The value columns; None: every other column of `per_image` not
        in `images` (columns of the images table describe the image and
        are not values).
    confidence : float, default 0.95
        Level of the interval, between 0 and 1.

    Returns
    -------
    DataFrame
        A long table: the `by` columns, the `keys`, ``metric`` (the
        value column's name), ``mean``, ``std``, ``n``, ``ci_low``,
        ``ci_high``.

    Raises
    ------
    TableError
        If `per_image` lacks a column, names an unknown image, repeats
        an image (and key) or has a value column that is not numeric,
        or if a `by` column has the name of an output column.
    ParamsError
        If `confidence` is not between 0 and 1, or `by`, `keys` or
        `values` are not column names.

    Warns
    -----
    MissingImagesWarning
        If images of `images` have no row.

    Notes
    -----
    Repeats raise: a particle or agglomerate table must be aggregated
    per image first. An image of `images` missing from `per_image` has
    NaN values, with the warning. NaN values are left out per column,
    so ``n`` is per column; ``n = 1`` gives NaN ``std`` and interval.
    """
    level = _number("confidence", confidence)
    if not 0 < level < 1:
        raise ParamsError(f"confidence must be in (0, 1), got {confidence!r}")
    check_images(images)
    groups = group_images(images, by)
    key_columns = names(keys, "keys")
    require(per_image, key_columns, "per_image")
    image_positions(images, per_image, "per_image")
    _check_one_row_per_image(per_image, key_columns)
    if values is None:
        values = [
            c
            for c in per_image.columns
            if c != "image"
            and c not in key_columns
            and c not in images.columns
        ]
    else:
        values = names(values, "values")
        require(per_image, values, "per_image")
    for column in values:
        if not pd.api.types.is_numeric_dtype(per_image[column]):
            raise TableError(
                f"per_image: value column {column!r} must be numeric"
            )
    _warn_missing_images(images, per_image)

    # Every image with every key combination, so a missing value is NaN
    # (left out of n), not a row that silently isn't there.
    grid = images[["image"]].reset_index(drop=True)
    grid["_group"] = groups.codes
    if key_columns:
        combos = per_image[key_columns].drop_duplicates()
        grid = grid.merge(combos, how="cross")
    data = grid.merge(
        per_image[["image", *key_columns, *values]],
        on=["image", *key_columns],
        how="left",
    ).sort_values("_group", kind="stable")
    # observed=True: no rows for categories of a key that never occur
    grouped = data.groupby(
        ["_group", *key_columns], sort=False, dropna=False, observed=True
    )[values]
    out = pd.DataFrame(
        {
            "mean": grouped.mean().stack(future_stack=True),
            "std": grouped.std().stack(future_stack=True),
            "n": grouped.count().stack(future_stack=True),
        }
    )
    out.index = out.index.set_names("metric", level=-1)
    out = out.reset_index()
    out["n"] = out["n"].astype(np.int64)
    out[["ci_low", "ci_high"]] = _interval(out, level)
    labels = groups.labels.iloc[out.pop("_group")].reset_index(drop=True)
    return with_labels(labels, out)


def _families(
    metrics: str | Iterable[str] | None,
    particles: pd.DataFrame,
    agglomerates: pd.DataFrame,
) -> list[str]:
    if metrics is None:
        # structure is opt-in: plain tables keep their summary.
        present = all(
            c in table.columns
            for table, name in (
                (particles, "particles"),
                (agglomerates, "agglomerates"),
            )
            for c in _STRUCTURE_COLUMNS[name]
        )
        return [f for f in _FAMILIES if f != "structure" or present]
    chosen = names(metrics, "metrics")
    unknown = [f for f in chosen if f not in _FAMILIES]
    if unknown:
        raise ParamsError(
            f"metrics: unknown families {unknown}; known: {list(_FAMILIES)}"
        )
    return [f for f in _FAMILIES if f in chosen]


def _ratios(
    counts: dict[str, NDArray[np.int64]],
) -> dict[str, NDArray[np.float64]]:
    primary = counts["N_primary"]
    # Explicit: 0 merges / (0 - 1) particles would give -0.0, not NaN.
    merge_ratio = np.full(len(primary), np.nan)
    np.divide(
        counts["N_merges"], primary - 1, out=merge_ratio, where=primary >= 2
    )
    return {
        "Ra": counts["N_aggl"] / primary,
        "agglomerated_fraction": 1 - counts["N_pp1"] / primary,
        "merge_ratio": merge_ratio,
        "n_ppA": counts["N_ppA"] / counts["N_aggl"],
        "n_ppP": primary / counts["N_aerosol"],
    }


def _contacts(
    agglomerates: pd.DataFrame,
    a_group: NDArray[np.intp],
    k: int,
    counts: dict[str, NDArray[np.int64]],
) -> dict[str, NDArray[np.generic]]:
    contact_count = numbers(agglomerates, "contact_count", "agglomerates")
    n_contacts = np.bincount(
        a_group, weights=contact_count, minlength=k
    ).astype(np.int64)
    return {
        "N_contacts": n_contacts,
        "N_loops": n_contacts - counts["N_merges"],
        "coordination_mean": 2 * n_contacts / counts["N_primary"],
    }


def _structure(
    particles: pd.DataFrame,
    agglomerates: pd.DataFrame,
    p_group: NDArray[np.intp],
    a_group: NDArray[np.intp],
    k: int,
) -> dict[str, NDArray[np.generic]]:
    def total(column: str) -> NDArray[np.int64]:
        values = numbers(agglomerates, column, "agglomerates")
        return np.bincount(a_group, weights=values, minlength=k).astype(
            np.int64
        )

    small = numbers(agglomerates, "small_core_member_count", "agglomerates")
    out: dict[str, NDArray[np.generic]] = {
        "N_cores": total("core_count"),
        "N_small_cores": np.bincount(a_group[small > 0], minlength=k).astype(
            np.int64
        ),
        "N_links": total("link_count"),
        "N_dangling_ends": total("dangling_end_count"),
    }
    segment = particles["segment_type"].to_numpy()
    n_primary = np.bincount(p_group, minlength=k)
    for label in SEGMENT_TYPES:
        name = f"{label.replace(' ', '_')}_fraction"
        out[name] = (
            np.bincount(p_group[segment == label], minlength=k) / n_primary
        )
    return out


def _per_area(
    images: pd.DataFrame,
    groups: Grouping,
    agglomerates: pd.DataFrame,
    a_group: NDArray[np.intp],
    counts: dict[str, NDArray[np.int64]],
) -> dict[str, NDArray[np.float64]]:
    # bincount keeps NaN: one image without fov_area makes its group's
    # per-area values NaN instead of dividing by a partial area.
    fov = np.bincount(
        groups.codes,
        weights=numbers(images, "fov_area", "images"),
        minlength=groups.k,
    )
    area = np.bincount(
        a_group,
        weights=numbers(agglomerates, "area", "agglomerates"),
        minlength=groups.k,
    )
    return {
        "coverage": area / fov,
        "N_primary_per_area": counts["N_primary"] / fov,
        "N_aerosol_per_area": counts["N_aerosol"] / fov,
        "N_aggl_per_area": counts["N_aggl"] / fov,
    }


def _size(
    prefix: str, d: NDArray[np.float64], group: NDArray[np.intp], k: int
) -> dict[str, NDArray[np.generic]]:
    described = _describe(d, group, k)
    return {
        f"{prefix}_mean": described["mean"].to_numpy(),
        f"{prefix}_std": described["std"].to_numpy(),
        f"{prefix}10": described[0.1].to_numpy(),
        f"{prefix}50": described[0.5].to_numpy(),
        f"{prefix}90": described[0.9].to_numpy(),
    }


def _describe(
    values: NDArray[np.float64], group: NDArray[np.intp], k: int
) -> pd.DataFrame:
    """Return mean, sample std and quantiles per group; NaN when empty.

    Columns ``mean``, ``std``, ``0.1``, ``0.5``, ``0.9`` (floats); one
    row per group 0 … k-1.
    """
    grouped = pd.Series(values, dtype=np.float64).groupby(group)
    # linear interpolation between sorted values (pandas default)
    quantiles = grouped.quantile(np.array(_QUANTILES)).unstack()
    out = pd.DataFrame({"mean": grouped.mean(), "std": grouped.std()}).join(
        quantiles
    )
    return out.reindex(
        index=pd.RangeIndex(k), columns=["mean", "std", *_QUANTILES]
    )


def _check_members(
    images: pd.DataFrame,
    p_image: NDArray[np.intp],
    a_image: NDArray[np.intp],
    members: NDArray[np.float64],
) -> None:
    # Every particle is a member of one agglomerate of its image, so the
    # member counts add up to the particle rows. Tables filtered apart
    # (only the agglomerates, or only the particles) would give wrong
    # counts and ratios without a sign; a gap in member_count too.
    n = len(images)
    declared = np.bincount(a_image, weights=members, minlength=n)
    found = np.bincount(p_image, minlength=n)
    wrong = images["image"].to_numpy()[declared != found].tolist()
    if wrong:
        raise TableError(
            f"particles and agglomerates do not match in images "
            f"{wrong[:_MAX_LISTED]}: the agglomerates' member_count must "
            f"add up to the particle rows of each image; filter both "
            f"tables with select_agglomerates"
        )


def _check_one_row_per_image(
    per_image: pd.DataFrame, key_columns: list[str]
) -> None:
    repeated = per_image.duplicated(["image", *key_columns])
    if repeated.any():
        listed = per_image.loc[repeated, "image"].unique().tolist()
        raise TableError(
            f"per_image: one row per image"
            f"{' and key' if key_columns else ''} expected, repeated: "
            f"{listed[:_MAX_LISTED]}; aggregate a particle or agglomerate "
            f"table per image first (groupby('image'))"
        )


def _warn_missing_images(
    images: pd.DataFrame, per_image: pd.DataFrame
) -> None:
    missing = images.loc[~images["image"].isin(per_image["image"]), "image"]
    if len(missing):
        listed = missing.tolist()[:_MAX_LISTED]
        more = len(missing) - len(listed)
        suffix = f" and {more} more" if more > 0 else ""
        warnings.warn(
            f"{len(missing)} image(s) have no row in per_image, their "
            f"values are missing (NaN): {listed}{suffix}; if a value is "
            f"a count, add those images with 0",
            MissingImagesWarning,
            stacklevel=3,
        )


def _interval(out: pd.DataFrame, level: float) -> NDArray[np.float64]:
    n = out["n"].to_numpy(dtype=np.float64)
    mean = out["mean"].to_numpy(dtype=np.float64)
    std = out["std"].to_numpy(dtype=np.float64)
    # n = 1 has std NaN, so the interval is NaN too (never 0).
    with np.errstate(divide="ignore", invalid="ignore"):
        half = stats.t.ppf((1 + level) / 2, n - 1) * std / np.sqrt(n)
    return np.column_stack([mean - half, mean + half])
