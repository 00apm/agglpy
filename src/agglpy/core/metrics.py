"""Population metrics: pooled per group of images, or across images.

``summary`` pools every image of a group: counts are summed, ratios
come from the summed counts (never means of per-image ratios) and size
descriptors from the pooled rows. ``summary_across_images`` computes
the same metrics per image, then their mean ± confidence interval over
the images of each group (``values_across_images``, which also takes
the user's own per-image values).

The functions are unit-agnostic: they assume one common unit for all
rows. Pooling px tables of images with different pixel sizes gives
wrong sizes; convert each image with ``properties.to_physical`` first.
All metrics use the visible values only, never the ``*_with_hidden``
properties (D-046).
"""

from collections.abc import Iterable
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.core._images import (
    Grouping,
    check_images,
    group_images,
    image_positions,
    names,
    numbers,
    require,
)
from agglpy.errors import ParamsError

# Built-in metrics by family, in the column order of ``summary``.
_FAMILIES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "counts": ("N_primary", "N_aerosol", "N_pp1", "N_ppA", "N_aggl"),
        "ratios": ("Ra", "agglomerated_fraction", "n_ppA", "n_ppP"),
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


_QUANTILES = [0.1, 0.5, 0.9]


def summary(
    images: pd.DataFrame,
    particles: pd.DataFrame,
    agglomerates: pd.DataFrame,
    by: str | Iterable[str] | None = None,
    *,
    metrics: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Built-in metrics of each group of images, pooled.

    Families (``metrics=``; default all) and their columns:

    - ``counts``: ``N_primary`` (primary particles), ``N_aerosol``
      (aerosol particles: agglomerates and single particles), ``N_pp1``
      (single particles), ``N_ppA = N_primary - N_pp1`` (particles in
      agglomerates), ``N_aggl = N_aerosol - N_pp1`` (agglomerates of
      two or more particles).
    - ``ratios``, from the pooled counts: ``Ra = N_aggl / N_primary``,
      the agglomeration ratio of Gotoh et al. (1996), equal to theirs
      when no particle is lost or added; ``agglomerated_fraction =
      1 - N_pp1 / N_primary``, the number fraction of primary particles
      bound in agglomerates (ER in agglpy <= 0.4); ``n_ppA = N_ppA /
      N_aggl``; ``n_ppP = N_primary / N_aerosol``.
    - ``per_area``, over the summed ``fov_area``: ``coverage`` (summed
      agglomerate ``area`` / FoV area: the covered surface fraction;
      agglomerates never overlap, so this is the exact union of all
      circles) and ``N_primary_per_area``, ``N_aerosol_per_area``,
      ``N_aggl_per_area``. Circles crossing the image border count in
      full, so these are biased high for large agglomerates on small
      images.
    - ``particle_size``: mean, sample std, D10, D50, D90 (number-based,
      linear interpolation) and Sauter mean diameter ``ΣD³ / ΣD²`` of
      the particle ``D``.
    - ``aerosol_size``: mean, std, D10, D50, D90 of the agglomerate
      ``D`` over all aerosol particles (single particles included).
    - ``member_count``: std and 10/50/90 % quantiles of
      ``member_count`` over all aerosol particles (the mean is
      ``n_ppP``).

    No particles give zero counts and NaN ratios and sizes; particles
    without agglomerates give ``n_ppA`` NaN and ``Ra`` 0. Divisions
    follow numpy (0 / 0 is NaN).

    Args:
        images: The images table (``agglpy.core._images``).
        particles: Particle table with ``image``, ``D``.
        agglomerates: Agglomerate table with ``image``,
            ``member_count``, ``D``, ``area``.
        by: ``None`` (all images, one row), ``"image"``, an images-table
            column or a list of them (one row per existing
            combination; an empty cell is its own group).
        metrics: Family names; narrows the table.

    Returns:
        One row per group: the ``by`` columns, ``n_images`` (blank
        images included), then the metrics.

    Raises:
        TableError: If a table lacks a column or names an unknown image.
        ParamsError: If a family name is unknown.
    """
    families = _families(metrics)
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
    # Group of each particle and each agglomerate, via its image.
    p_group = groups.codes[image_positions(images, particles, "particles")]
    a_group = groups.codes[
        image_positions(images, agglomerates, "agglomerates")
    ]
    k = groups.k
    members = numbers(agglomerates, "member_count", "agglomerates")

    n_primary = np.bincount(p_group, minlength=k)
    n_aerosol = np.bincount(a_group, minlength=k)
    n_pp1 = np.bincount(a_group[members == 1], minlength=k)
    counts = {
        "N_primary": n_primary,
        "N_aerosol": n_aerosol,
        "N_pp1": n_pp1,
        "N_ppA": n_primary - n_pp1,
        "N_aggl": n_aerosol - n_pp1,
    }
    columns: dict[str, NDArray[np.generic]] = {
        "n_images": np.bincount(groups.codes, minlength=k)
    }
    # 0 / 0 -> NaN is intended (an image without particles, D-022).
    with np.errstate(divide="ignore", invalid="ignore"):
        if "counts" in families:
            columns.update(counts)
        if "ratios" in families:
            columns.update(_ratios(counts))
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
    out = pd.concat([groups.labels, pd.DataFrame(columns)], axis=1)
    return out


def _families(metrics: str | Iterable[str] | None) -> list[str]:
    if metrics is None:
        return list(_FAMILIES)
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
    return {
        "Ra": counts["N_aggl"] / primary,
        "agglomerated_fraction": 1 - counts["N_pp1"] / primary,
        "n_ppA": counts["N_ppA"] / counts["N_aggl"],
        "n_ppP": primary / counts["N_aerosol"],
    }


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
    """Mean, sample std and quantiles per group; NaN for empty groups.

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
