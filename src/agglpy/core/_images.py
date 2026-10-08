"""The images table and the groups of images that ``by=`` forms.

Every core function that groups takes the images table first, so an
image without particles or agglomerates keeps its row (D-022). Until
Phase 3 makes it ``Results.images``, it is a plain DataFrame:

==========  ========  ================================================
column      dtype     meaning
==========  ========  ================================================
image       any       image key: unique, no gaps; the particle and
                      agglomerate tables refer to it in their own
                      ``image`` column
fov_area    float     analysed area of the image (field of view), in
                      the tables' unit squared; NaN when unknown
                      (particles-only entries). Read only by the
                      ``per_area`` metrics of ``summary``
others      any       group columns for ``by=``
==========  ========  ================================================
"""

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from agglpy.errors import ParamsError, TableError

_MAX_LISTED = 10


class Grouping(NamedTuple):
    """Groups of images formed by ``by=``.

    ``codes[i]`` is the group (0 … k-1) of row ``i`` of the images
    table; ``labels`` has one row per group, in group order, with the
    ``by`` columns (no columns for ``by=None``).
    """

    columns: list[str]
    codes: NDArray[np.intp]
    labels: pd.DataFrame

    @property
    def k(self) -> int:
        return len(self.labels)


def names(value: str | Iterable[str] | None, argument: str) -> list[str]:
    """Column names given as ``None``, one name or a sequence of names.

    A single string is one name, never a sequence of letters.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    out = list(value) if isinstance(value, Iterable) else [value]
    if not all(isinstance(name, str) for name in out):
        raise ParamsError(f"{argument} must be column names, got {value!r}")
    return out


def check_images(images: pd.DataFrame, *, fov_area: bool = False) -> None:
    """Check the images table (module docstring).

    Raises:
        TableError: If ``image`` is missing, has gaps or repeats, or,
            with ``fov_area=True``, ``fov_area`` is missing, not
            numeric or not > 0 where given.
    """
    require(images, ["image"], "images")
    keys = images["image"]
    if keys.isna().any():
        raise TableError("images: column 'image' has gaps")
    repeated = keys[keys.duplicated()].unique().tolist()
    if repeated:
        raise TableError(
            f"images: column 'image' must be unique, repeated: "
            f"{repeated[:_MAX_LISTED]}"
        )
    if fov_area:
        require(images, ["fov_area"], "images")
        area = numbers(images, "fov_area", "images")
        given = ~np.isnan(area)
        if not (np.isfinite(area[given]).all() and (area[given] > 0).all()):
            raise TableError(
                "images: column 'fov_area' must be > 0 where given "
                "(NaN when unknown)"
            )


def group_images(
    images: pd.DataFrame, by: str | Iterable[str] | None
) -> Grouping:
    """Form the groups of images for ``by=`` (call ``check_images`` first).

    ``None`` is one group of all images; ``"image"`` one group per
    image; any images-table column, or a list of them, one group per
    existing combination, sorted. An empty cell forms its own group,
    labelled NaN, sorted last (pandas would drop it).

    Raises:
        TableError: If a ``by`` column is not in the images table.
        ParamsError: If ``by`` is not column names.
    """
    columns = names(by, "by")
    missing = [c for c in columns if c not in images.columns]
    if missing:
        raise TableError(f"by: no column {missing} in the images table")
    if not columns:
        one = np.zeros(len(images), dtype=np.intp)
        return Grouping(columns, one, pd.DataFrame(index=range(1)))
    keyed = images[columns].reset_index(drop=True)
    codes = np.asarray(
        keyed.groupby(columns, dropna=False, sort=True).ngroup(),
        dtype=np.intp,
    )
    # One row per group, in group order: its first image's values.
    first = np.unique(codes, return_index=True)[1]
    return Grouping(columns, codes, keyed.iloc[first].reset_index(drop=True))


def image_positions(
    images: pd.DataFrame, table: pd.DataFrame, name: str
) -> NDArray[np.intp]:
    """Row of the images table for each row of ``table``.

    Raises:
        TableError: If ``table`` has no ``image`` column or names an
            image that is not in the images table.
    """
    require(table, ["image"], name)
    positions = np.asarray(
        pd.Index(images["image"]).get_indexer(table["image"]), dtype=np.intp
    )
    unknown = table["image"][positions < 0].unique().tolist()
    if unknown:
        raise TableError(
            f"{name}: images not in the images table: {unknown[:_MAX_LISTED]}"
        )
    return positions


def require(table: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise TableError(f"{name}: missing columns: {missing}")


def numbers(
    table: pd.DataFrame, column: str, name: str
) -> NDArray[np.float64]:
    """A numeric column as float64; NaN stays NaN."""
    try:
        values = pd.to_numeric(table[column], errors="raise")
    except (ValueError, TypeError) as exc:
        raise TableError(
            f"{name}: column {column!r} must be numeric: {exc}"
        ) from exc
    return values.to_numpy(dtype=np.float64, na_value=np.nan)
