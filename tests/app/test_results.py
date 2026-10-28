"""``Results``: the statistics methods over the analysed tables."""

import os
import warnings
from collections.abc import Callable

import pandas as pd
import pytest

from agglpy.app import Collection
from agglpy.app._results import Results
from agglpy.core import distributions, metrics
from agglpy.core.distributions import size_classes
from agglpy.errors import MissingImagesWarning, ParamsError

EDGES = size_classes(1e-9, 1e-6, step=10**0.1, scale="log")


@pytest.fixture
def res(analysis_col: Collection) -> Results:
    return analysis_col.analyze(progress=False)


def _per_image(r: Results) -> pd.DataFrame:
    # D7-019 has no agglomerates: no row
    return r.agglomerates.groupby("image", as_index=False).D.mean()


Pair = Callable[[Results], tuple[pd.DataFrame, pd.DataFrame]]

CALLS: dict[str, Pair] = {
    "summary": lambda r: (
        metrics.summary(r.images, r.particles, r.agglomerates, "series"),
        r.summary("series"),
    ),
    "summary_across_images": lambda r: (
        metrics.summary_across_images(
            r.images,
            r.particles,
            r.agglomerates,
            "series",
            metrics="counts",
            confidence=0.9,
        ),
        r.summary_across_images("series", metrics="counts", confidence=0.9),
    ),
    "values_across_images": lambda r: (
        metrics.values_across_images(
            r.images, _per_image(r), "series", values="D"
        ),
        r.values_across_images(_per_image(r), "series", values="D"),
    ),
    "distribution": lambda r: (
        distributions.distribution(
            r.images, r.agglomerates, "D", EDGES, "series"
        ),
        r.distribution("agglomerates", "D", EDGES, "series"),
    ),
    "distribution_across_images": lambda r: (
        distributions.distribution_across_images(
            r.images, r.particles, "D", EDGES, closed="left"
        ),
        r.distribution_across_images("particles", "D", EDGES, closed="left"),
    ),
}


@pytest.mark.parametrize("method", list(CALLS))
def test_methods_equal_their_core_call(res: Results, method: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected, got = CALLS[method](res)
    pd.testing.assert_frame_equal(got, expected)


@pytest.mark.parametrize(
    "method", ["distribution", "distribution_across_images"]
)
def test_bad_table_name(res: Results, method: str) -> None:
    with pytest.raises(
        ParamsError,
        match="table must be 'particles' or 'agglomerates', got 'images'",
    ):
        getattr(res, method)("images", "D", EDGES)


def test_a_user_column_is_seen(res: Results) -> None:
    res.particles["d_um"] = res.particles.D * 1e6
    got = res.distribution("particles", "d_um", [0.01, 0.1, 1.0])
    assert got["amount"].sum() == len(res.particles)


def test_method_warning_points_at_the_caller(res: Results) -> None:
    with pytest.warns(MissingImagesWarning) as ws:
        res.values_across_images(_per_image(res), values="D")
    (w,) = ws
    assert os.path.normcase(w.filename) == os.path.normcase(__file__)
    assert "D7-019" in str(w.message)
