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
from agglpy.errors import MissingImagesWarning, ParamsError, TableError

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


# ---------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------


def test_subset_by_images_and_agglomerates_chained(res: Results) -> None:
    s = res.subset(images=res.images.series == "a", label="series a")
    assert s.images.image.tolist() == ["D7-017", "D7-018"]
    assert set(s.particles.image) == {"D7-017", "D7-018"}
    assert sorted(s._origin.images) == ["D7-017", "D7-018"]
    assert sorted(s._origin.files) == ["D7-017", "D7-018"]
    t = s.subset(agglomerates=s.agglomerates.member_count >= 2, label="pairs")
    assert t.agglomerates.member_count.min() >= 2
    assert len(t.images) == 2  # the images table stays whole
    # only the members of the kept agglomerates
    kept = set(zip(t.agglomerates.image, t.agglomerates.agglomerate_id))
    assert set(zip(t.particles.image, t.particles.agglomerate_id)) == kept
    for table in (t.images, t.particles, t.agglomerates):
        assert table.index.equals(pd.RangeIndex(len(table)))
    assert t._origin.steps == (
        {
            "label": "series a",
            "images": {"kept": 2, "of": 3},
            "agglomerates": {"kept": 4, "of": 4},
            "particles": {"kept": 5, "of": 5},
        },
        {
            "label": "pairs",
            "images": {"kept": 2, "of": 2},
            "agglomerates": {"kept": 1, "of": 4},
            "particles": {"kept": 2, "of": 5},
        },
    )
    assert t._origin.excluded == res._origin.excluded
    assert repr(t).endswith("; 2 subset steps)")
    # repeated names once, in the order of the results
    named = res.subset(images=["D7-019", "D7-017", "D7-017"])
    assert named.images.image.tolist() == ["D7-017", "D7-019"]
    assert named._origin.steps[0]["label"] is None
    assert res.subset(images="D7-018").images.image.tolist() == ["D7-018"]
    assert len(res.images) == 3 and res._origin.steps == ()


def test_subset_both_is_one_step_images_first(res: Results) -> None:
    s = res.subset(
        images=["D7-018", "D7-019"],
        agglomerates=res.agglomerates.member_count >= 2,
    )
    assert s.images.image.tolist() == ["D7-018", "D7-019"]
    assert len(s.agglomerates) == 1
    assert s._origin.steps == (
        {
            "label": None,
            "images": {"kept": 2, "of": 3},
            "agglomerates": {"kept": 1, "of": 4},
            "particles": {"kept": 2, "of": 5},
        },
    )


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({}, ParamsError, "images= or agglomerates="),
        ({"images": "X"}, ParamsError, "no image 'X' in the collection"),
        (
            {"images": "D7-020"},
            ParamsError,
            r"'D7-020' is not in these results \(include false, or left "
            r"out by an earlier subset\)",
        ),
        ({"images": []}, ParamsError, "no image left"),
        ({"images": pd.Series([True])}, TableError, "res.images' index"),
        (
            {"agglomerates": pd.Series([True])},
            TableError,
            "agglomerate table's index",
        ),
    ],
)
def test_subset_refuses(
    res: Results, kwargs: dict[str, object], error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        res.subset(**kwargs)  # type: ignore[arg-type]


def test_subset_images_mask_must_be_true_false(res: Results) -> None:
    with pytest.raises(TableError, match="True / False Series"):
        res.subset(images=res.images.width)
    with pytest.raises(ParamsError, match="no image left"):
        res.subset(images=res.images.series == "c")


def test_added_column_travels(res: Results) -> None:
    res.agglomerates["dense"] = res.agglomerates.member_count > 1
    s = res.subset(images=["D7-018"])
    assert "dense" in s.agglomerates.columns
