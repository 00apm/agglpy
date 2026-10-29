"""``Results``: statistics, subset and export."""

import os
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from agglpy import _yaml
from agglpy.app import Collection
from agglpy.app._results import Results
from agglpy.core import distributions, metrics
from agglpy.core.distributions import size_classes
from agglpy.errors import (
    MissingImagesWarning,
    OutputExistsError,
    ParamsError,
    TableError,
)

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


# ---------------------------------------------------------------------
# export
# ---------------------------------------------------------------------


def test_export_writes_tables_units_and_record(
    res: Results, tmp_path: Path
) -> None:
    out = res.subset(images=["D7-017", "D7-018"], label="two").export(
        tmp_path / "out", tables={"summary": res.summary("series")}
    )
    assert out == tmp_path / "out"
    xl = pd.ExcelFile(out / "results.xlsx")
    assert xl.sheet_names == [
        "images",
        "agglomerates",
        "particles",
        "summary",
        "units",
        "info",
    ]
    xl.close()
    rows = pd.read_csv(out / "csv" / "units.csv").values.tolist()
    for row in (
        ["images", "pixel_size", "m"],
        ["images", "fov_area", "m²"],
        ["images", "width", "px"],
        ["images", "height", "px"],
        ["particles", "x", "m"],
        ["particles", "r", "m"],
        ["particles", "volume", "m³"],
        ["agglomerates", "area", "m²"],
        ["agglomerates", "D_feret_max", "m"],
    ):
        assert row in rows
    listed = {(table, column) for table, column, _ in rows}
    assert ("agglomerates", "member_count") not in listed  # no dimension
    assert not any(table == "summary" for table, _ in listed)
    rec = _yaml.load(out / "record.yml")
    analysis = rec["analysis"]
    assert analysis["unit"] == "m" and analysis["sheet"] == "images.xlsx"
    assert analysis["collection"] == res._origin.collection.as_posix()
    assert analysis["analyzed"] == res._origin.analyzed
    assert list(rec["images"]) == ["D7-017", "D7-018"]
    made_by = rec["images"]["D7-017"]["made_by"]
    assert made_by["kind"] == "detection" and "hct" in made_by["params"]
    d18 = rec["images"]["D7-018"]
    assert d18["made_by"]["source"]["format"] == "agglpy"
    assert d18["particles"]["state"] == "matching"
    assert d18["particle_set"] == "corrected"
    assert rec["excluded"] == [{"image": "D7-020", "note": "out of focus"}]
    assert [s["label"] for s in rec["subset"]] == ["two"]
    with pytest.raises(OutputExistsError):
        res.export(out)
    res.export(out, overwrite=True)
    assert _yaml.load(out / "record.yml")["subset"] == []


@pytest.mark.parametrize("name", ["images", "agglomerates", "particles"])
def test_export_extra_table_named_like_a_result(
    res: Results, tmp_path: Path, name: str
) -> None:
    with pytest.raises(ParamsError, match=f"{name!r}"):
        res.export(tmp_path / "out", tables={name: res.summary()})
    assert not (tmp_path / "out").exists()


def test_added_column_is_exported_without_unit(
    res: Results, tmp_path: Path
) -> None:
    res.agglomerates["dense"] = res.agglomerates.member_count > 1
    out = res.subset(images=["D7-018"]).export(tmp_path / "out")
    header = (out / "csv" / "agglomerates.csv").read_text("utf-8")
    assert "dense" in header.splitlines()[0].split(",")
    units = pd.read_csv(out / "csv" / "units.csv")
    assert "dense" not in set(units.column)
