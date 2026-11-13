"""``Results``: statistics, subset and export."""

import os
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agglpy import _yaml
from agglpy.app import Collection
from agglpy.app._results import Results
from agglpy.core import distributions, metrics
from agglpy.core.distributions import size_classes
from agglpy.core.structure import AGGLOMERATE_COLUMNS, PARTICLE_COLUMNS
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


def test_bad_table_name(res: Results) -> None:
    for method in (res.distribution, res.distribution_across_images):
        with pytest.raises(
            ParamsError,
            match="table must be 'particles' or 'agglomerates', got 'images'",
        ):
            method("images", "D", EDGES)


def test_a_user_column_travels(res: Results, tmp_path: Path) -> None:
    # the methods see it, subset keeps it, export writes it without unit
    res.particles["d_um"] = res.particles.D * 1e6
    got = res.distribution("particles", "d_um", [0.01, 0.1, 1.0])
    assert got["amount"].sum() == len(res.particles)
    res.agglomerates["dense"] = res.agglomerates.member_count > 1
    s = res.subset(images=["D7-018"])
    assert "dense" in s.agglomerates.columns
    out = s.export(tmp_path / "out")
    header = (out / "csv" / "agglomerates.csv").read_text("utf-8")
    assert "dense" in header.splitlines()[0].split(",")
    units = pd.read_csv(out / "csv" / "units.csv")
    assert "dense" not in set(units.column)


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
    # masks made from res: not True / False, or keeping no image
    with pytest.raises(TableError, match="True / False Series"):
        res.subset(images=res.images.width)
    with pytest.raises(ParamsError, match="no image left"):
        res.subset(images=res.images.series == "c")


# ---------------------------------------------------------------------
# contacts
# ---------------------------------------------------------------------


def test_contacts_table(res: Results) -> None:
    c = res.contacts
    assert list(c.columns) == [
        "image",
        "agglomerate_id",
        "id_a",
        "id_b",
        "inside",
    ]
    # D7-017: two separate discs; D7-018: a touching pair and a single
    # circle; D7-019: blank
    assert c.image.tolist() == ["D7-018"]
    assert not c.inside.any()
    assert c.index.equals(pd.RangeIndex(1))


def test_contacts_dtypes_with_a_blank_image(res: Results) -> None:
    assert res.contacts.dtypes.drop("image").tolist() == [np.int64] * 3 + [
        bool
    ]
    blank = res.subset(images="D7-019")
    assert len(blank.contacts) == 0
    assert blank.contacts.dtypes.drop("image").tolist() == [np.int64] * 3 + [
        bool
    ]


def test_contacts_match_contact_count_through_subset(res: Results) -> None:
    for r in (
        res,
        res.subset(agglomerates=res.agglomerates.member_count >= 2),
        res.subset(agglomerates=res.agglomerates.member_count < 2),
        res.subset(images="D7-018"),
        res.subset(images="D7-017"),
    ):
        key = ["image", "agglomerate_id"]
        rows = r.contacts.groupby(key).size()
        counts = r.agglomerates.set_index(key).contact_count
        assert rows.reindex(counts.index, fill_value=0).tolist() == (
            counts.tolist()
        )
        assert r.contacts.index.equals(pd.RangeIndex(len(r.contacts)))
    assert len(res.subset(images="D7-017").contacts) == 0
    assert len(res.subset(images="D7-018").contacts) == 1


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
        "contacts",
        "summary",
        "units",
        "info",
    ]
    xl.close()
    contacts = pd.read_csv(out / "csv" / "contacts.csv")
    assert contacts.columns.tolist() == [
        "image",
        "agglomerate_id",
        "id_a",
        "id_b",
        "inside",
    ]
    assert contacts.image.tolist() == ["D7-018"]
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
    assert not any(table in ("summary", "contacts") for table, _ in listed)
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


def test_export_extra_table_named_like_a_result(
    res: Results, tmp_path: Path
) -> None:
    for name in ("images", "agglomerates", "particles", "contacts"):
        with pytest.raises(ParamsError, match=f"{name!r}"):
            res.export(tmp_path / "out", tables={name: res.summary()})
    assert not (tmp_path / "out").exists()


def test_export_folders_default_relative_and_guard(
    analysis_col: Collection,
    res: Results,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exports = (analysis_col.folder / "exports").resolve()
    # no folder: exports/<kind> in the collection
    assert res.export() == exports / "results"
    assert (exports / "results" / "results.xlsx").is_file()
    d = res.draw("D7-017", color_by="D")
    assert d.export() == [exports / "drawings" / "D7-017.png"]
    # a second export to the same default needs overwrite=True
    with pytest.raises(OutputExistsError, match="overwrite=True"):
        d.export()
    with pytest.raises(OutputExistsError, match="overwrite=True"):
        res.export()
    assert res.export(overwrite=True) == exports / "results"
    # a relative folder is taken from the current working directory
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert res.export("rel") == (elsewhere / "rel").resolve()
    assert d.export("rel") == [(elsewhere / "rel" / "D7-017.png").resolve()]
    assert (elsewhere / "rel" / "results.xlsx").is_file()
    # nothing goes into the work folder, written or not
    work = analysis_col.folder / "agglpy"
    for bad in (work, work / "x", work / "x" / ".." / "y"):
        with pytest.raises(
            ParamsError, match="inside the collection's work folder"
        ):
            res.export(bad)
        with pytest.raises(ParamsError, match="work folder; export elsewhere"):
            d.export(bad)
        assert not bad.joinpath("results.xlsx").exists()
        assert not bad.joinpath("D7-017.png").exists()
    assert not (work / "x").exists() and not (work / "y").exists()


# ---------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------


def test_with_structure_columns(res: Results) -> None:
    s = res.with_structure()
    assert s is not res and "segment_type" not in res.particles
    assert list(s.particles.columns) == [
        *res.particles.columns,
        *PARTICLE_COLUMNS,
    ]
    assert list(s.agglomerates.columns) == [
        *res.agglomerates.columns,
        *AGGLOMERATE_COLUMNS,
    ]
    assert set(s.particles.segment_type) == {"no core"}
    pair = s.agglomerates[s.agglomerates.member_count == 2].iloc[0]
    members = s.particles[
        (s.particles.image == pair.image)
        & (s.particles.agglomerate_id == pair.agglomerate_id)
    ]
    assert pair.longest_chain == 1 and pair.branch_count == 1
    # in m, like the diameters
    assert pair.longest_chain_length == pytest.approx(members.D.sum())
    # the rows stay where they were
    assert s.particles[["image", "id"]].equals(res.particles[["image", "id"]])
    assert s.images is res.images and s.contacts is res.contacts


def test_structure_dtypes_with_a_blank_image(res: Results) -> None:
    for s in (
        res.with_structure(),
        res.subset(images="D7-019").with_structure(),
    ):
        dtypes = s.particles.dtypes
        assert dtypes["segment_type"] == np.dtype(object)
        assert dtypes["segment_id"] == np.int64
        assert dtypes["branch_depth"] == np.int64
        assert dtypes["tip"] == np.dtype(bool)
        assert s.agglomerates.dtypes["longest_chain"] == np.int64
    assert len(res.subset(images="D7-019").with_structure().particles) == 0


def _segments(r: Results) -> set[frozenset[tuple[str, int]]]:
    """The particles of each segment, by (image, id)."""
    keys = r.particles.groupby(["image", "segment_id"]).id
    return {frozenset((image, i) for i in ids) for (image, _), ids in keys}


def test_with_structure_again_and_around_subset(res: Results) -> None:
    s = res.with_structure()
    again = s.with_structure()
    assert list(again.particles.columns) == list(s.particles.columns)
    assert list(again.agglomerates.columns) == list(s.agglomerates.columns)

    mask = res.agglomerates.member_count >= 2
    first = res.subset(agglomerates=mask).with_structure()
    after = s.subset(agglomerates=mask)
    for table in ("particles", "agglomerates"):
        pd.testing.assert_frame_equal(
            getattr(first, table).drop(columns="segment_id", errors="ignore"),
            getattr(after, table).drop(columns="segment_id", errors="ignore"),
        )
    assert _segments(first) == _segments(after)


def test_with_structure_refuses_min_core_members(res: Results) -> None:
    with pytest.raises(ParamsError, match="min_core_members"):
        res.with_structure(min_core_members=2)


def test_structure_export_units_and_record(
    res: Results, tmp_path: Path
) -> None:
    s = res.with_structure(min_core_members=8)
    out = s.export(tmp_path / "out")
    rows = pd.read_csv(out / "csv" / "units.csv").values.tolist()
    assert ["particles", "branch_length", "m"] in rows
    assert ["particles", "dangling_end_length", "m"] in rows
    assert ["agglomerates", "longest_chain_length", "m"] in rows
    assert ["agglomerates", "branch_length_max", "m"] in rows
    listed = {(table, column) for table, column, _ in rows}
    assert ("agglomerates", "longest_chain") not in listed
    assert ("particles", "branch_depth") not in listed
    structure = {
        "min_core_members": 8,
        "base": "largest member, ties: lower id",
    }
    rec = _yaml.load(out / "record.yml")
    assert rec["structure"] == structure
    assert list(rec)[-1] == "structure"
    plain = _yaml.load(res.export(tmp_path / "plain") / "record.yml")
    assert "structure" not in plain
    # after subset: the key stays, the step is listed
    rec = _yaml.load(
        s.subset(images="D7-018", label="one").export(tmp_path / "sub")
        / "record.yml"
    )
    assert rec["structure"] == structure
    assert [step["label"] for step in rec["subset"]] == ["one"]


@pytest.mark.parametrize("call", ["summary", "summary_across_images"])
def test_structure_summary_needs_the_columns(res: Results, call: str) -> None:
    with pytest.raises(TableError, match=r"with_structure\(\)"):
        getattr(res, call)(metrics="structure")
    with pytest.raises(TableError, match=r"with_structure\(\)"):
        getattr(res, call)(metrics=["counts", "structure"])
    assert "N_cores" not in res.summary().columns
    assert "N_cores" in res.with_structure().summary().columns


def test_draw_by_segment_type(res: Results) -> None:
    fig = res.with_structure().draw(what="particles", color_by="segment_type")[
        "D7-018"
    ]
    texts = [
        t.get_text()
        for ax in fig.axes
        if ax.get_legend()
        for t in ax.get_legend().get_texts()
    ]
    assert "no core" in texts
