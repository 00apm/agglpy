"""Collection.status: where each image's particles come from."""

import os
import warnings
from pathlib import Path

import pandas as pd
import pytest

from agglpy.app import Collection
from agglpy.errors import (
    DuplicateParticlesWarning,
    LeftoverFoldersWarning,
    NoParticlesWarning,
    ParamsError,
    UnreadableRecordWarning,
)

from support.collection import HCT, work, write_disc_image
from support.tiffs import write_zeiss

COLUMNS = [
    "image",
    "include",
    "file",
    "pixel_size_nm",
    "pixel_size_source",
    "particle_set",
    "particle_count",
    "outside_image",
    "detection",
    "correction",
    "kept",
    "warnings",
]
NAMES = ["D7-017", "D7-018", "D7-019"]
OTHER_DISCS = [(100.2, 80.3, 14.2)]
NOT_DONE = pytest.mark.filterwarnings(
    "ignore::agglpy.errors.IncompleteRunWarning"
)


def csv(col: Collection, name: str = "D7-017") -> Path:
    return work(col, name) / "particles_hct.csv"


def record_path(col: Collection, name: str = "D7-017") -> Path:
    return work(col, name) / "particles_hct_record.yml"


def image(col: Collection, name: str = "D7-017") -> Path:
    return col.folder.parent / "images" / f"{name}.tif"


def row(col: Collection, name: str = "D7-017") -> pd.Series:
    table = col.status()
    return table.set_index("image").loc[name]


def test_fresh_collection(particles_col: Collection) -> None:
    table = particles_col.status()
    assert table.columns.tolist() == COLUMNS
    assert table["image"].tolist() == NAMES
    assert table["include"].tolist() == [True] * 3
    assert table["file"].tolist() == ["ok"] * 3
    assert table["pixel_size_nm"].isna().all()
    assert table["pixel_size_source"].tolist() == ["none"] * 3
    assert table["particle_set"].tolist() == ["none"] * 3
    assert table["particle_count"].tolist() == [0] * 3
    assert table["outside_image"].tolist() == [0] * 3
    assert table["detection"].tolist() == ["not detected"] * 3
    assert table["correction"].tolist() == [""] * 3
    assert table["kept"].tolist() == [""] * 3
    assert table["warnings"].tolist() == [""] * 3
    assert table.dtypes.astype(str).to_dict() == {
        "image": "object",
        "include": "bool",
        "file": "object",
        "pixel_size_nm": "Float64",
        "pixel_size_source": "object",
        "particle_set": "object",
        "particle_count": "Int64",
        "outside_image": "Int64",
        "detection": "object",
        "correction": "object",
        "kept": "object",
        "warnings": "object",
    }


def test_hct_unset_is_not_set(tmp_path: Path, images: Path) -> None:
    col = Collection.create(tmp_path / "col", images=images)
    assert col.status()["detection"].tolist() == ["not set"] * 3


def test_after_detect_up_to_date(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    table = particles_col.status()
    assert table["particle_set"].tolist() == ["hct"] * 3
    assert table["particle_count"].tolist() == [2] * 3
    assert table["outside_image"].tolist() == [0] * 3
    assert table["detection"].tolist() == ["up to date"] * 3


def test_params_changed(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    particles_col.set_params(
        "D7-018", hct={"ranges": [{"d_min": 12, "d_max": 40}]}
    )
    assert particles_col.status()["detection"].tolist() == [
        "up to date",
        "params changed",
        "up to date",
    ]


def test_image_changed(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    write_disc_image(image(particles_col, "D7-019"), OTHER_DISCS)
    assert row(particles_col, "D7-019")["detection"] == "image changed"


def test_image_changed_wins_over_params_changed(
    particles_col: Collection,
) -> None:
    particles_col.detect(progress=False)
    write_disc_image(image(particles_col), OTHER_DISCS)
    particles_col.set_params(
        None, hct={"ranges": [{"d_min": 12, "d_max": 40}]}
    )
    assert row(particles_col)["detection"] == "image changed"


def test_edited_by_hand(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    with csv(particles_col).open("a", encoding="utf-8") as fh:
        fh.write("3,10.5,10.5,5.0\n")
    r = row(particles_col)
    assert r["detection"] == "edited by hand"
    assert r["particle_count"] == 3


def test_no_record(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    record_path(particles_col).unlink()
    assert row(particles_col)["detection"] == "no record"


def test_unreadable_csv(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    csv(particles_col).write_text("id;x;y;r\n1;10;10;5\n", encoding="utf-8")
    with pytest.warns(
        UserWarning, match=r"particles_hct\.csv in D7-017/ does not read"
    ):
        r = row(particles_col)
    assert r["detection"] == "unreadable"
    assert r["particle_set"] == "hct"
    assert pd.isna(r["particle_count"])
    assert pd.isna(r["outside_image"])


def test_unreadable_record_warns_and_reads_as_no_record(
    particles_col: Collection,
) -> None:
    particles_col.detect(progress=False)
    record_path(particles_col).write_text("format: [", encoding="utf-8")
    with pytest.warns(UnreadableRecordWarning) as ws:
        r = row(particles_col)
    assert r["detection"] == "no record"
    (w,) = [w for w in ws if w.category is UnreadableRecordWarning]
    assert os.path.normcase(w.filename) == os.path.normcase(__file__)


def test_hct_emptied_after_detection_is_params_changed(
    particles_col: Collection,
) -> None:
    particles_col.detect(progress=False)
    particles_col.set_params(None, hct={"ranges": []})
    table = particles_col.status()
    assert table["detection"].tolist() == ["params changed"] * 3
    assert table["particle_set"].tolist() == ["hct"] * 3


@NOT_DONE
def test_correction_labels(
    particles_col: Collection, fiji_csv: Path, tmp_path: Path
) -> None:
    particles_col.detect(progress=False)
    particles_col.import_particles("D7-017", fiji_csv)
    r = row(particles_col)
    assert r["particle_set"] == "corrected"
    assert r["correction"] == "current detection"
    assert r["kept"] == "2 of 2 detected"
    assert r["detection"] == "up to date"
    # a new detection: the correction started from the older one
    write_disc_image(image(particles_col), OTHER_DISCS)
    particles_col.detect("D7-017", force=True, progress=False)
    r = row(particles_col)
    assert r["correction"] == "older detection"
    assert r["particle_set"] == "corrected"
    # a correction drawn from scratch
    scratch = tmp_path / "scratch.csv"
    scratch.write_text("id,x,y,r\n1,10.5,10.5,5.0\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="keeps only 0 of 1"):
        particles_col.import_particles("D7-017", scratch)
    r = row(particles_col)
    assert r["correction"] == "not from detection"
    assert r["kept"] == "0 of 1 detected"
    corrected = work(particles_col, "D7-017") / "particles_corrected.csv"
    with corrected.open("a", encoding="utf-8") as fh:
        fh.write("2,60.5,60.5,5.0\n")
    r = row(particles_col)
    assert r["correction"] == "edited by hand"
    assert r["particle_set"] == "corrected"
    assert r["particle_count"] == 2


def test_correction_without_detection(
    particles_col: Collection, fiji_csv: Path
) -> None:
    particles_col.import_particles("D7-017", fiji_csv)
    r = row(particles_col)
    assert r["correction"] == "not from detection"
    assert r["kept"] == ""
    assert r["detection"] == "not detected"


def test_missing_image(tmp_path: Path) -> None:
    # Zeiss: 1 nm at 1024 px is 16 nm at 64 px
    (tmp_path / "images").mkdir()
    write_zeiss(tmp_path / "images" / "a.tif", 64, 48, raw=1e-9)
    col = Collection.create(tmp_path / "col", images=tmp_path / "images")
    col.set_params(None, hct=HCT)
    with pytest.warns(NoParticlesWarning):
        col.detect(progress=False)
    (tmp_path / "images" / "a.tif").unlink()
    r = row(col, "a")
    assert r["file"] == "missing"
    assert r["detection"] == "up to date"
    assert r["pixel_size_nm"] == pytest.approx(16.0)
    assert r["pixel_size_source"] == "image file"
    assert r["outside_image"] == 0


def test_unreadable_image(particles_col: Collection) -> None:
    image(particles_col).write_bytes(b"junk")
    assert row(particles_col)["file"] == "unreadable"


def test_pixel_size_precedence(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    write_zeiss(folder / "a.tif", 64, 48, raw=1e-9)
    write_disc_image(folder / "b.tif", OTHER_DISCS)
    col = Collection.create(tmp_path / "col", images=folder)
    table = col.status()
    assert table["pixel_size_nm"].tolist()[0] == pytest.approx(16.0)
    assert pd.isna(table["pixel_size_nm"].tolist()[1])
    assert table["pixel_size_source"].tolist() == ["image file", "none"]
    sheet = col.read_sheet()
    sheet["pixel_size_nm"] = [2.5, 3.0]
    col.write_sheet(sheet)
    table = col.status()
    assert table["pixel_size_nm"].tolist() == [2.5, 3.0]
    assert table["pixel_size_source"].tolist() == ["images.xlsx"] * 2
    csv_col = Collection.create(
        tmp_path / "csv", images=folder, sheet_format="csv"
    )
    sheet = csv_col.read_sheet()
    sheet["pixel_size_nm"] = ["2.5", None]
    csv_col.write_sheet(sheet)
    table = csv_col.status()
    assert table["pixel_size_source"].tolist() == ["images.csv", "none"]


def test_outside_image_counts_circles_beyond_the_image(
    particles_col: Collection,
) -> None:
    particles_col.detect(progress=False)
    csv(particles_col).write_text(
        "id,x,y,r\n1,-50.0,50.0,5.0\n2,60.5,50.5,15.0\n", encoding="utf-8"
    )
    assert row(particles_col)["outside_image"] == 1


def test_warnings_column_from_the_set_in_use(
    particles_col: Collection,
) -> None:
    particles_col.set_params(
        None,
        hct={
            "ranges": [{"d_min": 12, "d_max": 44}, {"d_min": 20, "d_max": 40}]
        },
    )
    with pytest.warns(DuplicateParticlesWarning):
        particles_col.detect(progress=False)
    assert (
        particles_col.status()["warnings"].tolist()
        == ["DuplicateParticlesWarning"] * 3
    )


def test_leftover_folder_warns_with_hint(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    (particles_col.folder / "agglpy" / "old").mkdir()
    (particles_col.folder / "agglpy" / "notes.txt").write_text("x")
    sheet = particles_col.read_sheet()
    sheet.loc[0, "image"] = "D7-017a"
    particles_col.write_sheet(sheet)
    with pytest.warns(LeftoverFoldersWarning) as ws:
        table = particles_col.status()
    (w,) = [w for w in ws if w.category is LeftoverFoldersWarning]
    assert str(w.message) == (
        "agglpy/ has folders for images not in images.xlsx: D7-017 (same "
        "image file as D7-017a: rename the folder to D7-017a to keep its "
        "particles), old"
    )
    assert os.path.normcase(w.filename) == os.path.normcase(__file__)
    assert csv(particles_col).is_file()
    assert table.set_index("image").loc["D7-017a", "detection"] == (
        "not detected"
    )


def test_record_without_csv_is_ignored(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    csv(particles_col).unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = row(particles_col)
    assert r["detection"] == "not detected"
    assert r["particle_set"] == "none"


@NOT_DONE
@pytest.mark.filterwarnings("ignore:correction keeps only")
def test_status_writes_nothing(
    particles_col: Collection, fiji_csv: Path
) -> None:
    particles_col.detect(progress=False)
    particles_col.import_particles("D7-018", fiji_csv)
    image(particles_col, "D7-019").unlink()

    def snapshot() -> dict[str, tuple[bytes, int]]:
        return {
            str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(particles_col.folder.parent.rglob("*"))
            if p.is_file()
        }

    before = snapshot()
    particles_col.status()
    assert snapshot() == before


def test_broken_yml_raises(particles_col: Collection) -> None:
    path = particles_col.folder / "agglpy.yml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("d_max: 44", "d_max: many"), "utf-8")
    with pytest.raises(ParamsError, match="d_max"):
        particles_col.status()


def test_empty_sheet(tmp_path: Path) -> None:
    table = Collection.create(tmp_path / "col").status()
    assert table.columns.tolist() == COLUMNS
    assert len(table) == 0


# ---------------------------------------------------------------------
# Review focus
# ---------------------------------------------------------------------


def test_new_mtime_same_bytes_is_up_to_date(
    particles_col: Collection,
) -> None:
    particles_col.detect(progress=False)
    for name in NAMES:
        later = image(particles_col, name).stat().st_mtime_ns + 10**9
        os.utime(image(particles_col, name), ns=(later, later))
    assert particles_col.status()["detection"].tolist() == ["up to date"] * 3
    rep = particles_col.detect(progress=False)
    assert rep["message"].tolist() == ["up to date"] * 3


def test_folder_differing_in_case_is_not_leftover(
    particles_col: Collection,
) -> None:
    particles_col.detect(progress=False)
    agglpy = particles_col.folder / "agglpy"
    # two steps: Windows refuses a rename that changes only the case
    (agglpy / "D7-017").rename(agglpy / "tmp")
    (agglpy / "tmp").rename(agglpy / "d7-017")
    with warnings.catch_warnings():
        warnings.simplefilter("error", LeftoverFoldersWarning)
        r = row(particles_col)
    if (agglpy / "D7-017").exists():  # a case-insensitive file system
        assert r["particle_set"] == "hct"
        assert r["detection"] == "up to date"


def test_folder_differing_in_unicode_form_is_leftover(tmp_path: Path) -> None:
    # NTFS keeps "ó" as one character and as "o" + accent apart, as a
    # case-sensitive file system keeps "d7-017" and "D7-017" apart
    write_disc_image(tmp_path / "images" / "pr\u00f3bka.tif", OTHER_DISCS)
    col = Collection.create(tmp_path / "col", images=tmp_path / "images")
    other = col.folder / "agglpy" / "pro\u0301bka"
    other.mkdir()
    if (col.folder / "agglpy" / "pr\u00f3bka").exists():
        pytest.skip("this file system joins the two forms")
    with pytest.warns(LeftoverFoldersWarning, match="pro\u0301bka"):
        col.status()
