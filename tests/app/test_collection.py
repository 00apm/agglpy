"""Collection: create, open, add images, params, the sheet."""

import os
import shutil
from pathlib import Path

import openpyxl
import pytest

import agglpy
from agglpy.app import Collection
from agglpy.app._config import NEUTRAL, template
from agglpy.app._sources import resolve_stored
from agglpy.errors import (
    OutputExistsError,
    OutputLockedError,
    ParamsError,
    SheetError,
)
from agglpy.params import PreprocessParams

H = ["image", "file", "include", "pixel_size_nm", "note"]


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    (raw / "sub").mkdir(parents=True)
    for name in ("D7-017.tif", "D7-019.TIF", "D7-021.png", "notes.txt"):
        (raw / name).write_bytes(name.encode())
    (raw / "sub" / "D7-017_HCT.jpg").touch()
    return raw


@pytest.fixture
def col(tmp_path: Path, raw: Path) -> Collection:
    return Collection.create(tmp_path / "col", images=raw)


def yml(col: Collection) -> Path:
    return col.folder / "agglpy.yml"


def test_create_writes_sheet_yml_and_work_folder(
    tmp_path: Path, col: Collection
) -> None:
    folder = tmp_path / "col"
    assert sorted(p.name for p in folder.iterdir()) == [
        "agglpy",
        "agglpy.yml",
        "images.xlsx",
    ]
    assert list((folder / "agglpy").iterdir()) == []
    t = col.read_sheet()
    assert list(t.columns) == H
    assert t["image"].tolist() == ["D7-017", "D7-019", "D7-021"]
    assert t["file"].tolist()[0] == "../raw/D7-017.tif"
    assert t["include"].isna().all() and t["pixel_size_nm"].isna().all()
    assert yml(col).read_text(encoding="utf-8") == template()


def test_create_then_open_gives_neutral_params(col: Collection) -> None:
    opened = Collection.open(col.folder)
    assert opened.params() == NEUTRAL
    assert opened.params("D7-019") == NEUTRAL


def test_create_csv(tmp_path: Path, raw: Path) -> None:
    col = Collection.create(tmp_path / "col", images=raw, sheet_format="csv")
    assert (col.folder / "images.csv").exists()
    assert not (col.folder / "images.xlsx").exists()
    assert col.sheet_path.name == "images.csv"
    assert col.read_sheet()["image"].tolist() == ["D7-017", "D7-019", "D7-021"]


def test_create_bad_sheet_format(tmp_path: Path, raw: Path) -> None:
    with pytest.raises(ParamsError, match="sheet_format"):
        Collection.create(tmp_path / "col", images=raw, sheet_format="ods")  # type: ignore[arg-type]
    assert not (tmp_path / "col").exists()


@pytest.mark.parametrize("name", ["images.xlsx", "images.csv", "agglpy.yml"])
def test_create_refuses_existing_files(
    tmp_path: Path, raw: Path, name: str
) -> None:
    folder = tmp_path / "col"
    folder.mkdir()
    (folder / name).write_text("x", encoding="utf-8")
    with pytest.raises(OutputExistsError, match=name):
        Collection.create(folder, images=raw)
    assert [p.name for p in folder.iterdir()] == [name]
    assert (folder / name).read_text(encoding="utf-8") == "x"


def test_create_empty_collection(tmp_path: Path) -> None:
    col = Collection.create(tmp_path / "a" / "col")
    assert list(col.read_sheet().columns) == H
    assert len(col.read_sheet()) == 0
    assert Collection.open(tmp_path / "a" / "col").params() == NEUTRAL


def test_create_with_rule(tmp_path: Path, raw: Path) -> None:
    col = Collection.create(
        tmp_path / "col", images=raw, columns_from_path=r"(?P<series>D\d+)-\d+"
    )
    t = col.read_sheet()
    assert list(t.columns) == [*H, "series"]
    assert t["series"].tolist() == ["D7", "D7", "D7"]


def test_rule_numeric_values_are_numbers_in_xlsx(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("D7-17.tif", "D7-19.tif"):
        (raw / name).touch()
    col = Collection.create(
        tmp_path / "col", images=raw, columns_from_path=r"D\d+-(?P<n>\d+)"
    )
    assert col.read_sheet()["n"].tolist() == [17, 19]
    (raw / "D7-017.tif").touch()
    col.add_images(raw / "D7-017.tif")
    assert col.read_sheet()["n"].tolist() == [17, 19, None]
    col.add_images(raw / "D7-017.tif", columns_from_path=r"D\d+-(?P<m>\d+)")
    other = Collection.create(
        tmp_path / "col2",
        images=raw / "D7-017.tif",
        columns_from_path=r"D\d+-(?P<n>\d+)",
    )
    assert other.read_sheet()["n"].tolist() == ["017"]


def test_rule_values_are_text_in_csv(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "D7-17.tif").touch()
    col = Collection.create(
        tmp_path / "col",
        images=raw,
        columns_from_path=r"D\d+-(?P<n>\d+)",
        sheet_format="csv",
    )
    assert col.read_sheet()["n"].tolist() == ["17"]


def test_create_name_clash_writes_nothing(tmp_path: Path, raw: Path) -> None:
    raw2 = tmp_path / "raw2"
    raw2.mkdir()
    (raw2 / "d7-017.png").touch()
    with pytest.raises(SheetError) as e:
        Collection.create(tmp_path / "col", images=[raw, raw2])
    message = str(e.value)
    assert "../raw/D7-017.tif and ../raw2/d7-017.png" in message
    assert "'D7-017'" in message
    assert "(?P<image>" in message and "pattern=" in message
    assert not (tmp_path / "col" / "images.xlsx").exists()
    assert not (tmp_path / "col" / "agglpy.yml").exists()


def test_create_bad_name_writes_nothing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "CON.tif").touch()
    with pytest.raises(SheetError, match="device name"):
        Collection.create(tmp_path / "col", images=raw)
    assert not (tmp_path / "col" / "images.xlsx").exists()


def test_source_images_untouched(tmp_path: Path, raw: Path) -> None:
    def state() -> dict[str, tuple[bytes, int]]:
        return {
            str(p): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(raw.rglob("*"))
            if p.is_file()
        }

    before = state()
    col = Collection.create(tmp_path / "col", images=raw)
    col.add_images(raw, pattern="**/*")
    col.set_params("D7-019", preprocess={"median_blur": 3})
    assert state() == before


def test_add_images_skips_known_and_appends(
    tmp_path: Path, raw: Path, col: Collection
) -> None:
    new = tmp_path / "new"
    new.mkdir()
    (new / "D7-030.tif").touch()
    assert col.add_images([raw, new]) == ["D7-030"]
    t = col.read_sheet()
    assert t["image"].tolist() == ["D7-017", "D7-019", "D7-021", "D7-030"]
    assert t["file"].tolist()[3] == "../new/D7-030.tif"
    assert col.add_images(raw) == []


def test_add_images_keeps_typed_values(
    tmp_path: Path, raw: Path, col: Collection
) -> None:
    t = col.read_sheet()
    t.loc[1, "note"] = "blurry"
    t.loc[2, "include"] = False
    col.write_sheet(t)
    (raw / "D7-022.tif").touch()
    assert col.add_images(raw) == ["D7-022"]
    t = col.read_sheet()
    assert t.loc[1, "note"] == "blurry"
    assert t.loc[2, "include"] is False


def test_add_images_new_rule_column(raw: Path, col: Collection) -> None:
    (raw / "E1-001.tif").touch()
    assert col.add_images(raw, columns_from_path=r"(?P<series>[A-Z]\d+)-") == [
        "E1-001"
    ]
    t = col.read_sheet()
    assert list(t.columns) == [*H, "series"]
    assert t["series"].tolist() == [None, None, None, "E1"]


def test_add_images_clash_with_sheet(tmp_path: Path, col: Collection) -> None:
    new = tmp_path / "new"
    new.mkdir()
    (new / "d7-017.png").touch()
    before = col.sheet_path.read_bytes()
    with pytest.raises(
        SheetError, match=r"\.\./raw/D7-017\.tif and \.\./new/d7-017\.png"
    ):
        col.add_images(new)
    assert col.sheet_path.read_bytes() == before


def test_open_reports_yml_entry_for_unknown_image(col: Collection) -> None:
    yml(col).write_text(
        template().replace("images: {}", "images:\n  D7-099: {}"),
        encoding="utf-8",
    )
    with pytest.raises(ParamsError, match="D7-099"):
        Collection.open(col.folder)


def test_open_missing_files(tmp_path: Path, col: Collection) -> None:
    with pytest.raises(FileNotFoundError, match=r"Collection\.create"):
        Collection.open(tmp_path / "nothing")
    yml(col).unlink()
    with pytest.raises(FileNotFoundError, match=r"agglpy\.yml"):
        Collection.open(col.folder)


def test_every_call_rereads_files(col: Collection) -> None:
    col.set_params("D7-019", preprocess={"median_blur": 3})
    assert col.params("D7-019").preprocess.median_blur == 3
    text = yml(col).read_text(encoding="utf-8")
    assert "median_blur: 3" in text
    yml(col).write_text(
        text.replace("median_blur: 3", "median_blur: 5"), encoding="utf-8"
    )
    assert col.params("D7-019").preprocess.median_blur == 5
    assert col.params("D7-017").preprocess.median_blur is None


def test_set_params_checks_before_writing(col: Collection) -> None:
    before = yml(col).read_bytes()
    with pytest.raises(ParamsError, match="D7-019"):
        col.set_params("D7-019", hct={"ranges": [{"d_min": 40, "d_max": 30}]})
    assert yml(col).read_bytes() == before


def test_set_params_defaults_and_objects(col: Collection) -> None:
    col.set_params(None, preprocess=PreprocessParams(median_blur=3))
    assert col.params("D7-017").preprocess.median_blur == 3


def test_set_params_unknown_image(col: Collection) -> None:
    unknown = r"no image 'D7-099' in images\.xlsx"
    with pytest.raises(ParamsError, match=unknown):
        col.set_params("D7-099", region={"bottom_fraction": 0.1})
    with pytest.raises(ParamsError, match=unknown):
        col.params("D7-099")


def test_reset_params(col: Collection) -> None:
    col.set_params("D7-019", preprocess={"median_blur": 3})
    assert yml(col).read_text(encoding="utf-8") != template()
    col.reset_params("D7-019", "preprocess.median_blur")
    assert yml(col).read_text(encoding="utf-8") == template()
    col.set_params("D7-019", preprocess={"median_blur": 3})
    col.reset_params("D7-019")
    assert yml(col).read_text(encoding="utf-8") == template()


def test_reset_params_removes_orphan_entry(col: Collection) -> None:
    yml(col).write_text(
        template().replace("images: {}", "images:\n  D7-099: {}"),
        encoding="utf-8",
    )
    col.reset_params("D7-099")
    assert yml(col).read_text(encoding="utf-8") == template()
    with pytest.raises(ParamsError, match="no image 'D7-098'"):
        col.reset_params("D7-098")


def test_write_sheet_round_trip(col: Collection) -> None:
    t = col.read_sheet()
    t.loc[1, "note"] = "blurry"
    col.write_sheet(t)
    assert col.read_sheet().loc[1, "note"] == "blurry"


def test_write_sheet_invalid_writes_nothing(col: Collection) -> None:
    before = col.sheet_path.read_bytes()
    t = col.read_sheet()
    t.loc[0, "include"] = "maybe"
    with pytest.raises(SheetError, match="row 2, include"):
        col.write_sheet(t)
    assert col.sheet_path.read_bytes() == before


def test_write_sheet_dropping_image_with_params(col: Collection) -> None:
    col.set_params("D7-019", preprocess={"median_blur": 3})
    before = col.sheet_path.read_bytes()
    t = col.read_sheet()
    with pytest.raises(ParamsError, match=r"reset_params\('D7-019'\)"):
        col.write_sheet(t.drop(index=1))
    assert col.sheet_path.read_bytes() == before
    col.reset_params("D7-019")
    col.write_sheet(t.drop(index=1))
    assert col.read_sheet()["image"].tolist() == ["D7-017", "D7-021"]


def test_formula_sheet_fails_open(col: Collection) -> None:
    book = openpyxl.load_workbook(col.sheet_path)
    book.worksheets[0]["F1"] = "group"
    book.worksheets[0]["F2"] = "=1+1"
    book.save(col.sheet_path)
    with pytest.raises(SheetError, match="formula"):
        Collection.open(col.folder)


def test_repr(col: Collection) -> None:
    assert repr(col) == f"Collection('{col.folder.as_posix()}', 3 images)"


def test_top_level_name() -> None:
    assert agglpy.Collection is Collection


# Review focus


def test_add_images_on_locked_sheet(
    tmp_path: Path, col: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    new = tmp_path / "new"
    new.mkdir()
    (new / "D7-030.tif").touch()
    before = col.sheet_path.read_bytes()

    def locked(src: object, dst: object) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", locked)
    with pytest.raises(OutputLockedError, match=r"images\.xlsx"):
        col.add_images(new)
    monkeypatch.undo()
    assert col.sheet_path.read_bytes() == before
    assert sorted(p.name for p in col.folder.iterdir()) == [
        "agglpy",
        "agglpy.yml",
        "images.xlsx",
    ]


def test_non_ascii_and_space_paths_round_trip(tmp_path: Path) -> None:
    raw = tmp_path / "próbki 2026"
    raw.mkdir()
    (raw / "D7-019 ł.tif").touch()
    col = Collection.create(tmp_path / "kolekcja ż", images=raw)
    col = Collection.open(tmp_path / "kolekcja ż")
    t = col.read_sheet()
    assert t["image"].tolist() == ["D7-019 ł"]
    assert t["file"].tolist() == ["../próbki 2026/D7-019 ł.tif"]
    assert resolve_stored(t["file"][0], col.folder).exists()


def test_moved_tree_keeps_relative_paths(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "raw").mkdir(parents=True)
    (tree / "raw" / "D7-017.tif").touch()
    Collection.create(tree / "col", images=tree / "raw")
    moved = tmp_path / "moved"
    shutil.copytree(tree, moved)
    shutil.rmtree(tree)
    col = Collection.open(moved / "col")
    for text in col.read_sheet()["file"]:
        path = resolve_stored(text, col.folder)
        assert path.exists()
        assert path.is_relative_to(moved.resolve())


# Final review


COMMENTED = (
    "defaults: {}\nimages:\n  # tuned by hand\n  D7-019:\n    preprocess:\n"
    "      median_blur: 3\n"
)


@pytest.mark.parametrize("how", ["reset", "prune"])
def test_removing_the_only_entry_keeps_yml_readable(
    col: Collection, how: str
) -> None:
    yml(col).write_text(COMMENTED, encoding="utf-8")
    if how == "reset":
        col.reset_params("D7-019")
    else:
        col.set_params("D7-019", preprocess={"median_blur": None})
    text = yml(col).read_text(encoding="utf-8")
    assert "# tuned by hand" in text
    assert Collection.open(col.folder).params("D7-019") == NEUTRAL

