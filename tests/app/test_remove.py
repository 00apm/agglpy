"""Collection.remove_images: sheet row, yml entry and work folder."""

import os
import stat
import warnings
from pathlib import Path

import pytest

from agglpy.app import Collection
from agglpy.errors import (
    LeftoverFoldersWarning,
    OutputExistsError,
    OutputLockedError,
    ParamsError,
)


def _files(col: Collection) -> dict[Path, bytes]:
    """Every file of the collection folder with its bytes."""
    return {
        p: p.read_bytes() for p in sorted(col.folder.rglob("*")) if p.is_file()
    }


def test_remove_drops_row_entry_and_folder(
    analysis_col: Collection, tmp_path: Path
) -> None:
    col = analysis_col
    col.set_params("D7-017", preprocess={"median_blur": 5})
    col.set_params("D7-019", preprocess={"median_blur": 7})
    assert col.remove_images("D7-017") == ["D7-017"]
    assert col.read_sheet()["image"].tolist() == [
        "D7-018",
        "D7-019",
        "D7-020",
    ]
    text = (col.folder / "agglpy.yml").read_text(encoding="utf-8")
    assert "D7-017" not in text
    assert "D7-019" in text
    assert not (col.folder / "agglpy" / "D7-017").exists()
    assert (tmp_path / "images" / "D7-017.tif").is_file()
    with warnings.catch_warnings():
        warnings.simplefilter("error", LeftoverFoldersWarning)
        col.status()
    assert col.params("D7-019").preprocess.median_blur == 7


def test_remove_refuses_a_correction(analysis_col: Collection) -> None:
    col = analysis_col
    before = _files(col)
    with pytest.raises(OutputExistsError, match=r"D7-018.*force=True"):
        col.remove_images(["D7-017", "D7-018"])
    assert _files(col) == before


def test_remove_a_correction_with_force(analysis_col: Collection) -> None:
    col = analysis_col
    assert col.remove_images("D7-018", force=True) == ["D7-018"]
    assert not (col.folder / "agglpy" / "D7-018").exists()
    assert "D7-018" not in col.read_sheet()["image"].tolist()


def test_remove_by_mask_without_work_folder(
    analysis_col: Collection,
) -> None:
    col = analysis_col
    assert not (col.folder / "agglpy" / "D7-020").exists()
    removed = col.remove_images(col.status().particle_set == "none")
    assert removed == ["D7-020"]
    assert len(col.read_sheet()) == 3


def test_remove_unknown_name_writes_nothing(
    analysis_col: Collection,
) -> None:
    before = _files(analysis_col)
    with pytest.raises(ParamsError, match="no image 'x'"):
        analysis_col.remove_images(["D7-017", "x"])
    assert _files(analysis_col) == before


def test_remove_nothing_selected(analysis_col: Collection) -> None:
    before = _files(analysis_col)
    assert analysis_col.remove_images([]) == []
    assert _files(analysis_col) == before


def test_remove_locked_file_changes_nothing(
    analysis_col: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    col = analysis_col
    before = _files(col)
    rename = Path.rename

    def locked(self: Path, target: Path) -> Path:
        if self.name == "particles_hct.csv":  # open in Excel
            raise PermissionError(13, "in use", str(self))
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", locked)
    with pytest.raises(OutputLockedError, match=r"particles_hct\.csv"):
        col.remove_images("D7-017")
    monkeypatch.undo()
    assert _files(col) == before


def test_remove_busy_folder_changes_nothing(
    analysis_col: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    col = analysis_col
    before = _files(col)
    rename = Path.rename

    def busy(self: Path, target: Path) -> Path:
        if self.name == "D7-017":  # a terminal's working folder
            raise PermissionError(32, "in use", str(self))
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", busy)
    with pytest.raises(OutputLockedError, match="D7-017"):
        col.remove_images("D7-017")
    monkeypatch.undo()
    assert _files(col) == before


def test_remove_read_only_file(analysis_col: Collection) -> None:
    col = analysis_col
    csv = col.folder / "agglpy" / "D7-017" / "particles_hct.csv"
    os.chmod(csv, stat.S_IREAD)
    assert col.remove_images("D7-017") == ["D7-017"]
    assert not csv.parent.exists()


def test_remove_then_add_images_adds_it_again(
    analysis_col: Collection, tmp_path: Path
) -> None:
    analysis_col.remove_images("D7-017")
    assert analysis_col.add_images(tmp_path / "images") == ["D7-017"]


def test_remove_from_a_csv_sheet(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("A.tif", "B.tif"):
        (raw / name).write_bytes(name.encode())
    col = Collection.create(tmp_path / "c", images=raw, sheet_format="csv")
    assert col.remove_images("A") == ["A"]
    assert col.read_sheet()["image"].tolist() == ["B"]
