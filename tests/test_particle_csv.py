"""Import of external primary particle .csv files (agglpy, agglpy_old, ImageJ)."""

import logging
from pathlib import Path

import pytest

from agglpy.errors import ParticleCsvStructureError
from agglpy.img_ds import (
    load_agglpy_csv,
    load_agglpy_old_csv,
    load_imagej_csv,
    recognize_particle_csv,
)

# Minimal valid content per format: (loader, header, one data row)
AGGLPY = (load_agglpy_csv, "ID,X,Y,R", "0,100,100,30")
AGGLPY_OLD = (
    load_agglpy_old_csv,
    "ID,X (pixels),Y (pixels),Radius (pixels)",
    "0,100.5,100.5,30.0",
)
IMAGEJ_HEADER = " ,Index,Name,Type,Group,X,Y,Width,Height"
IMAGEJ = (load_imagej_csv, IMAGEJ_HEADER, "1,0,0001-0100,Oval,none,70,70,60,60")


def write_csv(path: Path, lines: list[str], bom: bool = False) -> Path:
    """Write lines as a UTF-8 file, optionally with a byte order mark."""
    encoding = "utf-8-sig" if bom else "utf-8"
    path.write_text("\n".join(lines) + "\n", encoding=encoding)
    return path


# ----------- encoding: files are read as UTF-8 on every platform


@pytest.mark.parametrize(
    ("loader", "header", "row"),
    [AGGLPY, AGGLPY_OLD],
    ids=["agglpy", "agglpy_old"],
)
def test_loader_reads_non_ascii_as_utf8(tmp_path: Path, loader, header, row):
    """A UTF-8 'µm' must stay 'µm' (with encoding="ansi" it became 'Âµm')."""
    path = write_csv(tmp_path / "p.csv", [header + ",unit", row + ",µm"])

    df = loader(path)

    assert df.loc[0, "unit"] == "µm"


@pytest.mark.parametrize(
    ("loader", "header", "row", "expected_type"),
    [(*AGGLPY, "agglpy"), (*AGGLPY_OLD, "agglpy_old"), (*IMAGEJ, "ImageJ")],
    ids=["agglpy", "agglpy_old", "ImageJ"],
)
def test_loader_handles_utf8_bom(
    tmp_path: Path, loader, header, row, expected_type
):
    """Excel and some Windows tools start UTF-8 files with a byte order mark.

    It must not end up glued to the first column name.
    """
    path = write_csv(tmp_path / "p.csv", [header, row], bom=True)

    assert recognize_particle_csv(path) == expected_type
    df = loader(path)
    assert len(df.index) == 1


def test_real_imagej_csv_loads(input_multi_wdir: Path):
    """The ImageJ export used in the test data loads completely."""
    path = input_multi_wdir / "images" / "D7-017" / "D7-017_fitting.csv"

    df = load_imagej_csv(path)

    assert len(df.index) == 1003
    assert list(df.columns) == ["ID", "X", "Y", "D", "R"]


# ----------- ImageJ: non-circular ROIs are dropped with a warning


def test_imagej_drops_non_circular_rois_with_warning(tmp_path: Path, caplog):
    rows = [
        "1,0,0001-0100,Oval,none,70,70,60,60",  # circle: kept
        "2,1,0002-0200,Oval,none,170,170,60,40",  # ellipse: dropped
        "3,2,0003-0300,Rectangle,none,270,270,60,60",  # not an oval: dropped
        "4,3,0004-0400,Oval,none,370,370,20,20",  # circle: kept
    ]
    path = write_csv(tmp_path / "p.csv", [IMAGEJ_HEADER, *rows])

    with caplog.at_level(logging.WARNING, logger="agglpy"):
        df = load_imagej_csv(path)

    assert df["ID"].tolist() == [0, 3]
    assert len(caplog.records) == 1
    assert "Dropped 2 particles" in caplog.text
    assert "[1, 2]" in caplog.text


def test_imagej_all_circles_no_warning(input_multi_wdir: Path, caplog):
    """The real ImageJ export contains only circles, so nothing is dropped."""
    path = input_multi_wdir / "images" / "D7-017" / "D7-017_fitting.csv"

    with caplog.at_level(logging.WARNING, logger="agglpy"):
        load_imagej_csv(path)

    assert caplog.records == []


# ----------- missing columns raise ParticleCsvStructureError, not assert


@pytest.mark.parametrize(
    ("loader", "header", "row"),
    [AGGLPY, AGGLPY_OLD, IMAGEJ],
    ids=["agglpy", "agglpy_old", "ImageJ"],
)
def test_loader_missing_column_raises(tmp_path: Path, loader, header, row):
    """Validation must not rely on assert (skipped when Python runs with -O)."""
    # drop the last column from header and row
    path = write_csv(
        tmp_path / "p.csv",
        [header.rsplit(",", 1)[0], row.rsplit(",", 1)[0]],
    )

    with pytest.raises(ParticleCsvStructureError, match=r"p\.csv"):
        loader(path)


# ----------- type recognition


@pytest.mark.parametrize("full", [False, True])
def test_recognize_particle_csv_full_or_head(input_multi_wdir: Path, full: bool):
    """full=True reads the whole file instead of the first rows."""
    path = input_multi_wdir / "images" / "D7-017" / "D7-017_fitting.csv"

    assert recognize_particle_csv(path, full=full) == "ImageJ"
