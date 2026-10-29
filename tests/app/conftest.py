"""Fixtures of the application tests: collections with disc images."""

import warnings
from pathlib import Path

import pandas as pd
import pytest

from agglpy.app import Collection
from agglpy.io import read_image, write_particles
from agglpy.params import HCTParams, HCTRange
from agglpy.pipeline import detect_particles

from support.collection import (
    DISCS,
    HCT,
    write_blank_image,
    write_disc_image,
)
from support.tiffs import write_zeiss


@pytest.fixture
def images(tmp_path: Path) -> Path:
    """``images/`` with D7-017, D7-018 and D7-019 (two discs each)."""
    folder = tmp_path / "images"
    for name, discs in DISCS.items():
        write_disc_image(folder / f"{name}.tif", discs)
    return folder


@pytest.fixture
def particles_col(tmp_path: Path, images: Path) -> Collection:
    """A collection of the three images with one HCT range set."""
    col = Collection.create(tmp_path / "col", images=images)
    col.set_params(None, hct=HCT)
    return col


# A touching pair and a single circle (px), imported as D7-018's
# correction.
CORRECTED = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "x": [40.0, 58.0, 120.0],
        "y": [40.0, 40.0, 60.0],
        "r": [10.0, 10.0, 8.0],
    }
)


@pytest.fixture
def analysis_col(tmp_path: Path) -> Collection:
    """Four images, three of them ready for analyze().

    D7-017: two discs, pixel size 2.5 nm in the sheet, detected (hct).
    D7-018: a Zeiss TIFF (pixel size from its tag) with an imported
    correction: a touching pair and a single circle. D7-019: blank,
    2.5 nm in the sheet, detected with no particles. D7-020: two discs,
    include false, note "out of focus". User columns ``series`` and
    ``flow_rate``, empty on D7-020.
    """
    folder = tmp_path / "images"
    write_disc_image(folder / "D7-017.tif", DISCS["D7-017"])
    write_zeiss(folder / "D7-018.tif", 200, 160, raw=1e-9)
    write_blank_image(folder / "D7-019.tif")
    write_disc_image(folder / "D7-020.tif", DISCS["D7-019"])
    col = Collection.create(tmp_path / "col", images=folder)
    col.set_params(None, hct=HCT)
    sheet = col.read_sheet()
    sheet["pixel_size_nm"] = [2.5, None, 2.5, None]
    sheet["include"] = [True, True, True, False]
    sheet["note"] = [None, None, None, "out of focus"]
    sheet["series"] = ["a", "a", "b", None]
    sheet["flow_rate"] = [2500, 2500, 4500, None]
    col.write_sheet(sheet)
    source = tmp_path / "fiji" / "D7-018.csv"
    source.parent.mkdir()
    write_particles(CORRECTED, source)
    with warnings.catch_warnings():
        # the blank images give NoParticlesWarning, the correction of
        # D7-018 "keeps 0 of 0 detected circles"
        warnings.simplefilter("ignore")
        col.detect(progress=False)
        col.import_particles("D7-018", source)
    return col


IMAGEJ_HEADER = " ,Index,Name,Type,Group,X,Y,Width,Height"


def _imagej_rows(particles_col: Collection, name: str) -> list[str]:
    """The detected circles as Fiji writes them, 0.4 px off."""
    image = read_image(particles_col.folder.parent / "images" / f"{name}.tif")
    found = detect_particles(image, hct=HCTParams((HCTRange(12, 44),)))
    return [
        f"{i + 1},{i},roi-{i},Oval,none,"
        f"{x - r + 0.5 + 0.4},{y - r + 0.5 + 0.4},{2 * r},{2 * r}"
        for i, (x, y, r) in enumerate(found[["x", "y", "r"]].to_numpy())
    ]


@pytest.fixture
def fiji_csv(particles_col: Collection, tmp_path: Path) -> Path:
    path = tmp_path / "fiji" / "D7-017_v2.csv"
    path.parent.mkdir()
    lines = [IMAGEJ_HEADER, *_imagej_rows(particles_col, "D7-017")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
