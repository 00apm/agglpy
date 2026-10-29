"""Fixtures of the application tests: collections with disc images."""

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import tifffile

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
from support.discs import draw_discs
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


# Small discs (r 6) for the first range, large ones (r 14) for the
# second, and on T1 one disc (r 8.5) that both ranges find
TUNING_DISCS = {
    "T1": [
        (30.3, 30.6, 6),
        (80.2, 40.4, 6),
        (130.6, 30.2, 6),
        (40.5, 100.3, 14),
        (150.6, 90.2, 14),
        (95.4, 110.7, 8.5),
    ],
    "T2": [
        (50.3, 20.6, 6),
        (170.2, 40.4, 6),
        (60.5, 80.3, 14),
        (130.6, 110.2, 14),
    ],
}
TUNING_PARAMS = {
    "region": {"bottom_fraction": 0.1},
    "preprocess": {
        "median_blur": 3,
        "rolling_ball": {"radius": 20},
        "clahe": {"clip_limit": 2.0, "tile_grid": [4, 4]},
    },
    "hct": {"ranges": [{"d_min": 8, "d_max": 18}, {"d_min": 16, "d_max": 44}]},
}


@pytest.fixture
def tuning_col(tmp_path: Path) -> Collection:
    """Three 200 x 160 px images with two HCT ranges set, for tuning.

    With the defaults (region bottom 10 % cut, median 3, rolling ball
    r 20, CLAHE) range 1 (d 8-18) finds T1's three small discs and its
    r 8.5 disc, range 2 (d 16-44) the two large ones and the r 8.5 disc
    again (a duplicate pair). T2 has two small and two large discs and
    its own entry (range 1 with accumulator 14); T3.jpg is T1 as a
    JPEG.
    """
    folder = tmp_path / "images"
    folder.mkdir()
    t1 = draw_discs(160, 200, TUNING_DISCS["T1"])
    tifffile.imwrite(folder / "T1.tif", t1)
    tifffile.imwrite(
        folder / "T2.tif", draw_discs(160, 200, TUNING_DISCS["T2"])
    )
    ok, jpeg = cv2.imencode(".jpg", t1, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    np.asarray(jpeg).tofile(folder / "T3.jpg")
    col = Collection.create(tmp_path / "col", images=folder)
    col.set_params(None, **TUNING_PARAMS)
    col.set_params(
        "T2",
        hct={
            "ranges": [
                {"d_min": 8, "d_max": 18, "accumulator_threshold": 14},
                {"d_min": 16, "d_max": 44},
            ]
        },
    )
    return col
