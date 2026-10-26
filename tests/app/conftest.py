"""Fixtures of the application tests: collections with disc images."""

from pathlib import Path

import pytest

from agglpy.app import Collection
from agglpy.io import read_image
from agglpy.params import HCTParams, HCTRange
from agglpy.pipeline import detect_particles

from support.collection import DISCS, HCT, write_disc_image


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
