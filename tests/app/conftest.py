"""Fixtures of the application tests: collections with disc images."""

from pathlib import Path

import pytest

from agglpy.app import Collection

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
