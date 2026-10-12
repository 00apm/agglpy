"""Regression snapshot of the corrected path: Fiji CSV -> agglomerates.

The author's real workflow: HCT circles corrected by hand in Fiji,
exported from the ROI Manager as CSV, then grouped and measured. The
golden tests (test_golden.py) cover only the automatic HCT path.

The snapshot was checked once against agglpy 0.4 (item 2.2 Findings):
the same particles (0.5 px apart, D-051), the same agglomerates and
enclosed flags, the same sizes; only the centre of mass and Rg differ,
by definition (D-040).

Compared by content, never by id: particles are sorted by position and
size, and each agglomerate is labelled by the row of its first member.
After an intended change of results:

    pytest tests/golden/test_corrected.py --force-regen
"""

from pathlib import Path

import pandas as pd
import pytest

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.properties import agglomerate_properties
from agglpy.io import read_particles

from support.paths import DATA_DIR

IMAGES_DIR = DATA_DIR / "input" / "multiple_image" / "images"
IMAGE_NAMES = ["D7-017", "D7-019"]
# Sums over many members differ in the last digits between platforms
FLOAT_TOL = {"rtol": 1e-9, "atol": 0.0}

pytestmark = pytest.mark.slow


@pytest.fixture
def original_datadir() -> Path:
    """Where pytest-regressions reads and writes the expected files."""
    return DATA_DIR / "golden" / "expected"


def _canonical(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Particle and agglomerate tables of one image, without ids."""
    found = find_agglomerates(
        read_particles(IMAGES_DIR / name / f"{name}_fitting.csv")
    )
    particles = found.sort_values(["x", "y", "r"], ignore_index=True)
    assert not particles.duplicated(["x", "y", "r"]).any()
    # an agglomerate is labelled by the row of its first member
    label = particles.index.to_series().groupby(particles["agglomerate_id"])
    particles["agglomerate"] = label.transform("min")
    label_of = particles.groupby("agglomerate_id")["agglomerate"].first()

    agglomerates = agglomerate_properties(found)
    agglomerates.insert(
        0, "agglomerate", agglomerates["agglomerate_id"].map(label_of)
    )
    agglomerates = agglomerates.drop(columns="agglomerate_id")
    agglomerates = agglomerates.sort_values("agglomerate", ignore_index=True)
    particles = particles[["x", "y", "r", "enclosed", "agglomerate"]]
    return particles, agglomerates


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_corrected_membership(dataframe_regression, name: str) -> None:
    """The same particles are grouped together, the same are enclosed."""
    particles, _ = _canonical(name)
    dataframe_regression.check(
        particles, basename=f"{name}_corrected_particles"
    )


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_corrected_agglomerates(dataframe_regression, name: str) -> None:
    """Each agglomerate keeps its properties (px)."""
    _, agglomerates = _canonical(name)
    dataframe_regression.check(
        agglomerates,
        basename=f"{name}_corrected_agglomerates",
        default_tolerance=FLOAT_TOL,
    )
