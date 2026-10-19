"""Regression snapshots of the pipeline on D7-017 and D7-019.

Two paths per image, both through ``agglpy.pipeline``:

- automatic: ``detect_particles`` (region cut 0.1, no preprocessing,
  the HCT ranges below) → ``analyze``;
- corrected: the circles corrected by hand in Fiji
  (``<name>_fitting.csv``) → ``analyze``.

Pinned in px: the detected circles, which particles are grouped
together and enclosed, the edge flags and every agglomerate property.
Compared by content, never by id: particles are sorted by position and
size, and each agglomerate is labelled by the row of its first member.
The snapshots were checked once against agglpy 0.4.0 (same circles,
same agglomerates; only renamed, re-defined or px-exact values differ).

After an intended change of results, regenerate and review the diff:

    pytest tests/golden/test_pipeline_golden.py --force-regen
"""

import functools
import warnings
from pathlib import Path

import pandas as pd
import pytest

from agglpy.errors import DuplicateParticlesWarning
from agglpy.io import read_image, read_image_metadata, read_particles
from agglpy.params import HCTParams, HCTRange, RegionParams
from agglpy.pipeline import ImageResult, analyze, detect_particles

from support.paths import DATA_DIR

IMAGES_DIR = DATA_DIR / "input" / "multiple_image" / "images"
IMAGE_NAMES = ["D7-017", "D7-019"]
REGION = RegionParams(bottom_fraction=0.1)

# The settings of the v0.4 golden run (tests/data/golden/D7/settings.yml)
RANGES = {
    "D7-017": (
        HCTRange(3, 60, canny_threshold=220, accumulator_threshold=13),
        HCTRange(60, 240, canny_threshold=250, accumulator_threshold=24),
    ),
    "D7-019": (
        HCTRange(3, 60, canny_threshold=200, accumulator_threshold=14),
        HCTRange(60, 160, canny_threshold=170, accumulator_threshold=22),
    ),
}
# circles, and how many particles / agglomerates touch the region's edge
COUNTS = {"D7-017": (784, 12, 422, 12), "D7-019": (673, 9, 469, 9)}
# particles of the corrected sets touching the region's edge
CORRECTED_EDGE = {"D7-017": 19, "D7-019": 40}

EXACT = {"rtol": 0.0, "atol": 0.0}
# Sums over many members differ in the last digits between platforms
FLOAT_TOL = {"rtol": 1e-9, "atol": 0.0}

pytestmark = pytest.mark.slow


@pytest.fixture
def original_datadir() -> Path:
    """Where pytest-regressions reads and writes the expected files."""
    return DATA_DIR / "golden" / "expected" / "pipeline"


@functools.cache
def _automatic(name: str) -> tuple[pd.DataFrame, list, ImageResult]:
    """Detect and analyse once per image; the warnings of detection."""
    tif = IMAGES_DIR / name / f"{name}.tif"
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        circles = detect_particles(
            read_image(tif), region=REGION, hct=HCTParams(ranges=RANGES[name])
        )
    meta = read_image_metadata(tif)
    with warnings.catch_warnings():
        # the enclosed duplicates of the boundary pairs, warned again
        warnings.simplefilter("ignore", DuplicateParticlesWarning)
        result = analyze(
            circles, width=meta.width, height=meta.height, region=REGION
        )
    return circles, record, result


@functools.cache
def _corrected(name: str) -> ImageResult:
    meta = read_image_metadata(IMAGES_DIR / name / f"{name}.tif")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a clean set: no warning
        particles = read_particles(IMAGES_DIR / name / f"{name}_fitting.csv")
        return analyze(
            particles, width=meta.width, height=meta.height, region=REGION
        )


def _canonical(result: ImageResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Particle and agglomerate tables of one result, without ids."""
    particles = result.particles.sort_values(
        ["x", "y", "r"], ignore_index=True
    )
    assert not particles.duplicated(["x", "y", "r"]).any()
    # an agglomerate is labelled by the row of its first member
    label = particles.index.to_series().groupby(particles["agglomerate_id"])
    particles["agglomerate"] = label.transform("min")
    label_of = particles.groupby("agglomerate_id")["agglomerate"].first()

    agglomerates = result.agglomerates.copy()
    agglomerates.insert(
        0, "agglomerate", agglomerates["agglomerate_id"].map(label_of)
    )
    agglomerates = agglomerates.drop(columns="agglomerate_id")
    agglomerates = agglomerates.sort_values("agglomerate", ignore_index=True)
    particles = particles[
        ["x", "y", "r", "enclosed", "touches_edge", "agglomerate"]
    ]
    return particles, agglomerates


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_automatic_circles(dataframe_regression, name: str) -> None:
    """Detection finds the same circles (px)."""
    circles, record, _ = _automatic(name)
    # two pairs per image at the boundary between the ranges (r ~ 30)
    assert [w.category for w in record] == [DuplicateParticlesWarning]
    assert str(record[0].message).startswith("2 pairs")
    assert len(circles) == COUNTS[name][0]
    dataframe_regression.check(
        circles[["x", "y", "r"]].sort_values(
            ["x", "y", "r"], ignore_index=True
        ),
        basename=f"{name}_circles",
        default_tolerance=EXACT,
    )


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_automatic_particles(dataframe_regression, name: str) -> None:
    """The same particles are grouped, enclosed and touch the edge."""
    _, _, result = _automatic(name)
    particles, _ = _canonical(result)
    assert particles["touches_edge"].sum() == COUNTS[name][1]
    assert result.outside_ids == ()
    dataframe_regression.check(
        particles, basename=f"{name}_particles", default_tolerance=EXACT
    )


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_automatic_agglomerates(dataframe_regression, name: str) -> None:
    """Each agglomerate keeps its properties (px)."""
    _, _, result = _automatic(name)
    _, agglomerates = _canonical(result)
    assert len(agglomerates) == COUNTS[name][2]
    assert agglomerates["touches_edge"].sum() == COUNTS[name][3]
    dataframe_regression.check(
        agglomerates,
        basename=f"{name}_agglomerates",
        default_tolerance=FLOAT_TOL,
    )


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_corrected_particles(dataframe_regression, name: str) -> None:
    """The corrected set: same grouping, enclosed and edge flags."""
    result = _corrected(name)
    particles, _ = _canonical(result)
    assert result.outside_ids == ()
    assert particles["touches_edge"].sum() == CORRECTED_EDGE[name]
    dataframe_regression.check(
        particles,
        basename=f"{name}_corrected_particles",
        default_tolerance=EXACT,
    )


@pytest.mark.parametrize("name", IMAGE_NAMES)
def test_corrected_agglomerates(dataframe_regression, name: str) -> None:
    """The corrected set: each agglomerate keeps its properties (px)."""
    _, agglomerates = _canonical(_corrected(name))
    dataframe_regression.check(
        agglomerates,
        basename=f"{name}_corrected_agglomerates",
        default_tolerance=FLOAT_TOL,
    )
