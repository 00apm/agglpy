"""The new detector finds the golden HCT circles on D7-017 and D7-019.

Region cut (bottom fraction 0.1), no preprocessing, the golden HCT
ranges: ``detect_hct`` must give exactly the circles pinned in
``expected/<name>_hct_circles.csv`` by the v0.4 golden run. Exact,
because both sides are OpenCV's float32 values as their shortest
decimals. Compared by content (sorted by x, y, r), never by id.
"""

import warnings

import pandas as pd
import pytest

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.detect import detect_hct
from agglpy.core.preprocess import preprocess
from agglpy.core.properties import agglomerate_properties, particle_properties
from agglpy.core.region import region_bounds, region_slices
from agglpy.errors import DuplicateParticlesWarning
from agglpy.io import read_image
from agglpy.params import HCTParams, HCTRange, PreprocessParams, RegionParams

from support.paths import DATA_DIR

IMAGES_DIR = DATA_DIR / "input" / "multiple_image" / "images"
EXPECTED_DIR = DATA_DIR / "golden" / "expected"

# The golden settings (tests/data/golden/D7/settings.yml)
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

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", list(RANGES))
def test_detect_hct_finds_the_golden_circles(name: str) -> None:
    raw = read_image(IMAGES_DIR / name / f"{name}.tif")
    height, width = raw.shape
    bounds = region_bounds(RegionParams(bottom_fraction=0.1), width, height)
    rows, cols = region_slices(bounds)
    image = preprocess(raw[rows, cols], PreprocessParams())
    # two pairs per image at the boundary between the ranges (r ~ 30)
    with pytest.warns(DuplicateParticlesWarning, match="^2 pairs"):
        found = detect_hct(image, HCTParams(ranges=RANGES[name]))

    expected = pd.read_csv(
        EXPECTED_DIR / f"{name}_hct_circles.csv",
        usecols=["X", "Y", "R"],
        # the default parser misses the last bit of some values
        float_precision="round_trip",
    ).rename(columns={"X": "x", "Y": "y", "R": "r"})
    circles = found[["x", "y", "r"]].sort_values(
        ["x", "y", "r"], ignore_index=True
    )
    pd.testing.assert_frame_equal(circles, expected, check_exact=True)

    n_circles, n_edge, n_agglomerates, n_agglomerates_edge = COUNTS[name]
    assert len(found) == n_circles
    particles = particle_properties(found, bounds=bounds)
    assert particles["touches_edge"].sum() == n_edge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DuplicateParticlesWarning)
        grouped = find_agglomerates(found)
    agglomerates = agglomerate_properties(grouped, bounds=bounds)
    assert len(agglomerates) == n_agglomerates
    assert agglomerates["touches_edge"].sum() == n_agglomerates_edge
