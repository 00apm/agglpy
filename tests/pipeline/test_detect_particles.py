"""Detection on one image (``agglpy.pipeline.detect_particles``)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import agglpy.pipeline._detection as detection
from agglpy.core.detect import detect_hct
from agglpy.core.preprocess import preprocess
from agglpy.core.region import Bounds
from agglpy.errors import ImageError, NoParticlesWarning
from agglpy.params import (
    ClaheParams,
    HCTParams,
    HCTRange,
    PreprocessParams,
    RegionParams,
    RollingBallParams,
)
from agglpy.pipeline import detect_particles
from agglpy.tables import validate_particles

from support.discs import draw_discs

HCT = HCTParams(ranges=(HCTRange(12, 44),))
CUT = RegionParams(bottom_fraction=0.25)

# 200 x 160 px; a quarter cut off leaves rows 0..119: the second disc
# lies on the cut-off bar
IMAGE = draw_discs(160, 200, [(60.3, 50.6, 14.2), (150.4, 140.7, 14.2)])


def test_bottom_cut_keeps_raw_frame_coordinates_and_drops_the_bar():
    found = detect_particles(IMAGE, region=CUT, hct=HCT)
    assert found[["x", "y", "r"]].values.tolist() == [[60.5, 50.5, 15.0]]
    # without the cut both discs are found
    assert len(detect_particles(IMAGE, hct=HCT)) == 2


def test_a_top_left_cut_is_shifted_back_to_the_raw_frame(monkeypatch):
    # RegionParams only cuts the bottom; a region starting at column 10,
    # row 20 stands in for a later top / left cut
    monkeypatch.setattr(
        detection,
        "region_bounds",
        lambda region, width, height: Bounds(9.5, 19.5, 199.5, 119.5),
    )
    found = detect_particles(IMAGE, hct=HCT)
    assert found[["x", "y", "r"]].values.tolist() == [[60.5, 50.5, 15.0]]


def test_preprocess_params_reach_the_core(monkeypatch):
    calls = []
    real = detection.core_preprocess

    def spy(image, params):
        calls.append((image.shape, params))
        return real(image, params)

    monkeypatch.setattr(detection, "core_preprocess", spy)
    params = PreprocessParams(median_blur=3)
    detect_particles(IMAGE, region=CUT, preprocess=params, hct=HCT)
    assert calls == [((120, 200), params)]


def test_hct_params_reach_the_core(monkeypatch):
    calls = []
    real = detection.detect_hct

    def spy(image, params):
        calls.append(params)
        return real(image, params)

    monkeypatch.setattr(detection, "detect_hct", spy)
    detect_particles(IMAGE, hct=HCT)
    assert calls == [HCT]


def test_result_is_a_validated_particle_table():
    found = detect_particles(IMAGE, hct=HCT)
    assert found.columns.tolist() == ["id", "x", "y", "r"]
    assert found["id"].tolist() == list(range(1, len(found) + 1))
    assert found["id"].dtype == np.int64
    assert (found[["x", "y", "r"]].dtypes == np.float64).all()


def test_input_image_is_not_modified():
    image = IMAGE.copy()
    detect_particles(
        image, region=CUT, preprocess=PreprocessParams(median_blur=3), hct=HCT
    )
    np.testing.assert_array_equal(image, IMAGE)


def test_colour_image_raises_image_error():
    with pytest.raises(ImageError, match=r"\(40, 50, 3\)"):
        detect_particles(np.zeros((40, 50, 3), np.uint8), hct=HCT)


def test_nothing_found_warns_once():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        found = detect_particles(np.full((60, 80), 20, np.uint8), hct=HCT)
    assert [w.category for w in record] == [NoParticlesWarning]
    assert found.empty
    assert found.columns.tolist() == ["id", "x", "y", "r"]


PRE = PreprocessParams(
    median_blur=3,
    rolling_ball=RollingBallParams(radius=50),
    clahe=ClaheParams(2.0, (8, 8)),
)


def test_on_step_names_each_configured_step_in_order():
    steps = []
    detect_particles(IMAGE, preprocess=PRE, hct=HCT, on_step=steps.append)
    assert steps == ["median", "rolling ball", "clahe", "hct"]
    steps.clear()
    detect_particles(
        IMAGE,
        preprocess=PreprocessParams(clahe=PRE.clahe),
        hct=HCT,
        on_step=steps.append,
    )
    assert steps == ["clahe", "hct"]


def test_step_by_step_gives_the_same_circles():
    # no region cut: detect_particles shifts by 0, so the core in one
    # call is the reference
    reference = validate_particles(
        detect_hct(preprocess(IMAGE, PRE), HCT), warn_duplicates=False
    )
    pd.testing.assert_frame_equal(
        detect_particles(IMAGE, preprocess=PRE, hct=HCT), reference
    )


def test_input_image_unchanged_without_preprocessing():
    before = IMAGE.copy()
    detect_particles(IMAGE, hct=HCT)
    np.testing.assert_array_equal(IMAGE, before)
