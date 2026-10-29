"""Detection on one image (``agglpy.pipeline.detect_particles``)."""

import warnings
from dataclasses import replace

import cv2
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
from agglpy.pipeline._detection import (
    configured_steps,
    cut_region,
    hct_pass,
    matching_stages,
    particles_from_passes,
    preprocess_stages,
)
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
    real = detection.hct_pass

    def spy(image, hct_range):
        calls.append(hct_range)
        return real(image, hct_range)

    monkeypatch.setattr(detection, "hct_pass", spy)
    hct = HCTParams(ranges=(HCTRange(8, 18), HCTRange(12, 44)))
    detect_particles(IMAGE, hct=hct)
    assert calls == list(hct.ranges)


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


# small discs (r 6) for the first range, large (r 14) for the second
TWO_SIZES = draw_discs(
    160,
    200,
    [(30.3, 30.6, 6), (80.2, 40.4, 6), (60.5, 100.3, 14), (150.6, 60.2, 14)],
)
TWO_RANGES = HCTParams(ranges=(HCTRange(8, 18), HCTRange(20, 44)))
MEDIAN_CLAHE = PreprocessParams(median_blur=3, clahe=ClaheParams(2.0, (4, 4)))


def test_helpers_give_detect_particles():
    region = RegionParams(bottom_fraction=0.1)
    bounds, cut = cut_region(TWO_SIZES, region)
    assert cut.shape == (144, 200)
    assert np.shares_memory(cut, TWO_SIZES)
    stages = preprocess_stages(cut, MEDIAN_CLAHE)
    assert [(s.field, s.name) for s in stages] == [
        ("median_blur", "median"),
        ("clahe", "clahe"),
    ]
    assert stages[0].value == 3
    assert stages[1].value == MEDIAN_CLAHE.clahe
    assert not any(s.image.flags.writeable for s in stages)
    passes = [hct_pass(stages[-1].image, r) for r in TWO_RANGES.ranges]
    assert [len(found) for found in passes] == [2, 2]
    pd.testing.assert_frame_equal(
        particles_from_passes(stages[-1].image, passes, bounds),
        detect_particles(
            TWO_SIZES, region=region, preprocess=MEDIAN_CLAHE, hct=TWO_RANGES
        ),
    )


def test_stages_reuse_the_leading_equal_steps(monkeypatch):
    counts = {"median": 0, "clahe": 0}
    median, create_clahe = cv2.medianBlur, cv2.createCLAHE

    def count_median(*args, **kwargs):
        counts["median"] += 1
        return median(*args, **kwargs)

    def count_clahe(*args, **kwargs):
        counts["clahe"] += 1
        return create_clahe(*args, **kwargs)

    monkeypatch.setattr(cv2, "medianBlur", count_median)
    monkeypatch.setattr(cv2, "createCLAHE", count_clahe)
    _, cut = cut_region(TWO_SIZES, RegionParams())
    first = preprocess_stages(cut, MEDIAN_CLAHE)
    steps = []
    stronger = replace(MEDIAN_CLAHE, clahe=ClaheParams(3.0, (4, 4)))
    again = preprocess_stages(cut, stronger, reuse=first, on_step=steps.append)
    assert again[0] is first[0]
    assert again[1] is not first[1]
    assert counts == {"median": 1, "clahe": 2}
    assert steps == ["clahe"]
    # the same params: every stage reused, nothing reported
    steps.clear()
    same = preprocess_stages(cut, stronger, reuse=again, on_step=steps.append)
    assert all(a is b for a, b in zip(same, again, strict=True))
    assert steps == []
    # median switched off: clahe now runs on the cut, nothing reused
    alone = preprocess_stages(
        cut, replace(stronger, median_blur=None), reuse=again
    )
    assert alone[0] is not again[1]
    assert counts == {"median": 1, "clahe": 3}
    # the rule itself, as a preview asks it before running anything
    assert matching_stages(first, stronger) == 1
    assert matching_stages(again, stronger) == 2
    assert matching_stages(again, replace(stronger, median_blur=None)) == 0
    assert configured_steps(stronger) == [
        ("median_blur", stronger.median_blur),
        ("clahe", stronger.clahe),
    ]


def test_no_preprocessing_detects_on_the_cut():
    bounds, cut = cut_region(TWO_SIZES, RegionParams())
    assert preprocess_stages(cut, PreprocessParams()) == ()
    passes = [hct_pass(cut, r) for r in TWO_RANGES.ranges]
    pd.testing.assert_frame_equal(
        particles_from_passes(cut, passes, bounds),
        detect_particles(TWO_SIZES, hct=TWO_RANGES),
    )
