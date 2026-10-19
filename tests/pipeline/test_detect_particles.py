"""Detection on one image (``agglpy.pipeline.detect_particles``)."""

import warnings

import cv2
import numpy as np
import pytest

import agglpy.pipeline._detection as detection
from agglpy.errors import ImageError, NoParticlesWarning
from agglpy.params import HCTParams, HCTRange, PreprocessParams, RegionParams
from agglpy.pipeline import detect_particles

HCT = HCTParams(ranges=(HCTRange(12, 44),))
CUT = RegionParams(bottom_fraction=0.25)

# cv2.circle's sub-pixel mode: coordinates and radius in 1/16 px
SHIFT = 4


def _draw(height: int, width: int, discs) -> np.ndarray:
    """Bright anti-aliased discs (x, y, r in px) on a dark image."""
    image = np.full((height, width), 20, dtype=np.uint8)
    for x, y, r in discs:
        cv2.circle(
            image,
            (round(x * 2**SHIFT), round(y * 2**SHIFT)),
            round(r * 2**SHIFT),
            200,
            thickness=-1,
            lineType=cv2.LINE_AA,
            shift=SHIFT,
        )
    return image


# 200 x 160 px; a quarter cut off leaves rows 0..119: the second disc
# lies on the cut-off bar
IMAGE = _draw(160, 200, [(60.3, 50.6, 14.2), (150.4, 140.7, 14.2)])


def test_bottom_cut_keeps_raw_frame_coordinates_and_drops_the_bar():
    found = detect_particles(IMAGE, region=CUT, hct=HCT)
    assert found[["x", "y", "r"]].values.tolist() == [[60.5, 50.5, 15.0]]
    # without the cut both discs are found
    assert len(detect_particles(IMAGE, hct=HCT)) == 2


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
