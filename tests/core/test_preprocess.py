"""Image preprocessing (``agglpy.core.preprocess``), on small arrays."""

import cv2
import numpy as np
import pytest
from skimage.restoration import rolling_ball

from agglpy.core.preprocess import (
    _clahe,
    _median_blur,
    _subtract_background,
    preprocess,
)
from agglpy.errors import ImageError
from agglpy.params import ClaheParams, PreprocessParams, RollingBallParams

H, W = 60, 80
ROWS, COLS = np.mgrid[0:H, 0:W]
DISC_CENTRES = [(20, 20), (55, 35), (40, 45)]  # (x, y)
# three discs of radius 4 px
DISCS = np.zeros((H, W), dtype=bool)
for _x, _y in DISC_CENTRES:
    DISCS |= (COLS - _x) ** 2 + (ROWS - _y) ** 2 <= 16

ALL_STEPS = PreprocessParams(
    median_blur=3,
    rolling_ball=RollingBallParams(radius=15),
    clahe=ClaheParams(clip_limit=2.0, tile_grid=(4, 4)),
)


def _noisy(seed: int = 0) -> np.ndarray:
    """A dark image with a background gradient, bright discs and noise."""
    rng = np.random.default_rng(seed)
    values = 20 + 60 * COLS / (W - 1) + 120 * DISCS + rng.normal(0, 8, (H, W))
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def test_no_step_gives_an_equal_new_array():
    image = _noisy()
    out = preprocess(image, PreprocessParams())
    assert np.array_equal(out, image)
    assert out.dtype == np.uint8
    assert not np.shares_memory(out, image)


@pytest.mark.parametrize(
    "params",
    [
        PreprocessParams(median_blur=5),
        PreprocessParams(rolling_ball=RollingBallParams(radius=15)),
        PreprocessParams(
            rolling_ball=RollingBallParams(radius=15, light_background=True)
        ),
        PreprocessParams(clahe=ClaheParams(clip_limit=2.0, tile_grid=(4, 4))),
        ALL_STEPS,
    ],
    ids=["median", "rolling_ball", "rolling_ball_light", "clahe", "all"],
)
def test_input_is_never_modified(params):
    image = _noisy()
    before = image.copy()
    out = preprocess(image, params)
    assert np.array_equal(image, before)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert not np.shares_memory(out, image)


def test_order_is_median_rolling_ball_clahe():
    image = _noisy()
    expected = _clahe(
        _subtract_background(
            _median_blur(image, 3), RollingBallParams(radius=15)
        ),
        ClaheParams(clip_limit=2.0, tile_grid=(4, 4)),
    )
    assert np.array_equal(preprocess(image, ALL_STEPS), expected)
    # the legacy order (median, CLAHE, rolling ball) gives another image,
    # so this test tells the orders apart
    legacy = _subtract_background(
        _clahe(_median_blur(image, 3), ClaheParams(2.0, (4, 4))),
        RollingBallParams(radius=15),
    )
    assert not np.array_equal(expected, legacy)


def test_median_and_clahe_are_the_opencv_calls():
    image = _noisy()
    assert np.array_equal(
        preprocess(image, PreprocessParams(median_blur=5)),
        cv2.medianBlur(image, 5),
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    assert np.array_equal(
        preprocess(image, PreprocessParams(clahe=ClaheParams(2.0, (4, 4)))),
        clahe.apply(image),
    )


def test_rolling_ball_dark_background():
    # gradient 20 ... 80 with discs 120 brighter: the background goes to
    # about 0 (the ball's curvature leaves a few grey levels on the
    # slope), the discs keep their height
    image = np.rint(20 + 60 * COLS / (W - 1) + 120 * DISCS).astype(np.uint8)
    out = _subtract_background(image, RollingBallParams(radius=15))
    assert out[~DISCS].max() <= 5
    assert out[DISCS].min() >= 120


def test_rolling_ball_light_background_keeps_the_polarity():
    # background 230, dark discs 60: inverted 25 / 195, subtracted 0 /
    # 170, inverted back 255 / 85 (+1 next to a small disc, where the
    # ball's curvature lifts the estimate by 0.86)
    image = np.where(DISCS, 60, 230).astype(np.uint8)
    out = _subtract_background(
        image, RollingBallParams(radius=15, light_background=True)
    )
    assert np.unique(out[~DISCS]).tolist() == [255]
    assert np.abs(out[DISCS].astype(int) - 85).max() <= 1


def test_rolling_ball_light_background_on_a_gradient():
    # the old method returned an all-zero image here
    image = np.rint(200 + 40 * COLS / (W - 1) - 170 * DISCS).astype(np.uint8)
    out = _subtract_background(
        image, RollingBallParams(radius=15, light_background=True)
    )
    assert out[~DISCS].min() >= 250
    assert out[DISCS].max() <= 90


def test_presmooth_does_not_blur_the_result():
    # One bright pixel: the background is estimated on the blurred copy
    # (where the spike is 200 / 9), but subtracted from the original, so
    # the spike stays sharp. Blurring the result would leave about 22.
    image = np.zeros((21, 21), dtype=np.uint8)
    image[10, 10] = 200
    out = _subtract_background(image, RollingBallParams(radius=5))
    assert out[10, 10] == 200
    out[10, 10] = 0
    assert not out.any()


def test_rolling_ball_rounds_once_at_the_end():
    # The background has fractions; truncating them (scikit-image's own
    # uint8 cast) would make the result up to 1 level too low.
    image = np.rint(20 + 60 * COLS / (W - 1) + 120 * DISCS).astype(np.uint8)
    params = RollingBallParams(radius=7.3, presmooth=False)
    values = image.astype(np.float64)
    difference = values - rolling_ball(values, radius=7.3)
    rounded = np.clip(np.rint(difference), 0, 255).astype(np.uint8)
    truncated = np.clip(np.trunc(difference), 0, 255).astype(np.uint8)
    assert (rounded != truncated).any()  # the case tells them apart
    assert np.array_equal(_subtract_background(image, params), rounded)


@pytest.mark.parametrize(
    ("image", "got"),
    [
        (np.zeros((5, 4)), r"got float64 with shape \(5, 4\)"),
        (np.zeros((5, 4), dtype=np.uint16), "got uint16"),
        (np.zeros((5, 4, 3), dtype=np.uint8), r"shape \(5, 4, 3\)"),
        (np.zeros((0, 4), dtype=np.uint8), r"shape \(0, 4\)"),
        (np.zeros((5, 4), dtype=bool), "got bool"),
        ([[0, 1], [2, 3]], "got list"),
    ],
    ids=["float", "uint16", "colour", "empty", "bool", "list"],
)
def test_not_a_grey_image_raises(image, got):
    with pytest.raises(ImageError, match=r"2-D uint8 grey image.*" + got):
        preprocess(image, PreprocessParams())


@pytest.mark.parametrize(
    "view",
    [
        lambda a: a[:, ::2],
        lambda a: np.asfortranarray(a),
        lambda a: a[5:40, 10:70],
    ],
    ids=["strided", "fortran", "window"],
)
def test_a_view_gives_the_same_result_as_a_copy(view):
    image = _noisy()
    part = view(image)
    assert np.array_equal(
        preprocess(part, ALL_STEPS),
        preprocess(np.ascontiguousarray(part), ALL_STEPS),
    )


def test_a_one_pixel_image_runs_every_step():
    image = np.full((1, 1), 7, dtype=np.uint8)
    assert preprocess(image, ALL_STEPS).shape == (1, 1)
