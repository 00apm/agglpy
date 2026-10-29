"""HCT detection (``agglpy.core.detect``) on drawn images.

The contract test draws discs and checks that the detector finds them
where they were drawn; the other tests replace ``cv2.HoughCircles`` with
a spy to pin how the params reach OpenCV and how its output becomes a
particle table.
"""

import os
import warnings

import cv2
import numpy as np
import pandas as pd
import pytest

from agglpy.core.detect import detect_hct, hct_edges
from agglpy.errors import (
    DuplicateParticlesWarning,
    ImageError,
    NoParticlesWarning,
)
from agglpy.params import HCTParams, HCTRange

# cv2.circle's sub-pixel mode: coordinates and radius in 1/16 px
SHIFT = 4


def _draw_disc(image: np.ndarray, disc, value: int) -> None:
    x, y, r = disc
    cv2.circle(
        image,
        (round(x * 2**SHIFT), round(y * 2**SHIFT)),
        round(r * 2**SHIFT),
        value,
        thickness=-1,
        lineType=cv2.LINE_AA,
        shift=SHIFT,
    )


def _draw(height: int, width: int, discs) -> np.ndarray:
    """Bright anti-aliased discs (x, y, r in px) on a dark image."""
    image = np.full((height, width), 20, dtype=np.uint8)
    for disc in discs:
        _draw_disc(image, disc, 200)
    return image


def _drawn_radius(height: int, width: int, disc) -> float:
    """Radius of the circle with the drawn disc's area.

    Anti-aliased drawing makes a disc about 0.5 px larger than the
    radius asked for; the detector should find what was drawn.
    """
    alone = np.zeros((height, width), dtype=np.uint8)
    _draw_disc(alone, disc, 255)
    return float(np.sqrt(alone.sum() / 255 / np.pi))


def _random_discs(height: int, width: int, n: int, seed: int) -> list:
    """n separate discs, r 8-20 px, at random sub-pixel centres."""
    rng = np.random.default_rng(seed)
    discs: list[tuple[float, float, float]] = []
    while len(discs) < n:
        r = rng.uniform(8, 20)
        x = rng.uniform(r + 3, width - r - 3)
        y = rng.uniform(r + 3, height - r - 3)
        if all(np.hypot(x - a, y - b) > r + c + 8 for a, b, c in discs):
            discs.append((x, y, r))
    return discs


def test_contract_drawn_discs_come_back_where_they_were_drawn():
    # Non-square, so swapped x / y can't match. Centres at sub-pixel
    # positions: HCT reports centres on a k + 0.5 grid, so a disc drawn
    # at an integer centre always comes back 0.5 px off.
    height, width = 450, 600
    discs = _random_discs(height, width, n=40, seed=1)
    found = detect_hct(
        _draw(height, width, discs), HCTParams(ranges=(HCTRange(12, 44),))
    )

    assert len(found) == len(discs)
    errors = []
    for disc in discs:
        x, y, _ = disc
        distance = np.hypot(found["x"] - x, found["y"] - y)
        match = found.loc[distance.idxmin()]
        errors.append(
            (
                match["x"] - x,
                match["y"] - y,
                match["r"] - _drawn_radius(height, width, disc),
            )
        )
    dx, dy, dr = np.array(errors).T
    # each disc within 1 px (worst seen: centre 0.8, radius 0.63)
    assert np.hypot(dx, dy).max() <= 1.0
    assert np.abs(dr).max() <= 1.0
    # no systematic shift: a half-pixel convention error shows here
    for signed in (dx, dy, dr):
        assert abs(signed.mean()) <= 0.25


class _Spy:
    """Stands in for cv2.HoughCircles: records calls, returns output."""

    def __init__(self, outputs: list) -> None:
        self.calls: list[dict] = []
        self.outputs = list(outputs)

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def _circles(*rows: tuple[float, float, float]) -> np.ndarray:
    """OpenCV's output shape: (1, n, 3) float32."""
    return np.array([rows], dtype=np.float32)


def test_params_reach_opencv_by_name(monkeypatch):
    spy = _Spy([_circles((10.5, 10.5, 5.0)), None])
    monkeypatch.setattr(cv2, "HoughCircles", spy)
    params = HCTParams(
        ranges=(
            HCTRange(3, 61, canny_threshold=220, accumulator_threshold=13),
            HCTRange(
                61,
                240,
                min_dist_ratio=0.5,
                canny_threshold=250,
                accumulator_threshold=24,
            ),
        )
    )
    detect_hct(np.zeros((40, 50), dtype=np.uint8), params)

    assert spy.calls == [
        {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1,
            "minDist": 0.4 * 61 / 2,  # 0.4 x the largest radius
            "param1": 220.0,
            "param2": 13.0,
            # odd diameters round outward: the search covers 3-61 px
            "minRadius": 1,  # floor(3 / 2)
            "maxRadius": 31,  # ceil(61 / 2)
        },
        {
            "method": cv2.HOUGH_GRADIENT,
            "dp": 1,
            "minDist": 60.0,  # 0.5 x 240 / 2
            "param1": 250.0,
            "param2": 24.0,
            "minRadius": 30,  # floor(61 / 2)
            "maxRadius": 120,
        },
    ]


def test_rows_go_range_by_range_in_opencv_order(monkeypatch):
    spy = _Spy(
        [
            _circles((30.5, 40.5, 6.0), (10.5, 20.5, 5.0)),
            _circles((100.5, 90.5, 50.0)),
        ]
    )
    monkeypatch.setattr(cv2, "HoughCircles", spy)
    params = HCTParams(ranges=(HCTRange(3, 20), HCTRange(20, 120)))
    found = detect_hct(np.zeros((200, 200), dtype=np.uint8), params)

    expected = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x": [30.5, 10.5, 100.5],
            "y": [40.5, 20.5, 90.5],
            "r": [6.0, 5.0, 50.0],
        }
    )
    pd.testing.assert_frame_equal(found, expected)


def test_values_are_the_shortest_decimals(monkeypatch):
    # float32 29.4 is 29.399999618530273 as float64
    spy = _Spy([_circles((8.5, 1248.5, 29.4), (10.5, 11.5, 21.8))])
    monkeypatch.setattr(cv2, "HoughCircles", spy)
    found = detect_hct(
        np.zeros((10, 10), dtype=np.uint8),
        HCTParams(ranges=(HCTRange(3, 60),)),
    )
    assert repr(found["r"].tolist()) == "[29.4, 21.8]"
    assert found["r"].dtype == np.float64


def test_nothing_found_is_an_empty_table_and_one_warning():
    image = np.zeros((20, 30), dtype=np.uint8)
    params = HCTParams(ranges=(HCTRange(4, 10), HCTRange(10, 20)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = detect_hct(image, params)

    assert [w.category for w in caught] == [NoParticlesWarning]
    assert str(caught[0].message) == (
        "no circles found in the 30 x 20 px image (pixel intensities 0-0, "
        "of possible 0-255); check preprocessing and HCT params"
    )
    # the warning points at the caller, not at agglpy
    assert os.path.normcase(caught[0].filename) == os.path.normcase(__file__)
    assert list(found.columns) == ["id", "x", "y", "r"]
    assert len(found) == 0
    assert found["id"].dtype == np.int64
    assert found["r"].dtype == np.float64


def test_the_warning_states_the_intensity_range():
    image = np.full((20, 30), 7, dtype=np.uint8)
    image[0, 0] = 9
    with pytest.warns(NoParticlesWarning, match="pixel intensities 7-9"):
        detect_hct(image, HCTParams(ranges=(HCTRange(4, 10),)))


def test_one_disc_found_by_two_ranges_is_kept_twice_and_warned():
    image = _draw(80, 100, [(50.5, 40.25, 15)])
    params = HCTParams(ranges=(HCTRange(10, 40), HCTRange(20, 50)))
    with pytest.warns(DuplicateParticlesWarning, match=r"\(1, 2\)") as rec:
        found = detect_hct(image, params)
    assert found["id"].tolist() == [1, 2]
    assert os.path.normcase(rec[0].filename) == os.path.normcase(__file__)


def test_input_is_not_modified():
    image = _draw(120, 160, _random_discs(120, 160, n=4, seed=0))
    before = image.copy()
    detect_hct(image, HCTParams(ranges=(HCTRange(12, 44),)))
    assert np.array_equal(image, before)


def _window(image: np.ndarray) -> np.ndarray:
    # the image as a window of a larger one, like the region cut
    larger = np.zeros((image.shape[0] + 30, image.shape[1] + 40), np.uint8)
    larger[10:-20, 15:-25] = image
    return larger[10:-20, 15:-25]


@pytest.mark.parametrize(
    "view",
    [
        lambda a: np.repeat(a, 2, axis=1)[:, ::2],
        np.asfortranarray,
        _window,
    ],
    ids=["strided", "fortran", "window"],
)
def test_a_view_gives_the_same_table_as_a_copy(view):
    # detect_hct hands its input to OpenCV without copying it, so the
    # region cut (a view) must give what a contiguous copy gives
    image = _draw(120, 160, _random_discs(120, 160, n=4, seed=0))
    part = view(image)
    assert np.array_equal(part, image)
    assert not part.flags.c_contiguous
    params = HCTParams(ranges=(HCTRange(12, 44),))
    expected = detect_hct(image, params)
    assert len(expected) == 4
    pd.testing.assert_frame_equal(detect_hct(part, params), expected)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((20, 30)),
        np.zeros((20, 30, 3), dtype=np.uint8),
        np.zeros((0, 30), dtype=np.uint8),
    ],
    ids=["float", "colour", "empty"],
)
def test_not_a_grey_image_raises(image):
    with pytest.raises(ImageError, match="2-D uint8 grey image"):
        detect_hct(image, HCTParams(ranges=(HCTRange(4, 10),)))


def _ring_image() -> tuple[np.ndarray, int]:
    # one disc and the largest |dx| + |dy| of HoughCircles' Sobel step
    image = np.full((80, 80), 20, np.uint8)
    cv2.circle(image, (40, 40), 15, 60, -1, cv2.LINE_AA)
    dx, dy = (
        cv2.Sobel(
            image, cv2.CV_16S, *order, ksize=3, borderType=cv2.BORDER_REPLICATE
        ).astype(int)
        for order in ((1, 0), (0, 1))
    )
    return image, int((np.abs(dx) + np.abs(dy)).max())


@pytest.mark.parametrize(
    "below, found", [(1, True), (0.6, True), (0.4, False), (0, False)]
)
def test_hct_edges_are_what_houghcircles_sees(below, found):
    # OpenCV rounds the threshold half to even: top - 0.4 rounds to top,
    # which no gradient exceeds, so no edge and no circle
    image, top = _ring_image()
    assert top == 240
    hct = HCTRange(
        20, 40, canny_threshold=top - below, accumulator_threshold=5
    )
    edges = hct_edges(image, hct)
    assert edges.dtype == np.uint8
    assert edges.shape == image.shape
    assert set(np.unique(edges)) <= {0, 255}
    assert bool(edges.any()) is found
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoParticlesWarning)
        circles = detect_hct(image, HCTParams(ranges=(hct,)))
    assert (len(circles) > 0) is found
    if found:
        # the outline: every edge pixel within 2 px of r = 15
        yy, xx = np.nonzero(edges)
        assert np.all(np.abs(np.hypot(xx - 40, yy - 40) - 15) <= 2)


def test_hct_edges_input():
    image, _ = _ring_image()
    before = image.copy()
    hct_edges(_window(image), HCTRange(20, 40))
    assert np.array_equal(image, before)
    for bad in (image.astype(float), np.dstack([image] * 3)):
        with pytest.raises(ImageError, match="2-D uint8 grey image"):
            hct_edges(bad, HCTRange(20, 40))
