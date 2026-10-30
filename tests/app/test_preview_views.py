"""The views of a tuning preview: one full-size figure each."""

import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from agglpy.app import Collection, Preview
from agglpy.app import _preview as preview_module
from agglpy.core.detect import hct_edges
from agglpy.errors import ParamsError
from agglpy.params import HCTRange

from support.render import render

TAB10 = [
    np.round(np.asarray(matplotlib.colormaps["tab10"](i)[:3]) * 255)
    for i in range(10)
]


@pytest.fixture
def p(tuning_col: Collection) -> Preview:
    return tuning_col["T1"].preview(progress=False)


def _legend_texts(fig: Figure) -> list[str]:
    for ax in fig.axes:
        key = ax.get_legend()
        if key is not None:
            return [t.get_text() for t in key.get_texts()]
    return []


def test_view_sizes_and_stage_names(p: Preview) -> None:
    assert render(p.image()).shape[:2] == (160, 200)
    assert render(p.edges(1)).shape[:2] == (160, 200)  # no strip
    assert render(p.circles()).shape[:2] == (160, 224)  # legend strip
    # a stage is the region, drawn at its place; below it the background
    stage = render(p.stage("rolling ball"))
    assert stage.shape[:2] == (160, 200)
    held = {s.name: s.image for s in p._state.prepared.stages}
    np.testing.assert_array_equal(stage[:144, :, 0], held["rolling ball"])
    assert (stage[144:, :, :3] == 0).all()
    # the raw image, with the region outlined in yellow
    raw = render(p.image())
    image = p._state.prepared.raw
    np.testing.assert_array_equal(raw[1:140, 1:-1, 0], image[1:140, 1:-1])
    edge = raw[142:146, 100, :3].astype(int)
    assert any(r > 150 and g > 150 and b < 100 for r, g, b in edge)
    with pytest.raises(
        ParamsError,
        match="no stage 'blur': T1 ran median, rolling ball, clahe",
    ):
        p.stage("blur")
    with pytest.raises(ParamsError, match="T1 ran no preprocessing"):
        p.change_preprocess(
            median_blur=None, rolling_ball=None, clahe=None
        ).stage("median")
    with pytest.raises(ParamsError, match="range 3 does not exist"):
        p.edges(3)
    with pytest.raises(ParamsError, match="range 0 does not exist"):
        p.circles(0)


def test_edges_view_is_hct_edges(p: Preview) -> None:
    final = p._state.prepared.final
    assert p.params.hct is not None
    for k, hct in enumerate(p.params.hct.ranges, start=1):
        shown = render(p.edges(k))
        expected = hct_edges(final, hct) > 0
        assert expected.any()
        # white on black inside the region, black below it
        np.testing.assert_array_equal(shown[:144, :, 0] > 127, expected)
        assert (shown[144:, :, :3] == 0).all()
    # over the raw image: yellow edges, the image elsewhere
    over = render(p.edges(1, image="raw"))
    yy, xx = np.nonzero(hct_edges(final, p.params.hct.ranges[0]))
    assert (over[yy, xx, :3] == [255, 255, 0]).all()
    raw = p._state.prepared.raw
    off = ~(hct_edges(final, p.params.hct.ranges[0]) > 0)
    np.testing.assert_array_equal(over[:144, :, 2][off], raw[:144][off])


def test_circles_display_defaults_and_options(p: Preview) -> None:
    # the small disc at (30.3, 30.6), r 6: found by range 1
    centre = (30, 30)
    raw = p._state.prepared.raw
    shown = render(p.circles())
    # outlines only, over the raw image
    np.testing.assert_array_equal(shown[centre][:3], [raw[centre]] * 3)
    ring = shown[30, 35:40, :3].astype(float)
    # antialiased between pixels; any grey is >= 149 away from it
    assert np.abs(ring - TAB10[0]).sum(axis=1).min() < 100
    assert _legend_texts(p.circles()) == ["range 1 (4)", "range 2 (3)"]
    # on the background
    assert (render(p.circles(image=None))[centre][:3] == 0).all()
    # a filled circle: tinted, neither the image nor the background
    filled = render(p.circles(alpha=0.5))[centre][:3].astype(int)
    assert not (filled == raw[centre]).all()
    assert not (filled == 0).all()
    # over the preprocessed region
    final = p._state.prepared.final
    np.testing.assert_array_equal(
        render(p.circles(image="preprocessed"))[centre][:3],
        [final[centre]] * 3,
    )
    # one pass, in its own colour
    assert _legend_texts(p.circles(2)) == ["range 2 (3)"]
    second = render(p.circles(2))
    np.testing.assert_array_equal(second[centre][:3], [raw[centre]] * 3)
    assert np.abs(second[30, 35:40, :3] - TAB10[0]).sum(axis=1).min() > 100
    with pytest.raises(ParamsError, match="image must be"):
        p.circles(image="stage")  # type: ignore[arg-type]


def test_views_without_ranges(p: Preview) -> None:
    none = p.remove_range(1).remove_range(1)
    assert render(none.image()).shape[:2] == (160, 200)
    for view in (none.circles, lambda: none.edges(1)):
        with pytest.raises(ParamsError, match="T1 has no HCT ranges"):
            view()


# -----------------------------------------------------------------
# Comparison: two previews of T1 whose range 1 differs in canny
# -----------------------------------------------------------------

# Range 1's circles (region frame) by canny threshold: one found by
# both, one only at 200 (gone), one only at 150 (new)
SAME, GONE, NEW = (30.0, 30.0, 6.0), (100.0, 40.0, 8.0), (150.0, 100.0, 9.0)
FAKE_CIRCLES = {200.0: [SAME, GONE], 150.0: [SAME, NEW]}
# Range 1's edge pixels (row, column) by canny threshold
BOTH_PX, GONE_PX, NEW_PX = (10, 10), (20, 20), (30, 30)
FAKE_EDGES = {200.0: [BOTH_PX, GONE_PX], 150.0: [BOTH_PX, NEW_PX]}
LIME, RED = [0, 255, 0], [255, 0, 0]


@pytest.fixture
def pair(
    tuning_col: Collection, monkeypatch: pytest.MonkeyPatch
) -> tuple[Preview, Preview]:
    """``(p, q)``: ``q`` with canny 200 in range 1, ``p`` with 150.

    HCT passes and edge images are fixed by the threshold; range 2
    finds one circle far from the others.
    """

    def fake_pass(image: np.ndarray, hct: HCTRange) -> np.ndarray:
        if hct.d_min != 8:
            return np.array([[60.0, 120.0, 12.0]], np.float32)
        return np.array(FAKE_CIRCLES[hct.canny_threshold], np.float32)

    def fake_edges(image: np.ndarray, hct: HCTRange) -> np.ndarray:
        edges = np.zeros(image.shape, np.uint8)
        for row, column in FAKE_EDGES[hct.canny_threshold]:
            edges[row, column] = 255
        return edges

    monkeypatch.setattr(preview_module, "hct_pass", fake_pass)
    monkeypatch.setattr(preview_module, "hct_edges", fake_edges)
    q = tuning_col["T1"].preview(progress=False)
    return q.change_hct(range=1, canny_threshold=150), q


def _near(
    shown: np.ndarray, circle: tuple[float, float, float], colour: list[int]
) -> bool:
    """Whether the outline right of ``circle``'s centre has ``colour``."""
    x, y, r = (round(v) for v in circle)
    ring = shown[y, x + r - 2 : x + r + 3, :3].astype(float)
    return bool(np.abs(ring - colour).sum(axis=1).min() < 100)


def _panel_texts(ax: Axes) -> list[str]:
    return [t.get_text() for t in ax.texts]


def _image_axes(fig: Figure) -> list[Axes]:
    return [ax for ax in fig.axes if ax.get_label() != "agglpy legend"]


def test_circles_compare_overlay(pair: tuple[Preview, Preview]) -> None:
    p, q = pair
    fig = p.circles(1, compare=q)
    assert _legend_texts(fig) == ["new (1)", "gone (1)", "same (1)"]
    shown = render(fig)
    assert shown.shape[:2] == (160, 224)
    assert _near(shown, NEW, LIME)
    assert _near(shown, GONE, RED)
    assert _near(shown, SAME, [128, 128, 128])
    assert not _near(shown, SAME, LIME) and not _near(shown, SAME, RED)
    # over the raw image by default, as the plain view
    raw = p._state.prepared.raw
    np.testing.assert_array_equal(shown[30, 30, :3], [raw[30, 30]] * 3)
    # every range: range 2's circle is the same in both
    assert _legend_texts(p.circles(compare=q)) == [
        "new (1)",
        "gone (1)",
        "same (2)",
    ]
    # the other way round: new and gone swap
    assert _legend_texts(q.circles(1, compare=p)) == [
        "new (1)",
        "gone (1)",
        "same (1)",
    ]
    assert _near(render(q.circles(1, compare=p)), NEW, RED)


def test_edges_compare_overlay(pair: tuple[Preview, Preview]) -> None:
    p, q = pair
    fig = p.edges(1, compare=q)
    assert _legend_texts(fig) == [
        "new (1 px)",
        "gone (1 px)",
        "same (1 px)",
    ]
    shown = render(fig)
    assert shown.shape[:2] == (160, 224)
    assert list(shown[BOTH_PX][:3]) == [255, 255, 255]
    assert list(shown[NEW_PX][:3]) == LIME
    assert list(shown[GONE_PX][:3]) == RED
    assert (shown[40:, :200, :3] == 0).all()  # on black
    # over the raw image: the image between the edges
    over = render(p.edges(1, compare=q, image="raw"))
    raw = p._state.prepared.raw
    assert list(over[NEW_PX][:3]) == LIME
    np.testing.assert_array_equal(over[50, 50, :3], [raw[50, 50]] * 3)


def test_side_by_side(pair: tuple[Preview, Preview]) -> None:
    p, q = pair
    fig = p.circles(1, compare=q, side_by_side=True)
    left, right = _image_axes(fig)
    assert left.get_shared_x_axes().joined(left, right)
    assert left.get_shared_y_axes().joined(left, right)
    shown = render(fig)
    assert shown.shape[:2] == (160, 424)  # two panels + strip
    assert _legend_texts(fig) == ["new (1)", "gone (1)", "same (1)"]
    # left: q's view (gone + same), right: p's view (new + same)
    left_px, right_px = shown[:, :200], shown[:, 200:400]
    assert _near(left_px, GONE, RED) and not _near(left_px, NEW, LIME)
    assert _near(right_px, NEW, LIME) and not _near(right_px, GONE, RED)
    assert _near(left_px, SAME, [128, 128, 128])
    assert _near(right_px, SAME, [128, 128, 128])
    assert _panel_texts(left) == ["before"]
    assert _panel_texts(right) == ["ranges[1].canny_threshold 200 → 150"]
    assert _panel_texts(
        _image_axes(q.circles(1, compare=q, side_by_side=True))[1]
    ) == ["same params"]
    # edges side by side: each panel its edges, both white
    edges = render(p.edges(1, compare=q, side_by_side=True))
    assert list(edges[GONE_PX][:3]) == RED
    assert list(edges[NEW_PX[0], 200 + NEW_PX[1]][:3]) == LIME
    assert list(edges[BOTH_PX[0], 200 + BOTH_PX[1]][:3]) == [255] * 3
    assert list(edges[NEW_PX][:3]) == [0, 0, 0]
    # without compare: the image | the view on the background
    plain = p.circles(1, side_by_side=True)
    left, right = _image_axes(plain)
    assert (_panel_texts(left), _panel_texts(right)) == (["raw"], ["circles"])
    shown = render(plain)
    raw = p._state.prepared.raw
    np.testing.assert_array_equal(shown[30, 30, :3], [raw[30, 30]] * 3)
    assert (shown[30, 230, :3] == 0).all()  # circle's centre on black
    assert _near(shown[:, 200:400], NEW, TAB10[0])
    assert _legend_texts(plain) == ["range 1 (2)"]
    edges = p.edges(1, side_by_side=True, image="preprocessed")
    left, right = _image_axes(edges)
    assert (_panel_texts(left), _panel_texts(right)) == (
        ["preprocessed"],
        ["edges"],
    )
    assert render(edges).shape[:2] == (160, 400)  # no strip


def test_compare_mismatches(tuning_col: Collection) -> None:
    p = tuning_col["T1"].preview(progress=False)
    other = tuning_col["T2"].preview(progress=False)
    for view in (p.circles, lambda **kw: p.edges(1, **kw)):
        with pytest.raises(ParamsError, match="preview of T1, got one of T2"):
            view(compare=other)
        with pytest.raises(ParamsError, match="must be another Preview"):
            view(compare="T2")
    one = p.remove_range(2)
    message = (
        "range 2 does not exist in compare: T1 has 2 ranges, 1 in compare"
    )
    with pytest.raises(ParamsError, match=message):
        p.edges(2, compare=one)
    with pytest.raises(ParamsError, match=message):
        p.circles(2, compare=one)
