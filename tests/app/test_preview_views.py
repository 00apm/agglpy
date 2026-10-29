"""The views of a tuning preview: one full-size figure each."""

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

from agglpy.app import Collection, Preview
from agglpy.core.detect import hct_edges
from agglpy.errors import ParamsError

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
