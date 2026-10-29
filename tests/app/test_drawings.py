"""``res.draw``: one figure per image, one colour scale, made on access."""

import numpy as np
import pytest
import tifffile
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from agglpy.app import Collection, Drawings
from agglpy.app._results import Results
from agglpy.errors import ImageError, ParamsError, TableError
from agglpy.io import read_particles

from support.collection import work
from support.render import render


@pytest.fixture
def res(analysis_col: Collection) -> Results:
    return analysis_col.analyze(progress=False)


def _strip(fig: Figure) -> Axes:
    (strip,) = (a for a in fig.axes if a.get_label() == "agglpy legend")
    return strip


def _norm(fig: Figure) -> tuple[float, float]:
    """The range of the colorbar in the strip."""
    (bar,) = _strip(fig).child_axes
    low, high = bar.get_ylim()
    return float(low), float(high)


def _legend(fig: Figure) -> list[str]:
    key = _strip(fig).get_legend()
    return [t.get_text() for t in key.get_texts()]


def test_one_scale_over_the_drawn_images(res: Results) -> None:
    d = res.draw(color_by="D")
    assert isinstance(d, Drawings)
    assert _norm(d["D7-017"]) == _norm(d["D7-018"])
    assert _norm(d["D7-017"]) == pytest.approx(
        (res.agglomerates.D.min(), res.agglomerates.D.max())
    )
    res.agglomerates["kind"] = np.where(
        res.agglomerates.member_count > 1, "pair", "single"
    )
    d = res.draw(color_by="kind")
    # D7-017 has only single discs and still lists "pair"
    assert _legend(d["D7-017"]) == _legend(d["D7-018"]) == ["pair", "single"]
    assert list(d) == ["D7-017", "D7-018", "D7-019"]
    assert "D7-020" not in d and len(d) == 3
    with pytest.raises(KeyError):
        d["D7-020"]
    # a new figure on each access, none held
    assert d["D7-018"] is not d["D7-018"]


def test_px_round_trip(res: Results, analysis_col: Collection) -> None:
    # color_by="r": the colours come from m, the circles stay in px
    d = res.draw("D7-018", what="particles", color_by="r", image=False)
    assert list(d) == ["D7-018"]
    (ax, _) = d["D7-018"].axes
    (fills,) = (c for c in ax.collections if len(c.get_offsets()) == 3)
    # _widths: half the diameters (get_widths needs matplotlib 3.9)
    drawn = np.column_stack([fills.get_offsets(), fills._widths])
    stored = read_particles(
        work(analysis_col, "D7-018") / "particles_corrected.csv"
    )
    expected = stored[["x", "y", "r"]].to_numpy()
    np.testing.assert_allclose(
        np.sort(drawn, axis=0), np.sort(expected, axis=0), rtol=0, atol=1e-9
    )


def test_draw_refuses_before_rendering(res: Results) -> None:
    with pytest.raises(ParamsError, match=r"column of res.images.*merge"):
        res.draw(color_by="flow_rate")
    with pytest.raises(ParamsError, match="what"):
        res.draw(what="images", color_by="D")  # type: ignore[arg-type]
    with pytest.raises(
        TableError, match=r"no column 'Q' in res\.agglomerates"
    ):
        res.draw(color_by="Q")
    with pytest.raises(ParamsError, match="no image 'X'"):
        res.draw("X", color_by="D")
    with pytest.raises(TableError, match="select"):
        res.draw(color_by="D", select=res.particles.D > 0)
    # the shared scale's checks: labels take no vmin
    res.agglomerates["kind"] = "single"
    with pytest.raises(ParamsError, match="vmin"):
        res.draw(color_by="kind", vmin=0)
    with pytest.raises(ParamsError, match="image_alpha"):
        res.draw(color_by="D", image_alpha=2)
    with pytest.raises(ParamsError, match="reads the image file"):
        res.draw(color_by="D", image=np.zeros((160, 200), np.uint8))


def test_image_underneath_read_on_access(
    res: Results, analysis_col: Collection
) -> None:
    d = res.draw("D7-017", color_by="D")
    (ax, _) = d["D7-017"].axes
    assert len(ax.images) == 1  # the image underneath
    plain = render(d["D7-017"])[5, 5, :3].astype(float)
    dimmed = res.draw(
        "D7-017", color_by="D", background="black", image_alpha=0.4
    )
    np.testing.assert_allclose(
        render(dimmed["D7-017"])[5, 5, :3], 0.4 * plain, atol=1
    )
    images = analysis_col.folder.parent / "images"
    (images / "D7-017.tif").unlink()
    # image=False never reads the file
    for background in ("white", "#FF000033", (0, 1, 0, 0.47), None):
        fig = res.draw(
            "D7-017", color_by="D", image=False, background=background
        )["D7-017"]
        assert not fig.axes[0].images
    with pytest.raises(FileNotFoundError, match=r"D7-017\.tif"):
        d["D7-017"]
    tifffile.imwrite(images / "D7-017.tif", np.zeros((80, 100), np.uint8))
    with pytest.raises(
        ImageError, match=r"now 100 x 80 px, analysed at 200 x 160.*analyze"
    ):
        d["D7-017"]


def test_legend_false_has_no_strip(res: Results) -> None:
    fig = res.draw("D7-017", color_by="D", legend=False)["D7-017"]
    width, height = fig.get_size_inches() * fig.dpi
    assert (round(width), round(height)) == (200, 160)
    assert len(fig.axes) == 1


def test_added_column_and_select_can_be_drawn(res: Results) -> None:
    res.agglomerates["dense"] = res.agglomerates.member_count > 1
    single = res.agglomerates.member_count == 1
    d = res.draw(color_by="dense", select=single)
    assert _legend(d["D7-018"]) == ["False", "True"]
    # changing the table afterwards does not change the drawings
    res.agglomerates.drop(columns="dense", inplace=True)
    assert isinstance(d["D7-018"], Figure)
