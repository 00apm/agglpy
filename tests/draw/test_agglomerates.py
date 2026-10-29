"""agglpy.draw.agglomerates: one filled shape and outline per agglomerate."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

from agglpy import draw
from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.properties import agglomerate_properties
from agglpy.errors import TableError
from agglpy.tables import make_particles

from support.render import render, rgba

WHITE = [255, 255, 255, 255]
VIRIDIS = matplotlib.colormaps["viridis"]


def tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A pair crossing at x = 40 (agglomerate 1) and a single particle."""
    p = find_agglomerates(
        make_particles([30, 50, 50], [40, 40, 90], [12, 12, 5])
    )
    return p, agglomerate_properties(p)


def drawn(p, agg, **options) -> tuple[np.ndarray, draw.Drawing]:
    fig, ax = draw.figure((120, 100), legend=False)
    options = {"legend": False, **options}
    drawing = draw.agglomerates(ax, p, agg, **options)
    return render(fig), drawing


def test_the_overlap_is_filled_once():
    p, agg = tables()
    image, _ = drawn(p, agg, color_by="member_count", linewidth=0)
    # same colour inside one circle only and inside both
    np.testing.assert_array_equal(image[40, 25], image[40, 40])
    assert (image[40, 25] != WHITE).any()


def test_colour_is_the_agglomerate_value():
    p, agg = tables()
    image, _ = drawn(
        p, agg, color_by="member_count", cmap="viridis", alpha=1.0
    )
    np.testing.assert_array_equal(image[40, 25], rgba(VIRIDIS(1.0)))
    np.testing.assert_array_equal(image[90, 50], rgba(VIRIDIS(0.0)))


def test_image_alpha_dims_the_image():
    p, agg = tables()
    image = np.full((120, 100), 200, dtype=np.uint8)
    rendered, _ = drawn(
        p, agg, color_by="member_count", image=image, image_alpha=0.4
    )
    # 0.4 * 200 + 0.6 * 255 on the white figure
    np.testing.assert_allclose(rendered[5, 5, :3], 233, atol=1)


def test_no_inner_circle_lines():
    p, agg = tables()
    image, _ = drawn(p, agg, color_by="member_count", alpha=0.0, linewidth=4)
    # x = 38 lies on the right circle's edge, inside the left circle
    np.testing.assert_array_equal(image[40, 38], WHITE)
    # the outer outline at the left circle's far edge is drawn
    assert (image[40, 18] != WHITE).any()


def test_outline_arcs_are_one_path_per_agglomerate():
    p, agg = tables()
    _, drawing = drawn(p, agg, color_by="member_count")
    assert len(drawing.outlines.get_paths()) == 2
    assert len(drawing.fills.get_paths()) == 2


def test_members_of_unselected_agglomerates_are_others():
    p, agg = tables()
    image, drawing = drawn(
        p,
        agg,
        color_by="member_count",
        select=agg["member_count"] > 1,
        linewidth=6,
    )
    np.testing.assert_array_equal(image[90, 50], WHITE)  # not filled
    assert (image[90, 55] != WHITE).any()  # its grey outline
    assert len(drawing.fills.get_paths()) == 1


def test_a_hole_in_a_ring_is_outlined_and_not_filled():
    angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    p = find_agglomerates(
        make_particles(
            60 + 40 * np.cos(angle), 60 + 40 * np.sin(angle), np.full(24, 7.0)
        )
    )
    agg = agglomerate_properties(p)
    fig, ax = draw.figure((120, 120), legend=False)
    draw.agglomerates(
        ax,
        p,
        agg,
        color_by="member_count",
        alpha=1.0,
        linewidth=0,
        legend=False,
    )
    image = render(fig)
    np.testing.assert_array_equal(image[60, 60], WHITE)  # the hole


def test_particles_need_agglomerate_ids():
    p, agg = tables()
    with pytest.raises(TableError, match="agglomerate_id"):
        drawn(p.drop(columns="agglomerate_id"), agg, color_by="area")


def test_repeated_agglomerate_ids_raise():
    p, agg = tables()
    with pytest.raises(TableError, match="agglomerate_id"):
        drawn(p, pd.concat([agg, agg]), color_by="area")


def test_select_is_on_the_agglomerate_table():
    p, agg = tables()
    with pytest.raises(TableError, match="select"):
        drawn(p, agg, color_by="area", select=p["r"] > 5)
