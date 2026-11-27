"""Population metrics (``agglpy.core.metrics``).

The synthetic summary cases (``support.synthetic.cases.SUMMARY``) run
once per implementation; the unit tests below run the core on tables
written by hand, with hand-worked values.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.metrics import _FAMILIES, _METRICS, summary
from agglpy.core.properties import (
    agglomerate_properties,
    particle_properties,
    to_physical,
)
from agglpy.core.structure import structure_properties
from agglpy.errors import ParamsError, TableError
from agglpy.tables import make_particles

from support.structure_cases import G1, G4, G5, graph_tables
from support.synthetic.adapters import (
    ADAPTERS,
    expect_known_failure,
    skip_unless_supported,
)
from support.synthetic.cases import SUMMARY, Case

RTOL = 1e-12


@pytest.mark.parametrize("case", SUMMARY, ids=lambda c: c.name)
def test_summary_cases(
    request: pytest.FixtureRequest, adapter: str, case: Case, tmp_path: Path
):
    skip_unless_supported(adapter, "summary")
    expect_known_failure(request, adapter, "summary", case)
    result = ADAPTERS[adapter](case, tmp_path)
    for name, value in case.summary.items():
        assert result.summary[name] == pytest.approx(
            value, rel=RTOL, abs=0, nan_ok=True
        ), f"{case.name}: {name}"


# ---------------------------------------------------------------------
# Unit tests of agglpy.core.metrics (one run, no adapter)
# ---------------------------------------------------------------------


def _scene(
    member_counts: dict[str, list[int]], fov_area: dict[str, float] = {}
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Images, particles and agglomerates from member counts per image.

    Every particle has ``D = 2``; agglomerate ``j`` of an image has
    ``D = 10 + j``, ``area = 1`` and ``contact_count = count - 1`` (a
    chain). An image with ``[]`` is blank.
    """
    images = pd.DataFrame(
        {
            "image": list(member_counts),
            "fov_area": [fov_area.get(i, math.nan) for i in member_counts],
        }
    )
    particles, agglomerates = [], []
    for image, counts in member_counts.items():
        for j, count in enumerate(counts):
            agglomerates.append((image, j, count, 10.0 + j, 1.0, count - 1))
            particles += [(image, j, 2.0)] * count
    return (
        images,
        pd.DataFrame(particles, columns=["image", "agglomerate_id", "D"]),
        pd.DataFrame(
            agglomerates,
            columns=[
                "image",
                "agglomerate_id",
                "member_count",
                "D",
                "area",
                "contact_count",
            ],
        ),
    )


# Image A: one doublet (Ra 1/2); image B: 20 particles, two doublets and
# 16 singles (Ra 2/20). Pooled Ra = 3/22 = 0.136; the mean of the
# per-image ratios, 0.30, would overstate it.
RA_EXAMPLE = {"A": [2], "B": [2, 2] + [1] * 16}


def test_summary_pools_counts_and_ratios():
    out = summary(*_scene(RA_EXAMPLE))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_images"] == 2
    assert (row["N_primary"], row["N_aerosol"], row["N_pp1"]) == (22, 19, 16)
    assert (row["N_ppA"], row["N_aggl"]) == (6, 3)
    assert row["Ra"] == pytest.approx(3 / 22, rel=RTOL)
    assert round(row["Ra"], 3) == 0.136
    assert row["agglomerated_fraction"] == pytest.approx(6 / 22, rel=RTOL)
    assert row["n_ppA"] == 2
    assert row["n_ppP"] == pytest.approx(22 / 19, rel=RTOL)


def test_summary_merges_and_contacts():
    out = summary(*_scene(RA_EXAMPLE))
    row = out.iloc[0]
    # N 22, N_aerosol 19 -> 3 merges; chains: 3 contacts, no loop
    assert row["N_merges"] == 3
    assert row["merge_ratio"] == pytest.approx(3 / 21, rel=RTOL)
    assert (row["N_contacts"], row["N_loops"]) == (3, 0)
    assert row["coordination_mean"] == pytest.approx(6 / 22, rel=RTOL)
    counts = out[["N_merges", "N_contacts", "N_loops"]]
    assert counts.dtypes.tolist() == [np.int64] * 3


def test_worked_example_of_seven_particles():
    # triangle, chain of three, single (see test_agglomerates)
    found, contacts = find_agglomerates(
        make_particles(
            [0, 8, 4, 100, 108, 116, 200], [0, 0, 7, 0, 0, 0, 0], [5] * 7
        )
    )
    images = pd.DataFrame({"image": ["A"], "fov_area": [math.nan]})
    row = summary(
        images,
        particle_properties(found).assign(image="A"),
        agglomerate_properties(found, contacts).assign(image="A"),
        metrics=["counts", "ratios", "contacts"],
    ).iloc[0]
    assert (row["N_primary"], row["N_aerosol"], row["N_merges"]) == (7, 3, 4)
    assert row["merge_ratio"] == pytest.approx(4 / 6, rel=RTOL)
    assert row["Ra"] == pytest.approx(2 / 7, rel=RTOL)
    # F = merge_ratio (N - 1) / N + Ra
    assert row["agglomerated_fraction"] == pytest.approx(
        row["merge_ratio"] * 6 / 7 + row["Ra"], rel=RTOL
    )
    assert (row["N_contacts"], row["N_loops"]) == (5, 1)
    assert row["coordination_mean"] == pytest.approx(10 / 7, rel=RTOL)


@pytest.mark.parametrize("counts", [[], [1]])
def test_merge_ratio_nan_below_two_particles(counts: list[int]):
    row = summary(*_scene({"A": counts})).iloc[0]
    # N = 0 would give -0.0 without the guard
    assert math.isnan(row["merge_ratio"])


def test_contacts_family_needs_contact_count():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    agglomerates = agglomerates.drop(columns="contact_count")
    summary(images, particles, agglomerates, metrics=["counts", "ratios"])
    with pytest.raises(TableError, match="contact_count"):
        summary(images, particles, agglomerates, metrics="contacts")


def test_summary_by_image():
    out = summary(*_scene(RA_EXAMPLE), by="image")
    assert out["image"].tolist() == ["A", "B"]
    assert out["Ra"].tolist() == [0.5, 0.1]
    assert out["n_images"].tolist() == [1, 1]


def test_summary_columns_in_family_order():
    out = summary(*_scene(RA_EXAMPLE), by="image")
    assert list(out.columns) == [
        "image",
        "n_images",
        "N_primary",
        "N_aerosol",
        "N_pp1",
        "N_ppA",
        "N_aggl",
        "N_merges",
        "Ra",
        "agglomerated_fraction",
        "merge_ratio",
        "n_ppA",
        "n_ppP",
        "N_contacts",
        "N_loops",
        "coordination_mean",
        "coverage",
        "N_primary_per_area",
        "N_aerosol_per_area",
        "N_aggl_per_area",
        "particle_D_mean",
        "particle_D_std",
        "particle_D10",
        "particle_D50",
        "particle_D90",
        "particle_SMD",
        "aerosol_D_mean",
        "aerosol_D_std",
        "aerosol_D10",
        "aerosol_D50",
        "aerosol_D90",
        "aerosol_member_count_std",
        "aerosol_member_count_q10",
        "aerosol_member_count_q50",
        "aerosol_member_count_q90",
    ]
    for column in ("n_images", "N_primary", "N_aerosol", "N_pp1"):
        assert out[column].dtype == np.int64


def test_blank_image_keeps_its_row():
    scene = _scene({"A": [2], "blank": []})
    out = summary(*scene, by="image").set_index("image")
    blank = out.loc["blank"]
    assert blank["n_images"] == 1
    assert (blank["N_primary"], blank["N_aerosol"], blank["N_aggl"]) == (
        0,
        0,
        0,
    )
    assert (blank["N_merges"], blank["N_contacts"], blank["N_loops"]) == (
        0,
        0,
        0,
    )
    for name in (
        "Ra",
        "agglomerated_fraction",
        "merge_ratio",
        "n_ppA",
        "n_ppP",
        "coordination_mean",
    ):
        assert math.isnan(blank[name]), name
    for name in ("particle_D_mean", "particle_SMD", "aerosol_D50"):
        assert math.isnan(blank[name]), name
    # pooled: the blank image counts as an image, adds no particles
    pooled = summary(*scene).iloc[0]
    assert (pooled["n_images"], pooled["N_primary"]) == (2, 2)


def test_no_images_at_all_gives_one_empty_row():
    out = summary(*_scene({}))
    assert out["n_images"].tolist() == [0]
    assert out["N_primary"].tolist() == [0]
    assert math.isnan(out["Ra"].iloc[0])
    assert len(summary(*_scene({}), by="image")) == 0


def test_singles_only():
    row = summary(*_scene({"A": [1, 1, 1]})).iloc[0]
    assert (row["Ra"], row["agglomerated_fraction"], row["n_ppP"]) == (0, 0, 1)
    assert math.isnan(row["n_ppA"])  # 0 particles / 0 agglomerates


def test_particle_size_descriptors():
    # D = 8 10 20 20 20 40 60 (the composed synthetic scene)
    d = [20, 40, 60, 20, 8, 20, 10]
    images = pd.DataFrame({"image": ["A"], "fov_area": [math.nan]})
    particles = pd.DataFrame({"image": "A", "D": d})
    agglomerates = pd.DataFrame(
        {"image": ["A"], "agglomerate_id": [0], "member_count": [7], "D": 1.0}
    )
    row = summary(images, particles, agglomerates, metrics="particle_size")
    row = row.iloc[0]
    assert row["particle_D_mean"] == pytest.approx(178 / 7, rel=RTOL)
    assert row["particle_D_std"] == pytest.approx(
        math.sqrt(14264 / 42), rel=RTOL
    )
    assert row["particle_D10"] == pytest.approx(9.2, rel=RTOL)
    assert row["particle_D50"] == 20
    assert row["particle_D90"] == pytest.approx(48, rel=RTOL)
    assert row["particle_SMD"] == pytest.approx(305512 / 6564, rel=RTOL)


def test_aerosol_size_and_member_count_over_all_aerosol_particles():
    # member counts 3 2 1 1 (singles included); D = 10 11 12 13
    row = summary(*_scene({"A": [3, 2, 1, 1]})).iloc[0]
    assert row["aerosol_D_mean"] == 11.5
    assert row["aerosol_D_std"] == pytest.approx(math.sqrt(5 / 3), rel=RTOL)
    assert row["aerosol_D10"] == pytest.approx(10.3, rel=RTOL)
    assert row["aerosol_D50"] == 11.5
    assert row["aerosol_D90"] == pytest.approx(12.7, rel=RTOL)
    # sorted 1 1 2 3: mean 1.75 (= n_ppP); positions 0.3, 1.5, 2.7
    assert row["n_ppP"] == 1.75
    assert row["aerosol_member_count_std"] == pytest.approx(
        math.sqrt(11 / 12), rel=RTOL
    )
    assert row["aerosol_member_count_q10"] == 1
    assert row["aerosol_member_count_q50"] == 1.5
    assert row["aerosol_member_count_q90"] == pytest.approx(2.7, rel=RTOL)


def test_per_area_metrics_divide_sums():
    # areas: A 1 agglomerate, B 18 (each area 1); FoV 100 and 300
    scene = _scene(RA_EXAMPLE, fov_area={"A": 100.0, "B": 300.0})
    row = summary(*scene, metrics="per_area").iloc[0]
    assert row["coverage"] == pytest.approx(19 / 400, rel=RTOL)
    assert row["N_primary_per_area"] == pytest.approx(22 / 400, rel=RTOL)
    assert row["N_aerosol_per_area"] == pytest.approx(19 / 400, rel=RTOL)
    assert row["N_aggl_per_area"] == pytest.approx(3 / 400, rel=RTOL)


def test_one_image_without_fov_area_makes_its_group_nan():
    scene = _scene(RA_EXAMPLE, fov_area={"A": 100.0})
    pooled = summary(*scene, metrics="per_area").iloc[0]
    # not 1 / 100 from image A alone
    assert math.isnan(pooled["coverage"])
    assert math.isnan(pooled["N_primary_per_area"])
    per_image = summary(*scene, by="image", metrics="per_area")
    assert per_image["coverage"].iloc[0] == 0.01
    assert math.isnan(per_image["coverage"].iloc[1])


def test_column_names_in_the_given_order():
    scene = _scene(RA_EXAMPLE)
    full = summary(*scene)
    out = summary(*scene, metrics=["particle_D50", "counts", "N_primary"])
    assert list(out.columns) == [
        "n_images",
        "particle_D50",
        "N_primary",
        "N_aerosol",
        "N_pp1",
        "N_ppA",
        "N_aggl",
        "N_merges",
    ]
    pd.testing.assert_frame_equal(out, full[out.columns])
    assert list(summary(*scene, metrics=["ratios", "counts"]).columns) == [
        "n_images",
        *_FAMILIES["ratios"],
        *_FAMILIES["counts"],
    ]
    # one string is one name, not letters
    assert list(summary(*scene, metrics="N_primary").columns) == [
        "n_images",
        "N_primary",
    ]
    assert list(summary(*scene, metrics=[]).columns) == ["n_images"]


def test_a_column_needs_only_its_family():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    images = images.drop(columns="fov_area")
    out = summary(images, particles, agglomerates, metrics="merge_ratio")
    assert list(out.columns) == ["n_images", "merge_ratio"]  # counts hidden
    with pytest.raises(TableError, match="fov_area"):
        summary(
            images, particles, agglomerates, metrics=["N_primary", "coverage"]
        )
    with pytest.raises(TableError, match="segment_type"):
        summary(images, particles, agglomerates, metrics="N_cores")


def test_by_clashes_only_with_shown_columns():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    images["N_aerosol"] = ["x", "y"]
    out = summary(
        images, particles, agglomerates, by="N_aerosol", metrics="merge_ratio"
    )
    assert list(out.columns) == ["N_aerosol", "n_images", "merge_ratio"]


def test_family_and_column_names_never_clash():
    assert not set(_FAMILIES) & set(_METRICS)


def test_register_dimensions_match_scaling():
    # Every length scaled by 2: each column scales by 2 ** dimension.
    scene = _scene(RA_EXAMPLE, fov_area={"A": 100.0, "B": 100.0})
    # unequal diameters: a std of 0 would scale by any power
    scene[1]["D"] = np.linspace(1.0, 3.0, len(scene[1]))
    images, particles, agglomerates = (t.copy() for t in scene)
    images["fov_area"] *= 4
    particles["D"] *= 2
    agglomerates["D"] *= 2
    agglomerates["area"] *= 4
    base = summary(*scene).iloc[0]
    scaled = summary(images, particles, agglomerates).iloc[0]
    # every column with a unit is checked, on a value that can show it
    with_unit = {n for n, m in _METRICS.items() if m.dimension}
    assert with_unit <= set(base.index)
    assert all(np.isfinite(base[n]) and base[n] != 0 for n in with_unit)
    for name in base.index.drop("n_images"):
        factor = 2.0 ** _METRICS[name].dimension
        expected = base[name] * factor
        assert scaled[name] == pytest.approx(expected, nan_ok=True), name


def test_fov_area_is_needed_only_for_per_area():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    images = images.drop(columns="fov_area")
    summary(images, particles, agglomerates, metrics="counts")
    with pytest.raises(TableError, match="fov_area"):
        summary(images, particles, agglomerates)


def test_unknown_metric_raises():
    for bad in ("sizes", "n_images"):
        with pytest.raises(ParamsError, match=rf"\['{bad}'\].*metric_names"):
            summary(*_scene(RA_EXAMPLE), metrics=["counts", bad])


def test_empty_by_cell_is_its_own_group():
    images, particles, agglomerates = _scene({"A": [2], "B": [1], "C": [3]})
    images["condition"] = ["x", None, "x"]
    out = summary(images, particles, agglomerates, by="condition")
    assert out["condition"].iloc[0] == "x"
    assert pd.isna(out["condition"].iloc[1])  # sorted last
    assert out["n_images"].tolist() == [2, 1]
    assert out["N_primary"].tolist() == [5, 1]


def test_by_two_columns_gives_existing_combinations():
    images, particles, agglomerates = _scene({"A": [2], "B": [1], "C": [3]})
    images["day"] = [1, 2, 1]
    images["sample"] = ["s", "s", "t"]
    out = summary(images, particles, agglomerates, by=["day", "sample"])
    assert out[["day", "sample"]].values.tolist() == [
        [1, "s"],
        [1, "t"],
        [2, "s"],
    ]
    assert out["N_primary"].tolist() == [2, 3, 1]


def test_result_does_not_depend_on_row_order_or_index():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    expected = summary(images, particles, agglomerates, by="image")
    shuffled = [
        t.sample(frac=1, random_state=1).set_axis(
            np.arange(len(t)) * 7 + 3, axis=0
        )
        for t in (particles, agglomerates)
    ]
    pd.testing.assert_frame_equal(
        summary(images, *shuffled, by="image"), expected
    )


def test_summary_is_unit_agnostic():
    # Two images of real circles, in px and converted with p = 0.5:
    # lengths scale by p, per-area values by 1 / p², the rest not.
    p = 0.5
    px, physical = [], []
    for image, (x, y, r) in {
        "A": ([0, 14, 2, 100], [0, 0, 0, 0], [10, 4, 2, 3]),
        "B": ([0, 30], [0, 0], [10, 20]),
    }.items():
        found, contacts = find_agglomerates(make_particles(x, y, r))
        tables = (
            particle_properties(found).assign(image=image),
            agglomerate_properties(found, contacts).assign(image=image),
        )
        px.append(tables)
        physical.append(tuple(to_physical(t, p) for t in tables))
    fov = pd.DataFrame({"image": ["A", "B"], "fov_area": [1e4, 2e4]})
    before = summary(
        fov, *(pd.concat(t) for t in zip(*px, strict=True)), by="image"
    )
    after = summary(
        fov.assign(fov_area=fov["fov_area"] * p**2),
        *(pd.concat(t) for t in zip(*physical, strict=True)),
        by="image",
    )
    for column in before.columns[1:]:
        if column.startswith(("particle_", "aerosol_D")):
            power = 1
        elif column.endswith("_per_area"):
            power = -2
        else:
            power = 0
        expected = before[column] * p**power
        np.testing.assert_allclose(
            after[column], expected, rtol=1e-12, err_msg=column
        )


@pytest.mark.parametrize(
    ("images", "message"),
    [
        (pd.DataFrame({"name": ["A"]}), "'image'"),
        (pd.DataFrame({"image": ["A", "A"]}), "repeated"),
        (pd.DataFrame({"image": ["A", None]}), "gaps"),
    ],
)
def test_bad_images_table_raises(images, message):
    _, particles, agglomerates = _scene({"A": [1]})
    with pytest.raises(TableError, match=message):
        summary(images, particles, agglomerates, metrics="counts")


@pytest.mark.parametrize("area", [0.0, -1.0, math.inf])
def test_fov_area_must_be_positive_where_given(area):
    scene = _scene({"A": [1], "B": [1]}, fov_area={"A": area, "B": 1.0})
    with pytest.raises(TableError, match="fov_area"):
        summary(*scene)


def test_rows_of_an_unknown_image_raise():
    # e.g. a typo, or an images table filtered without its rows
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    with pytest.raises(TableError, match=r"particles: .*\['B'\]"):
        summary(images.iloc[:1], particles, agglomerates.iloc[:1])


def test_agglomerates_filtered_alone_raise():
    # aggl[aggl.member_count >= 2] keeps the particles of the singles:
    # N_ppA 22 instead of 6, without a sign
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    multi = agglomerates[agglomerates["member_count"] >= 2]
    with pytest.raises(TableError, match=r"\['B'\].*select_agglomerates"):
        summary(images, particles, multi)


def test_particles_filtered_alone_raise():
    # a particle subset belongs in distribution, not in summary: here
    # N_ppA would be negative
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    with pytest.raises(TableError, match=r"\['A'\]"):
        summary(images, particles.iloc[2:], agglomerates, metrics="counts")


def test_member_count_gap_raises():
    images, particles, agglomerates = _scene({"A": [2, 1]})
    agglomerates["member_count"] = pd.array([2, None], dtype="Int64")
    with pytest.raises(TableError, match="member_count"):
        summary(images, particles, agglomerates)


def test_missing_columns_raise():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    with pytest.raises(TableError, match=r"agglomerates: .*member_count"):
        summary(images, particles, agglomerates.drop(columns="member_count"))
    with pytest.raises(TableError, match=r"particles: .*'D'"):
        summary(images, particles.drop(columns="D"), agglomerates)


def test_unknown_by_column_raises():
    with pytest.raises(TableError, match=r"by: .*condition"):
        summary(*_scene(RA_EXAMPLE), by="condition")


def test_by_column_named_like_a_metric_raises():
    # the group label "Ra" next to the metric Ra: two columns "Ra"
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    images["Ra"] = ["smooth", "rough"]
    with pytest.raises(TableError, match=r"by: .*'Ra'"):
        summary(images, particles, agglomerates, by="Ra")
    # harmless while nobody groups by it
    assert "Ra" in summary(images, particles, agglomerates).columns


def test_repeated_by_column_raises():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    images["g"] = ["x", "y"]
    with pytest.raises(ParamsError, match=r"by: .*'g'"):
        summary(images, particles, agglomerates, by=["g", "g"])


def test_categorical_by_gives_only_used_groups():
    # pandas would warn (observed default) and add a row for the unused
    # category "y"
    images, particles, agglomerates = _scene({"A": [1], "B": [2]})
    images["condition"] = pd.Categorical(["x", "x"], ["x", "y"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pooled = summary(images, particles, agglomerates, by="condition")
    assert pooled["N_primary"].tolist() == [3]


STRUCTURE = [
    "N_cores",
    "N_small_cores",
    "N_links",
    "N_dangling_ends",
    "core_fraction",
    "small_core_fraction",
    "link_fraction",
    "loop_fraction",
    "dangling_end_fraction",
    "no_core_fraction",
]


def _structured(image, graphs):
    """One image's tables from hand graphs, one agglomerate each."""
    edges, r, labels = [], [], []
    for label, (graph_edges, graph_r) in enumerate(graphs):
        offset = len(r)
        edges += [(a + offset, b + offset) for a, b in graph_edges]
        r += list(graph_r)
        labels += [label] * len(graph_r)
    p, a = structure_properties(*graph_tables(edges, r, labels))
    a["member_count"] = np.bincount(p["agglomerate_id"])
    return p.assign(image=image), a.assign(image=image)


def _structure_scene():
    """Image a: G4, G5, a single (29); b: G1, a 6-ring (11); blank."""
    ring = ([(i, (i + 1) % 6) for i in range(6)], [1.0] * 6)
    a = _structured("a", [G4, G5, ([], [1.0])])
    b = _structured("b", [G1, ring])
    images = pd.DataFrame({"image": ["a", "b", "blank"]})
    particles = pd.concat([a[0], b[0]], ignore_index=True)
    agglomerates = pd.concat([a[1], b[1]], ignore_index=True)
    return images, particles, agglomerates


def test_structure_family():
    out = summary(*_structure_scene(), by="image", metrics="structure")
    a, b = out.iloc[0], out.iloc[1]
    assert a[STRUCTURE[:4]].tolist() == [3, 0, 1, 2]
    assert a["core_fraction"] == 21 / 29
    assert a["dangling_end_fraction"] == 3 / 29
    assert a["link_fraction"] == 2 / 29
    assert a["loop_fraction"] == 2 / 29
    assert a["no_core_fraction"] == 1 / 29
    assert b[STRUCTURE[:4]].tolist() == [0, 1, 0, 0]
    assert b["small_core_fraction"] == 6 / 11
    assert b["no_core_fraction"] == 5 / 11
    fractions = [c for c in STRUCTURE if c.endswith("_fraction")]
    assert np.allclose(out[fractions].iloc[:2].sum(axis=1), 1)
    for column in STRUCTURE[:4]:
        assert out[column].dtype == np.int64


def test_structure_fractions_nan_without_particles():
    out = summary(*_structure_scene(), by="image", metrics="structure")
    blank = out.set_index("image").loc["blank"]
    assert blank[STRUCTURE[:4]].tolist() == [0, 0, 0, 0]
    assert blank[STRUCTURE[4:]].isna().all()


def _with_structure_columns(particles, agglomerates):
    particles = particles.assign(segment_type="no core")
    agglomerates = agglomerates.assign(
        core_count=0,
        small_core_member_count=0,
        link_count=0,
        dangling_end_count=0,
    )
    return particles, agglomerates


def test_default_summary_includes_structure_only_with_its_columns():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    plain = summary(images, particles, agglomerates)
    assert "N_cores" not in plain.columns
    out = summary(images, *_with_structure_columns(particles, agglomerates))
    columns = list(out.columns)
    at = columns.index("coordination_mean") + 1
    assert columns[at : at + len(STRUCTURE)] == STRUCTURE
    assert columns[at + len(STRUCTURE)] == "coverage"


@pytest.mark.parametrize(
    ("drop", "missing"),
    [("particles", "segment_type"), ("agglomerates", "link_count")],
)
def test_structure_by_name_without_columns(drop, missing):
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    particles, agglomerates = _with_structure_columns(particles, agglomerates)
    if drop == "particles":
        particles = particles.drop(columns=missing)
    else:
        agglomerates = agglomerates.drop(columns=missing)
    with pytest.raises(TableError, match=missing):
        summary(images, particles, agglomerates, metrics="structure")
    # the default leaves the family out instead
    assert "N_cores" not in summary(images, particles, agglomerates).columns
