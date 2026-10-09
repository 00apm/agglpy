"""Population metrics (``agglpy.core.metrics``).

The 1.4 synthetic summary cases (``support.synthetic.cases.SUMMARY``)
run once per implementation; the unit tests below run the core on
tables written by hand, with the values worked out in the 2.5 plan.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agglpy.core.agglomerates import find_agglomerates
from agglpy.core.metrics import (
    summary,
    summary_across_images,
    values_across_images,
)
from agglpy.core.properties import (
    agglomerate_properties,
    particle_properties,
    to_physical,
)
from agglpy.errors import MissingImagesWarning, ParamsError, TableError
from agglpy.tables import make_particles

from support.synthetic.adapters import (
    ADAPTERS,
    REAL_PIXEL_SIZE,
    expect_known_failure,
    skip_unless_supported,
)
from support.synthetic.cases import SUMMARY, Case

RTOL = 1e-12

# Diameters in the summary are compared in px; a real pixel size checks
# that they are converted and the counts and ratios are not.
SUMMARY_RUNS = [
    pytest.param(c, px, id=f"{c.name}-px{px:g}")
    for c in SUMMARY
    for px in (1.0, REAL_PIXEL_SIZE)
]


@pytest.mark.parametrize(("case", "pixel_size"), SUMMARY_RUNS)
def test_summary_cases(
    request: pytest.FixtureRequest,
    adapter: str,
    case: Case,
    pixel_size: float,
    tmp_path: Path,
):
    skip_unless_supported(adapter, "summary")
    expect_known_failure(request, adapter, "summary", case, pixel_size)
    result = ADAPTERS[adapter](case, tmp_path, pixel_size=pixel_size)
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
    ``D = 10 + j`` and ``area = 1``. An image with ``[]`` is blank.
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
            agglomerates.append((image, j, count, 10.0 + j, 1.0))
            particles += [(image, j, 2.0)] * count
    return (
        images,
        pd.DataFrame(particles, columns=["image", "agglomerate_id", "D"]),
        pd.DataFrame(
            agglomerates,
            columns=["image", "agglomerate_id", "member_count", "D", "area"],
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
        "Ra",
        "agglomerated_fraction",
        "n_ppA",
        "n_ppP",
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
    for name in ("Ra", "agglomerated_fraction", "n_ppA", "n_ppP"):
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
    # D = 8 10 20 20 20 40 60 (the composed scene of 1.4)
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


def test_metrics_narrow_the_table_in_family_order():
    scene = _scene(RA_EXAMPLE)
    out = summary(*scene, metrics=["ratios", "counts"])
    assert list(out.columns) == [
        "n_images",
        "N_primary",
        "N_aerosol",
        "N_pp1",
        "N_ppA",
        "N_aggl",
        "Ra",
        "agglomerated_fraction",
        "n_ppA",
        "n_ppP",
    ]
    assert list(summary(*scene, metrics="counts").columns)[-1] == "N_aggl"


def test_fov_area_is_needed_only_for_per_area():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    images = images.drop(columns="fov_area")
    summary(images, particles, agglomerates, metrics="counts")
    with pytest.raises(TableError, match="fov_area"):
        summary(images, particles, agglomerates)


def test_unknown_family_raises():
    with pytest.raises(ParamsError, match="families"):
        summary(*_scene(RA_EXAMPLE), metrics="sizes")


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
        found = find_agglomerates(make_particles(x, y, r))
        tables = (
            particle_properties(found).assign(image=image),
            agglomerate_properties(found).assign(image=image),
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
    # N_ppA 22 instead of 6, without a sign (review of 2.5)
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


# --- values_across_images and summary_across_images -------------------

# t(0.975, 1): the 95 % quantile factor for two images (scipy gives it).
T_975_1 = 12.706204736174698


def _images(*names: str, **columns: list) -> pd.DataFrame:
    return pd.DataFrame({"image": list(names), **columns})


def test_values_across_images_mean_std_and_interval():
    per_image = pd.DataFrame({"image": ["A", "B"], "Ra": [0.5, 0.1]})
    out = values_across_images(_images("A", "B"), per_image)
    assert list(out.columns) == [
        "metric",
        "mean",
        "std",
        "n",
        "ci_low",
        "ci_high",
    ]
    row = out.iloc[0]
    assert row["metric"] == "Ra"
    assert row["mean"] == pytest.approx(0.3, rel=RTOL)
    assert row["std"] == pytest.approx(math.sqrt(0.08), rel=RTOL)
    assert row["n"] == 2
    # t(0.975, 1) * std / sqrt(2) = 12.706... * 0.2
    half = T_975_1 * 0.2
    assert row["ci_low"] == pytest.approx(0.3 - half, rel=1e-9)
    assert row["ci_high"] == pytest.approx(0.3 + half, rel=1e-9)


def test_confidence_level_changes_the_interval():
    per_image = pd.DataFrame({"image": ["A", "B"], "v": [0.5, 0.1]})
    out = values_across_images(_images("A", "B"), per_image, confidence=0.9)
    # t(0.95, 1) = 6.313751514675043
    half = 6.313751514675043 * 0.2
    assert out["ci_high"].iloc[0] == pytest.approx(0.3 + half, rel=1e-9)


@pytest.mark.parametrize("level", [0, 1, 1.5, -0.1, "95%"])
def test_bad_confidence_raises(level):
    per_image = pd.DataFrame({"image": ["A"], "v": [1.0]})
    with pytest.raises(ParamsError, match="confidence"):
        values_across_images(_images("A"), per_image, confidence=level)


def test_one_image_gives_nan_spread_never_zero():
    per_image = pd.DataFrame({"image": ["A", "B"], "v": [0.5, 0.1]})
    out = values_across_images(_images("A", "B"), per_image, by="image")
    assert out["n"].tolist() == [1, 1]
    assert out["mean"].tolist() == [0.5, 0.1]
    assert out[["std", "ci_low", "ci_high"]].isna().all().all()


def test_nan_values_are_left_out_per_column():
    per_image = pd.DataFrame(
        {"image": ["A", "B", "C"], "N": [2, 0, 4], "Ra": [0.5, np.nan, 0.1]}
    )
    out = values_across_images(_images("A", "B", "C"), per_image)
    out = out.set_index("metric")
    assert out.loc["N", "n"] == 3
    assert out.loc["N", "mean"] == 2
    assert out.loc["Ra", "n"] == 2
    assert out.loc["Ra", "mean"] == pytest.approx(0.3, rel=RTOL)


def test_repeated_images_raise():
    # a particle table passed by mistake: one row per particle
    _, particles, _ = _scene(RA_EXAMPLE)
    with pytest.raises(TableError, match="one row per image"):
        values_across_images(_images("A", "B"), particles[["image", "D"]])


def test_missing_images_are_nan_with_a_warning():
    per_image = pd.DataFrame({"image": ["A", "C"], "v": [1.0, 3.0]})
    with pytest.warns(MissingImagesWarning, match=r"\['B'\].*0"):
        out = values_across_images(_images("A", "B", "C"), per_image)
    assert out["n"].iloc[0] == 2
    assert out["mean"].iloc[0] == 2


def test_unknown_image_in_per_image_raises():
    per_image = pd.DataFrame({"image": ["A", "X"], "v": [1.0, 3.0]})
    with pytest.raises(TableError, match=r"per_image: .*\['X'\]"):
        values_across_images(_images("A"), per_image)


def test_groups_come_from_the_images_table():
    images = _images("A", "B", "C", "D", condition=["x", "y", "x", None])
    per_image = pd.DataFrame(
        {"image": ["A", "B", "C", "D"], "v": [1.0, 5.0, 3.0, 7.0]}
    )
    out = values_across_images(images, per_image, by="condition")
    assert out["condition"].iloc[:2].tolist() == ["x", "y"]
    assert pd.isna(out["condition"].iloc[2])
    assert out["mean"].tolist() == [2, 5, 7]
    assert out["n"].tolist() == [2, 1, 1]


def test_images_table_columns_are_not_values():
    # per_image may carry the image's own columns: they are not averaged
    images = _images("A", "B", condition=["x", "x"])
    per_image = pd.DataFrame(
        {"image": ["A", "B"], "condition": ["x", "x"], "v": [1.0, 3.0]}
    )
    out = values_across_images(images, per_image, by="condition")
    assert out["metric"].tolist() == ["v"]


def test_text_value_column_raises():
    per_image = pd.DataFrame({"image": ["A"], "note": ["ok"]})
    with pytest.raises(TableError, match="'note' must be numeric"):
        values_across_images(_images("A"), per_image)


def test_keys_tell_values_of_one_image_apart():
    per_image = pd.DataFrame(
        {
            "image": ["A", "A", "B", "B"],
            "size_class": [2, 1, 2, 1],
            "fraction": [0.25, 0.75, 0.5, 0.5],
        }
    )
    out = values_across_images(_images("A", "B"), per_image, keys="size_class")
    # keys in the order they first appear; one row per key and metric
    assert out["size_class"].tolist() == [2, 1]
    assert out["metric"].tolist() == ["fraction", "fraction"]
    assert out["mean"].tolist() == [0.375, 0.625]
    assert out["n"].tolist() == [2, 2]


def test_categorical_by_and_keys_give_only_used_values():
    # pandas would warn (observed default) and add a row for the unused
    # category "y" / "c" with n = 0
    images = _images(
        "A", "B", condition=pd.Categorical(["x", "x"], ["x", "y"])
    )
    per_image = pd.DataFrame(
        {
            "image": ["A", "B"],
            "size_class": pd.Categorical(["a", "a"], ["a", "c"]),
            "v": [1.0, 3.0],
        }
    )
    scene_images, particles, agglomerates = _scene({"A": [1], "B": [2]})
    scene_images["condition"] = images["condition"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_group = values_across_images(
            images, per_image.drop(columns="size_class"), by="condition"
        )
        by_key = values_across_images(
            images[["image"]], per_image, keys="size_class"
        )
        pooled = summary(scene_images, particles, agglomerates, by="condition")
    assert pooled["N_primary"].tolist() == [3]
    assert by_group["condition"].tolist() == ["x"]
    assert by_group["n"].tolist() == [2]
    assert by_key["size_class"].tolist() == ["a"]
    assert by_key["n"].tolist() == [2]


def test_summary_across_images_uses_per_image_metrics():
    images, particles, agglomerates = _scene({**RA_EXAMPLE, "blank": []})
    out = summary_across_images(
        images, particles, agglomerates, metrics=["counts", "ratios"]
    )
    assert out["metric"].tolist() == [
        "N_primary",
        "N_aerosol",
        "N_pp1",
        "N_ppA",
        "N_aggl",
        "Ra",
        "agglomerated_fraction",
        "n_ppA",
        "n_ppP",
    ]
    out = out.set_index("metric")
    # mean of the per-image ratios: (0.5 + 0.1) / 2; the blank image
    # has Ra NaN and is left out, but counts its 0 particles
    assert out.loc["Ra", "mean"] == pytest.approx(0.3, rel=RTOL)
    assert out.loc["Ra", "n"] == 2
    assert out.loc["N_primary", "n"] == 3
    assert out.loc["N_primary", "mean"] == pytest.approx(22 / 3, rel=RTOL)


def test_summary_across_images_by_group():
    images, particles, agglomerates = _scene(RA_EXAMPLE)
    images["condition"] = ["x", "y"]
    out = summary_across_images(
        images, particles, agglomerates, by="condition", metrics="ratios"
    )
    assert list(out.columns) == [
        "condition",
        "metric",
        "mean",
        "std",
        "n",
        "ci_low",
        "ci_high",
    ]
    assert out["condition"].tolist() == ["x"] * 4 + ["y"] * 4
    ra = out[out["metric"] == "Ra"]
    assert ra["mean"].tolist() == [0.5, 0.1]


def test_summary_across_images_warns_about_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summary_across_images(*_scene({**RA_EXAMPLE, "blank": []}))
