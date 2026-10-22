"""Per-image params: merge rules and conversion from yml mappings."""

import copy

import numpy as np
import pytest

from agglpy.app._config import (
    NEUTRAL,
    ImageParams,
    Problem,
    merge,
    plain,
    to_params,
)
from agglpy.errors import ParamsError
from agglpy.params import (
    ClaheParams,
    HCTParams,
    HCTRange,
    PreprocessParams,
    RegionParams,
    RollingBallParams,
)

RB_DEFAULT = {
    "preprocess": {"rolling_ball": {"radius": 50, "light_background": True}}
}

# ---------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------


def test_merge_absent_key_uses_default() -> None:
    assert merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_merge_mappings_recursively() -> None:
    o = {"preprocess": {"rolling_ball": {"radius": 30}}}
    assert merge(RB_DEFAULT, o) == {
        "preprocess": {
            "rolling_ball": {"radius": 30, "light_background": True}
        }
    }


def test_merge_null_turns_step_off() -> None:
    merged = merge(RB_DEFAULT, {"preprocess": {"rolling_ball": None}})
    assert merged["preprocess"]["rolling_ball"] is None


def test_merge_mapping_over_null() -> None:
    assert merge({"rb": None}, {"rb": {"radius": 30}}) == {
        "rb": {"radius": 30}
    }


def test_merge_list_replaces() -> None:
    d = {
        "hct": {
            "ranges": [{"d_min": 8, "d_max": 30}, {"d_min": 20, "d_max": 60}]
        }
    }
    o = {"hct": {"ranges": [{"d_min": 5, "d_max": 9}]}}
    assert merge(d, o)["hct"]["ranges"] == [{"d_min": 5, "d_max": 9}]


def test_merge_does_not_modify_inputs() -> None:
    d = copy.deepcopy(RB_DEFAULT)
    o = {"preprocess": {"rolling_ball": {"radius": 30}}, "x": [1]}
    o_before = copy.deepcopy(o)
    merged = merge(d, o)
    merged["preprocess"]["rolling_ball"]["radius"] = 1
    merged["x"].append(2)
    assert d == RB_DEFAULT
    assert o == o_before


# ---------------------------------------------------------------------
# to_params
# ---------------------------------------------------------------------


def test_empty_mapping_gives_neutral() -> None:
    assert to_params({}) == (NEUTRAL, [])
    assert NEUTRAL == ImageParams(RegionParams(), PreprocessParams(), None)


def test_full_conversion() -> None:
    params, problems = to_params(
        {
            "region": {"bottom_fraction": 0.1},
            "preprocess": {
                "median_blur": 3,
                "rolling_ball": {"radius": 50, "light_background": True},
                "clahe": {"clip_limit": 2.0, "tile_grid": [8, 8]},
            },
            "hct": {
                "ranges": [
                    {"d_min": 8, "d_max": 30},
                    {"d_min": 20, "d_max": 60, "accumulator_threshold": 12},
                ]
            },
        }
    )
    assert problems == []
    assert params == ImageParams(
        RegionParams(0.1),
        PreprocessParams(
            3,
            RollingBallParams(50.0, light_background=True),
            ClaheParams(2.0, (8, 8)),
        ),
        HCTParams(
            (
                HCTRange(8, 30),
                HCTRange(20, 60, accumulator_threshold=12.0),
            )
        ),
    )


@pytest.mark.parametrize("hct", [{"ranges": []}, {}])
def test_empty_ranges_means_hct_not_set(hct: dict) -> None:
    params, problems = to_params({"hct": hct})
    assert problems == []
    assert params is not None
    assert params.hct is None


def test_hct_null_is_a_problem() -> None:
    params, problems = to_params({"hct": None})
    assert params is None
    assert problems == [
        Problem(
            "hct",
            "hct can't be null; write `ranges: []` to leave detection unset",
        )
    ]


def test_unknown_key_lists_valid_keys() -> None:
    _, p = to_params({"preprocess": {"median_blr": 3}})
    assert p == [
        Problem(
            "preprocess.median_blr",
            "unknown key (valid: median_blur, rolling_ball, clahe)",
        )
    ]


def test_unknown_section() -> None:
    _, p = to_params({"detect": {}})
    assert p == [
        Problem("detect", "unknown key (valid: region, preprocess, hct)")
    ]


def test_value_rule_problem_has_its_path() -> None:
    _, p = to_params({"hct": {"ranges": [{"d_min": 40, "d_max": 30}]}})
    assert p == [
        Problem(
            "hct.ranges[0]", "need 1 <= d_min < d_max, got d_min=40, d_max=30"
        )
    ]


def test_all_problems_are_collected() -> None:
    params, p = to_params(
        {
            "region": {"bottom_frac": 0.1},
            "preprocess": {"median_blur": 4, "rolling_ball": {"radius": -1}},
            "hct": {
                "ranges": [
                    {"d_min": 8, "d_max": 30},
                    {"d_min": "x", "d_max": 30},
                ]
            },
        }
    )
    assert params is None
    # A dataclass's own problem comes before its children's.
    assert [q.path for q in p] == [
        "region.bottom_frac",
        "preprocess",
        "preprocess.rolling_ball",
        "hct.ranges[1]",
    ]
    assert "median_blur must be odd" in p[1].message
    assert "radius must be > 0" in p[2].message


def test_missing_required_key() -> None:
    _, p = to_params(
        {"preprocess": {"rolling_ball": {"light_background": True}}}
    )
    assert p == [Problem("preprocess.rolling_ball", "missing key 'radius'")]


def test_section_must_be_a_mapping() -> None:
    _, p = to_params({"region": 0.1})
    assert p == [Problem("region", "must be a mapping, got 0.1")]


def test_optional_step_must_be_mapping_or_null() -> None:
    _, p = to_params({"preprocess": {"rolling_ball": 50}})
    assert p == [
        Problem("preprocess.rolling_ball", "must be a mapping or null, got 50")
    ]


def test_ranges_must_be_a_list() -> None:
    _, p = to_params({"hct": {"ranges": {"d_min": 8}}})
    assert p == [Problem("hct.ranges", "must be a list, got {'d_min': 8}")]


def test_range_must_be_a_mapping() -> None:
    _, p = to_params({"hct": {"ranges": [8]}})
    assert p == [Problem("hct.ranges[0]", "must be a mapping, got 8")]


def test_bool_is_not_a_number() -> None:
    _, p = to_params({"region": {"bottom_fraction": True}})
    assert len(p) == 1
    assert p[0].path == "region"
    assert "bottom_fraction must be a number" in p[0].message


# ---------------------------------------------------------------------
# plain
# ---------------------------------------------------------------------


def test_plain_accepts_params_objects_and_numpy() -> None:
    assert plain(HCTParams((HCTRange(8, 30),)), "hct") == {
        "ranges": [
            {
                "d_min": 8,
                "d_max": 30,
                "min_dist_ratio": 0.4,
                "canny_threshold": 200.0,
                "accumulator_threshold": 15.0,
            }
        ]
    }
    value = plain(
        {"a": np.int64(3), "b": (8, 8), "c": np.float64(0.5), "d": np.True_},
        "x",
    )
    assert value == {"a": 3, "b": [8, 8], "c": 0.5, "d": True}
    assert [type(v) for v in value.values()] == [int, list, float, bool]


@pytest.mark.parametrize("bad", [object(), {1, 2}, b"x"])
def test_plain_rejects_other_types(bad: object) -> None:
    with pytest.raises(ParamsError, match=r"preprocess\.a"):
        plain({"a": bad}, "preprocess")


def test_plain_names_list_positions() -> None:
    with pytest.raises(ParamsError, match=r"hct\.ranges\[1\]"):
        plain({"ranges": [{}, object()]}, "hct")


def test_problems_after_a_failed_list_are_kept() -> None:
    _, p = to_params({"hct": {"ranges": [8], "extra": 1}, "region": {"x": 1}})
    assert sorted(q.path for q in p) == [
        "hct.extra",
        "hct.ranges[0]",
        "region.x",
    ]
