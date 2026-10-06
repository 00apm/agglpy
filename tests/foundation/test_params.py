"""Params: validated on creation, immutable, comparable, hashable."""

import dataclasses

import pytest

from agglpy.errors import ParamsError
from agglpy.params import (
    AnalysisParams,
    ClaheParams,
    HCTParams,
    HCTRange,
    PreprocessParams,
    RollingBallParams,
)


def test_hct_range_defaults_match_the_legacy_hct() -> None:
    r = HCTRange(d_min=5, d_max=25)
    assert (r.dist2R, r.param1, r.param2) == (0.4, 200.0, 15.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_min": 0, "d_max": 10},
        {"d_min": 10, "d_max": 10},
        {"d_min": 20, "d_max": 10},
        {"d_min": 5.5, "d_max": 10},
        {"d_min": 5, "d_max": 10, "dist2R": 0},
        {"d_min": 5, "d_max": 10, "param1": -1},
        {"d_min": 5, "d_max": 10, "param2": float("nan")},
    ],
)
def test_invalid_hct_range_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ParamsError):
        HCTRange(**kwargs)


def test_bool_is_not_a_number() -> None:
    # bool is a subclass of int; YAML turns `yes` into True.
    with pytest.raises(ParamsError, match="d_min"):
        HCTRange(d_min=True, d_max=10)
    with pytest.raises(ParamsError, match="radius"):
        RollingBallParams(radius=True)


def test_error_message_names_the_field_and_value() -> None:
    with pytest.raises(ParamsError, match=r"param2.*-3"):
        HCTRange(d_min=5, d_max=10, param2=-3)


def test_ranges_list_becomes_tuple_and_is_hashable() -> None:
    params = HCTParams(ranges=[HCTRange(5, 25), HCTRange(20, 80)])
    assert isinstance(params.ranges, tuple)
    assert (
        len({params, HCTParams(ranges=(HCTRange(5, 25), HCTRange(20, 80)))})
        == 1
    )


def test_hct_params_need_at_least_one_range() -> None:
    with pytest.raises(ParamsError, match="at least one"):
        HCTParams(ranges=())


def test_hct_params_accept_only_ranges() -> None:
    with pytest.raises(ParamsError, match="HCTRange"):
        HCTParams(ranges=({"d_min": 5, "d_max": 25},))


def test_params_are_immutable_and_changed_by_replace() -> None:
    r = HCTRange(5, 25)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.param2 = 20  # type: ignore[misc]
    changed = dataclasses.replace(r, param2=20)
    assert changed.param2 == 20 and r.param2 == 15.0


def test_replace_validates_too() -> None:
    with pytest.raises(ParamsError):
        dataclasses.replace(HCTRange(5, 25), d_max=1)


def test_preprocess_defaults_do_nothing() -> None:
    p = PreprocessParams()
    assert p == PreprocessParams(0.0, None, None, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop_ratio": -0.1},
        {"crop_ratio": 1.0},
        {"median_blur": 4},
        {"median_blur": 1},
    ],
)
def test_invalid_preprocess_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ParamsError):
        PreprocessParams(**kwargs)


def test_clahe_tile_grid_list_becomes_tuple() -> None:
    c = ClaheParams(clip_limit=2.0, tile_grid=[8, 8])
    assert c.tile_grid == (8, 8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip_limit": 0, "tile_grid": (8, 8)},
        {"clip_limit": 2.0, "tile_grid": (8,)},
        {"clip_limit": 2.0, "tile_grid": (0, 8)},
    ],
)
def test_invalid_clahe_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ParamsError):
        ClaheParams(**kwargs)


def test_rolling_ball_needs_positive_radius() -> None:
    with pytest.raises(ParamsError):
        RollingBallParams(radius=0)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_collector_threshold_in_range(threshold: float) -> None:
    assert AnalysisParams(threshold).collector_threshold == threshold


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_collector_threshold_out_of_range(threshold: float) -> None:
    with pytest.raises(ParamsError):
        AnalysisParams(threshold)


def test_asdict_gives_a_plain_record() -> None:
    # The Phase 3 detection record stores params this way (D-021).
    record = dataclasses.asdict(HCTParams(ranges=(HCTRange(5, 25),)))
    assert record == {
        "ranges": (
            {
                "d_min": 5,
                "d_max": 25,
                "dist2R": 0.4,
                "param1": 200.0,
                "param2": 15.0,
            },
        )
    }
