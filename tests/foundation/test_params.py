"""Params: validated on creation, immutable, comparable, hashable."""

import dataclasses

import numpy as np
import pytest
import yaml

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


def test_values_are_stored_as_plain_python_types() -> None:
    # Values taken from a DataFrame are NumPy scalars; the record
    # (asdict, D-021) must still be plain data that YAML can write.
    hct = HCTRange(
        np.int64(5), np.int64(25), param1=200, param2=np.float64(15)
    )
    pre = PreprocessParams(
        crop_ratio=0,
        median_blur=np.int64(5),
        clahe=ClaheParams(np.float64(2.0), np.array([8, 8])),
        rolling_ball=RollingBallParams(np.int64(50), np.bool_(False)),
    )
    analysis = AnalysisParams(np.float64(0.5))

    assert [type(v) for v in dataclasses.astuple(hct)] == [
        int,
        int,
        float,
        float,
        float,
    ]
    assert type(pre.crop_ratio) is float and type(pre.median_blur) is int
    assert pre.clahe is not None and pre.rolling_ball is not None
    assert type(pre.clahe.clip_limit) is float
    assert [type(v) for v in pre.clahe.tile_grid] == [int, int]
    assert type(pre.rolling_ball.radius) is float
    assert type(pre.rolling_ball.light_background) is bool
    assert type(analysis.collector_threshold) is float
    yaml.safe_dump(dataclasses.asdict(hct))
    yaml.safe_dump(dataclasses.asdict(pre))


@pytest.mark.parametrize("field", ["light_background", "presmooth"])
@pytest.mark.parametrize("value", ["no", 1, None])
def test_rolling_ball_flags_must_be_bool(field: str, value: object) -> None:
    with pytest.raises(ParamsError, match=field):
        RollingBallParams(radius=5, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clahe", {"clip_limit": 2.0, "tile_grid": (8, 8)}),
        ("rolling_ball", 50),
    ],
)
def test_nested_params_must_have_their_class(
    field: str, value: object
) -> None:
    with pytest.raises(ParamsError, match=field):
        PreprocessParams(**{field: value})


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda: HCTParams(ranges=HCTRange(5, 25)), "ranges"),
        (lambda: HCTParams(ranges=None), "ranges"),
        (lambda: ClaheParams(2.0, 8), "tile_grid"),
    ],
)
def test_non_sequence_is_a_params_error(build: object, field: str) -> None:
    # YAML typo `tile_grid: 8`: a ParamsError naming the field, not a
    # bare TypeError from tuple().
    with pytest.raises(ParamsError, match=field):
        build()  # type: ignore[operator]
