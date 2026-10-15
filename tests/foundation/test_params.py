"""Params: validated on creation, immutable, comparable, hashable."""

import dataclasses

import numpy as np
import pytest
import yaml

from agglpy.errors import ParamsError
from agglpy.params import (
    ClaheParams,
    HCTParams,
    HCTRange,
    PreprocessParams,
    RegionParams,
    RollingBallParams,
)


def test_hct_range_defaults_match_the_legacy_hct() -> None:
    r = HCTRange(d_min=5, d_max=25)
    assert (
        r.min_dist_ratio,
        r.canny_threshold,
        r.accumulator_threshold,
    ) == (0.4, 200.0, 15.0)


@pytest.mark.parametrize("old", ["dist2R", "param1", "param2"])
def test_hct_range_old_names_are_gone(old: str) -> None:
    with pytest.raises(TypeError, match=old):
        HCTRange(d_min=5, d_max=25, **{old: 1.0})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_min": 0, "d_max": 10},
        {"d_min": 10, "d_max": 10},
        {"d_min": 20, "d_max": 10},
        {"d_min": 5.5, "d_max": 10},
        {"d_min": 5, "d_max": 10, "min_dist_ratio": 0},
        {"d_min": 5, "d_max": 10, "canny_threshold": -1},
        {"d_min": 5, "d_max": 10, "accumulator_threshold": float("nan")},
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
    with pytest.raises(ParamsError, match=r"accumulator_threshold.*-3"):
        HCTRange(d_min=5, d_max=10, accumulator_threshold=-3)


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
        r.accumulator_threshold = 20  # type: ignore[misc]
    changed = dataclasses.replace(r, accumulator_threshold=20)
    assert changed.accumulator_threshold == 20
    assert r.accumulator_threshold == 15.0


def test_replace_validates_too() -> None:
    with pytest.raises(ParamsError):
        dataclasses.replace(HCTRange(5, 25), d_max=1)


def test_preprocess_defaults_do_nothing() -> None:
    p = PreprocessParams()
    assert p == PreprocessParams(None, None, None)


def test_preprocess_fields_are_in_step_order() -> None:
    # The region left preprocessing: it is RegionParams now.
    names = [f.name for f in dataclasses.fields(PreprocessParams)]
    assert names == ["median_blur", "rolling_ball", "clahe"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"median_blur": 4},
        {"median_blur": 1},
    ],
)
def test_invalid_preprocess_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ParamsError):
        PreprocessParams(**kwargs)


@pytest.mark.parametrize("fraction", [0, 0.0, 0.1, 0.999])
def test_region_bottom_fraction_is_accepted(fraction: float) -> None:
    region = RegionParams(bottom_fraction=fraction)
    assert region.bottom_fraction == fraction
    assert type(region.bottom_fraction) is float


def test_region_default_keeps_the_whole_image() -> None:
    assert RegionParams().bottom_fraction == 0.0


@pytest.mark.parametrize(
    "fraction", [-0.1, 1.0, 1.5, float("nan"), "0.1", True]
)
def test_invalid_region_is_rejected(fraction: object) -> None:
    with pytest.raises(ParamsError, match="bottom_fraction"):
        RegionParams(bottom_fraction=fraction)  # type: ignore[arg-type]


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


def test_rolling_ball_defaults_suit_a_dark_background() -> None:
    # SEM images of particles are usually bright on dark
    params = RollingBallParams(radius=50)
    assert params.light_background is False
    assert params.presmooth is True


def test_rolling_ball_needs_positive_radius() -> None:
    with pytest.raises(ParamsError):
        RollingBallParams(radius=0)


def test_asdict_gives_a_plain_record() -> None:
    # The Phase 3 detection record stores params this way (D-021).
    record = dataclasses.asdict(HCTParams(ranges=(HCTRange(5, 25),)))
    assert record == {
        "ranges": (
            {
                "d_min": 5,
                "d_max": 25,
                "min_dist_ratio": 0.4,
                "canny_threshold": 200.0,
                "accumulator_threshold": 15.0,
            },
        )
    }


def test_values_are_stored_as_plain_python_types() -> None:
    # Values taken from a DataFrame are NumPy scalars; the record
    # (asdict, D-021) must still be plain data that YAML can write.
    hct = HCTRange(
        np.int64(5),
        np.int64(25),
        canny_threshold=200,
        accumulator_threshold=np.float64(15),
    )
    pre = PreprocessParams(
        median_blur=np.int64(5),
        clahe=ClaheParams(np.float64(2.0), np.array([8, 8])),
        rolling_ball=RollingBallParams(np.int64(50), np.bool_(False)),
    )

    assert [type(v) for v in dataclasses.astuple(hct)] == [
        int,
        int,
        float,
        float,
        float,
    ]
    assert type(pre.median_blur) is int
    assert pre.clahe is not None and pre.rolling_ball is not None
    assert type(pre.clahe.clip_limit) is float
    assert [type(v) for v in pre.clahe.tile_grid] == [int, int]
    assert type(pre.rolling_ball.radius) is float
    assert type(pre.rolling_ball.light_background) is bool
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
