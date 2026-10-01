import re
from pathlib import Path
from typing import Any

import pytest

from agglpy.cfg import (
    create_settings,
    create_settings_dict,
    find_all_images,
    load_manager_settings,
    load_yaml,
    validate_settings,
)
from agglpy.defaults import DEFAULT_SETTINGS_SCHEMA
from agglpy.errors import SettingsStructureError


def set_nested(
    config: dict[str, Any], keys: tuple[str, ...], value: Any
) -> None:
    """Set config[k1][k2]...[kn] = value."""
    *parents, last = keys
    for key in parents:
        config = config[key]
    config[last] = value


def test_fixture_raw_dir_has_settings_file(input_multi_raw_wdir: Path):
    """Sanity check on test data: the raw fixture dir ships a settings.yml."""
    assert (input_multi_raw_wdir / "settings.yml").exists()


def test_validate_settings_accepts_valid_config(
    expected_valid_config: dict[str, Any],
):
    """A complete, well-typed config passes validation."""
    config = expected_valid_config
    # No exception should be raised
    validate_settings(config, DEFAULT_SETTINGS_SCHEMA)


@pytest.mark.parametrize(
    ("config", "missing_key"),
    [
        pytest.param({}, "general", id="empty-config"),
        pytest.param(
            {"general": {"working_dir": "."}}, "metadata", id="only-general"
        ),
    ],
)
def test_validate_settings_missing_section_raises(config, missing_key):
    """The error names the first missing top-level section."""
    with pytest.raises(
        SettingsStructureError, match=f"Missing key '{missing_key}' in config"
    ):
        validate_settings(config, DEFAULT_SETTINGS_SCHEMA)


MSG_NOT_PAIR = "Condition 'ambient_temp' at .metadata.conditions must be a list of length 2"
MSG_NOT_NUMBER = (
    "The first element of 'ambient_temp' at .metadata.conditions"
    " must be a number (int or float)"
)
MSG_NOT_UNIT = (
    "The second element of 'ambient_temp' at .metadata.conditions"
    " must be a string representing a unit"
)


@pytest.mark.parametrize(
    ("value", "expected_msg"),
    [
        pytest.param([21], MSG_NOT_PAIR, id="list-too-short"),
        pytest.param(1, MSG_NOT_PAIR, id="number-not-list"),
        pytest.param("wrong condition", MSG_NOT_PAIR, id="string-not-list"),
        pytest.param(["1.23", "°C"], MSG_NOT_NUMBER, id="value-is-string"),
        pytest.param([3.21, 3.21], MSG_NOT_UNIT, id="unit-is-number"),
    ],
)
def test_validate_settings_bad_condition_raises(
    expected_valid_config: dict[str, Any], value, expected_msg
):
    """metadata.conditions entries must be [number, unit_string]."""
    expected_valid_config["metadata"]["conditions"]["ambient_temp"] = value
    with pytest.raises(SettingsStructureError, match=re.escape(expected_msg)):
        validate_settings(expected_valid_config, DEFAULT_SETTINGS_SCHEMA)


@pytest.mark.parametrize(
    ("keys", "value", "expected_msg"),
    [
        pytest.param(
            ("data", "default", "d_min"),
            "should_be_int",
            "Expected one of (<class 'list'>, <class 'int'>) "
            "at .data.default.d_min, but got str",
            id="one-of-several-types",
        ),
        pytest.param(
            ("general", "working_dir"),
            1,
            "Expected <class 'str'> at .general.working_dir, but got int",
            id="single-type",
        ),
    ],
)
def test_validate_settings_wrong_type_raises(
    expected_valid_config: dict[str, Any], keys, value, expected_msg
):
    """Type errors report the route to the bad value and the allowed types."""
    set_nested(expected_valid_config, keys, value)
    with pytest.raises(SettingsStructureError, match=re.escape(expected_msg)):
        validate_settings(expected_valid_config, DEFAULT_SETTINGS_SCHEMA)


@pytest.mark.parametrize(
    ("keys", "route"),
    [
        pytest.param(("extra_key",), "", id="root"),
        pytest.param(("analysis", "extra_key"), ".analysis", id="level-2"),
        pytest.param(
            ("export", "draw_particles", "extra_key"),
            ".export.draw_particles",
            id="level-3",
        ),
    ],
)
def test_validate_settings_extra_key_raises(
    expected_valid_config: dict[str, Any], keys, route
):
    """Unknown keys are rejected at every nesting level (catches typos)."""
    set_nested(expected_valid_config, keys, "This key should trigger an error")
    # "$" anchors the end, so the root case doesn't match a deeper route
    expected_msg = re.escape(f"Extra key 'extra_key' found at {route}") + "$"
    with pytest.raises(SettingsStructureError, match=expected_msg):
        validate_settings(expected_valid_config, DEFAULT_SETTINGS_SCHEMA)


def test_load_manager_settings_resolves_defaults(
    tests_dir, expected_valid_config_processed
):
    """Loading resolves YAML anchors/merge keys and turns "auto" into None."""
    settings = load_manager_settings(
        tests_dir / "data/input/valid_config_only/settings.yml"
    )
    assert settings == expected_valid_config_processed


def test_find_all_images(input_multi_raw_wdir):
    """Finds all .tif files in a flat dir."""
    images = find_all_images(input_multi_raw_wdir)
    expected = [
        input_multi_raw_wdir / Path("D7-017.tif"),
        input_multi_raw_wdir / Path("D7-019.tif"),
        input_multi_raw_wdir / Path("D7-021.tif"),
    ]
    assert images == expected


def test_create_settings_dict(input_multi_raw_wdir):
    """Default settings with one entry per image, each merging *default_img."""
    images = find_all_images(input_multi_raw_wdir)
    settings = create_settings_dict(images=images)
    expected = {
        "general": {"working_dir": "."},
        "metadata": {"conditions": {}},
        "data": {
            "default": {
                "img_file": "auto",
                "HCT_file": "auto",
                "magnification": "auto",
                "pixel_size": "auto",
                "crop_ratio": 0.0,
                "median_blur": 3,
                "rolling_ball": [50, True, True],
                "d_min": [3, 50],
                "d_max": [50, 140],
                "dist2R": 0.5,
                "param1": 200,
                "param2": 15,
                "additional_info": None,
            },
            "images": {
                "D7-017": {
                    "<<": "*default_img",
                    "img_file": "D7-017.tif",
                },
                "D7-019": {
                    "<<": "*default_img",
                    "img_file": "D7-019.tif",
                },
                "D7-021": {
                    "<<": "*default_img",
                    "img_file": "D7-021.tif",
                },
            },
            "exclude_images": [],
        },
        "analysis": {
            "PSD_space": None,
            "collector_threshold": 0.5,
        },
        "export": {
            "draw_particles": {
                "labels": True,
                "alpha": 0.2,
            },
        },
    }
    assert settings == expected


@pytest.mark.parametrize(
    ("input_dir", "expected_file"),
    [
        pytest.param(
            "multiple_image_raw", "settings.yml", id="dir-with-settings"
        ),
        pytest.param(
            "multiple_image_raw_no_settings",
            "settings_infer_images.yml",
            id="dir-without-settings",
        ),
    ],
)
def test_create_settings_writes_expected_yaml(
    tests_dir: Path, tmp_path: Path, input_dir, expected_file
):
    """Written YAML (with anchors) lists one entry per .tif found in dir_path."""
    output_path = tmp_path / "settings.yml"
    expected_path = tests_dir / "data/expected/settings" / expected_file

    create_settings(
        dir_path=tests_dir / "data/input" / input_dir,
        output_path=output_path,
    )

    # read_text() normalises line endings, so this works on every OS
    assert output_path.read_text(encoding="utf-8") == expected_path.read_text(
        encoding="utf-8"
    )


# ----------- YAML float parsing
# PyYAML implements YAML 1.1, where a float needs a dot AND a signed exponent,
# so "1e-6" or "1.0e6" would load as strings. load_yaml accepts them as floats.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1e-6", 1e-6),  # no dot
        ("1E-6", 1e-6),  # capital E
        ("1.0e6", 1e6),  # unsigned exponent
        ("-2.5e3", -2500.0),  # negative mantissa
        ("1.0e-6", 1e-6),  # already a float in YAML 1.1
        (".5e-3", 5e-4),  # already a float in YAML 1.1
    ],
)
def test_load_yaml_reads_exponent_notation_as_float(
    text: str, expected: float
):
    value = load_yaml(f"x: {text}")["x"]

    assert isinstance(value, float)
    assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("20", 20),  # int stays int
        ("D7-017", "D7-017"),  # image names stay strings
        (
            "auto",
            "auto",
        ),  # sentinels stay strings (resolved by handle_defaults)
        ("e5", "e5"),  # no mantissa, not a number
        ("'1e-6'", "1e-6"),  # explicitly quoted stays a string
    ],
)
def test_load_yaml_leaves_non_floats_unchanged(text: str, expected: Any):
    value = load_yaml(f"x: {text}")["x"]

    assert value == expected
    assert type(value) is type(expected)


def test_load_manager_settings_exponent_floats(
    input_multi_wdir: Path, tmp_path: Path
):
    """PSD_space in metres written as 1e-7 must load as float, not str."""
    text = (input_multi_wdir / "settings.yml").read_text(encoding="utf-8")
    # guard: the fixture still looks the way this test expects
    assert "start: 0\n" in text and "end: 10\n" in text
    text = text.replace("start: 0\n", "start: 1e-7\n")
    text = text.replace("end: 10\n", "end: 1e-5\n")
    settings_path = tmp_path / "settings.yml"
    settings_path.write_text(text, encoding="utf-8")

    settings = load_manager_settings(settings_path)

    assert settings["analysis"]["PSD_space"]["start"] == 1e-7
    assert settings["analysis"]["PSD_space"]["end"] == 1e-5
