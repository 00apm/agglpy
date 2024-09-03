import copy
from pathlib import Path
import re
from typing import Any

import pytest

from agglpy.cfg import (
    DEFAULT_SETTINGS_SCHEMA,
    load_manager_settings,
    validate_settings,
)
from agglpy.tests.fixtures import input_multi_raw_wdir, valid_config


def test_settings_file_exists(input_multi_raw_wdir: Path):
    assert (input_multi_raw_wdir / "settings.yml").exists()


def test_valid_config(valid_config: dict[str, Any]):
    config = valid_config
    # No exception should be raised
    validate_settings(config, DEFAULT_SETTINGS_SCHEMA)


def test_empty_config():
    config = {}
    with pytest.raises(ValueError, match="Missing key 'general' in config"):
        validate_settings(config, DEFAULT_SETTINGS_SCHEMA)


def test_missing_keys():
    config = {
        "general": {
            "working_dir": ".",
        }
        # 'metadata', 'data', 'analysis', 'export' keys are missing
    }
    with pytest.raises(ValueError, match="Missing key 'metadata' in config"):
        validate_settings(config, DEFAULT_SETTINGS_SCHEMA)


def test_wrong_condition_structure(valid_config: dict[str, Any]):
    config = valid_config
    cfg = list(copy.deepcopy(config) for i in range(3))
    cfg[0]["metadata"]["conditions"]["ambient_temp"] = [21]
    cfg[1]["metadata"]["conditions"]["ambient_temp"] = 1
    cfg[2]["metadata"]["conditions"]["ambient_temp"] = "wrong condition"

    cfg_wrong_list = list(copy.deepcopy(config) for i in range(2))
    cfg_wrong_list[0]["metadata"]["conditions"]["ambient_temp"] = [
        "1.23",
        "°C",
    ]
    cfg_wrong_list[1]["metadata"]["conditions"]["ambient_temp"] = [
        3.21,
        3.21,
    ]

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Condition 'ambient_temp' at .metadata.conditions must "
            "be a list of length 2"
        ),
    ):
        for c in cfg:
            validate_settings(c, DEFAULT_SETTINGS_SCHEMA)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The first element of 'ambient_temp' at .metadata.conditions"
            " must be a number (int or float)"
        ),
    ):
        validate_settings(cfg_wrong_list[0], DEFAULT_SETTINGS_SCHEMA)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The second element of 'ambient_temp' at .metadata.conditions"
            f" must be a string representing a unit"
        ),
    ):
        validate_settings(cfg_wrong_list[1], DEFAULT_SETTINGS_SCHEMA)


def test_wrong_value_type(valid_config: dict[str, Any]):
    config = valid_config
    config["data"]["default"]["Dmin"] = "should_be_int"
    expected_error_message = (
        "Expected one of (<class 'int'>, <class 'float'>, <class 'NoneType'>) "
        "at .data.default.Dmin, but got str"
    )
    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        validate_settings(config, DEFAULT_SETTINGS_SCHEMA)


def test_extra_keys(valid_config):
    config = valid_config
    # case1 error in root lvl
    config_extra_key_root = copy.deepcopy(config)
    config_extra_key_root["extra_key"] = "This key should trigger an error"
    # case2 error in lvl 2
    config_extra_key_lvl2 = copy.deepcopy(config)
    config_extra_key_lvl2["analysis"][
        "extra_key"
    ] = "This key should trigger an error"
    # case3 error in lvl 3
    config_extra_key_lvl3 = copy.deepcopy(config)
    config_extra_key_lvl3["export"]["draw_particles"][
        "extra_key"
    ] = "This key should trigger an error"

    # Test each configuration for extra keys
    with pytest.raises(
        ValueError,
        match="Extra key 'extra_key' found at ",
    ):
        validate_settings(config_extra_key_root, DEFAULT_SETTINGS_SCHEMA)

    with pytest.raises(
        ValueError,
        match="Extra key 'extra_key' found at .analysis",
    ):
        validate_settings(config_extra_key_lvl2, DEFAULT_SETTINGS_SCHEMA)

    with pytest.raises(
        ValueError,
        match="Extra key 'extra_key' found at .export.draw_particles",
    ):
        validate_settings(config_extra_key_lvl3, DEFAULT_SETTINGS_SCHEMA)


def test_valid_config_file(input_multi_raw_wdir):
    settings = load_manager_settings(input_multi_raw_wdir / "settings.yml")
    print(settings)
