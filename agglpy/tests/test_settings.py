import copy
import filecmp
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from agglpy.cfg import (create_settings,
                        create_settings_dict, find_all_images,
                        load_manager_settings, validate_settings)
from agglpy.defaults import DEFAULT_SETTINGS_SCHEMA
from agglpy.errors import SettingsStructureError
from agglpy.tests.fixtures import (expected_valid_config,
                                   expected_valid_config_processed,
                                   input_multi_raw_wdir, tests_dir)


def test_settings_file_exists(input_multi_raw_wdir: Path):
    assert (input_multi_raw_wdir / "settings.yml").exists()


def test_config_validation(expected_valid_config: dict[str, Any]):
    config = expected_valid_config
    # No exception should be raised
    validate_settings(config, DEFAULT_SETTINGS_SCHEMA)


def test_empty_config():
    config = {}
    with pytest.raises(SettingsStructureError, match="Missing key 'general' in config"):
        validate_settings(config, DEFAULT_SETTINGS_SCHEMA)


def test_missing_keys():
    config = {
        "general": {
            "working_dir": ".",
        }
        # 'metadata', 'data', 'analysis', 'export' keys are missing
    }
    with pytest.raises(SettingsStructureError, match="Missing key 'metadata' in config"):
        validate_settings(config, DEFAULT_SETTINGS_SCHEMA)


def test_wrong_condition_structure(expected_valid_config: dict[str, Any]):
    config = expected_valid_config
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
        SettingsStructureError,
        match=re.escape(
            "Condition 'ambient_temp' at .metadata.conditions must "
            "be a list of length 2"
        ),
    ):
        for c in cfg:
            validate_settings(c, DEFAULT_SETTINGS_SCHEMA)
    with pytest.raises(
        SettingsStructureError,
        match=re.escape(
            "The first element of 'ambient_temp' at .metadata.conditions"
            " must be a number (int or float)"
        ),
    ):
        validate_settings(cfg_wrong_list[0], DEFAULT_SETTINGS_SCHEMA)
    with pytest.raises(
        SettingsStructureError,
        match=re.escape(
            f"The second element of 'ambient_temp' at .metadata.conditions"
            f" must be a string representing a unit"
        ),
    ):
        validate_settings(cfg_wrong_list[1], DEFAULT_SETTINGS_SCHEMA)


def test_wrong_value_type(expected_valid_config: dict[str, Any]):
    # Case 1- wrong multiple type (tuple)
    config = expected_valid_config
    config["data"]["default"]["Dmin"] = "should_be_int"
    expected_error_message = (
        f"Expected one of (<class 'int'>, <class 'float'>, <class 'NoneType'>) "
        f"at .data.default.Dmin, but got str"
    )
    with pytest.raises(SettingsStructureError, match=re.escape(expected_error_message)):
        validate_settings(config, DEFAULT_SETTINGS_SCHEMA)

    # Case 2- wrong single type
    config2 = expected_valid_config
    config2["general"]["working_dir"] = 1
    expected_error_message2 = (
        f"Expected <class 'str'> at .general.working_dir, but got int"
    )
    with pytest.raises(SettingsStructureError, match=expected_error_message2):
        validate_settings(config2, DEFAULT_SETTINGS_SCHEMA)


def test_extra_keys(expected_valid_config):
    config = expected_valid_config
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
        SettingsStructureError,
        match="Extra key 'extra_key' found at ",
    ):
        validate_settings(config_extra_key_root, DEFAULT_SETTINGS_SCHEMA)

    with pytest.raises(
        SettingsStructureError,
        match="Extra key 'extra_key' found at .analysis",
    ):
        validate_settings(config_extra_key_lvl2, DEFAULT_SETTINGS_SCHEMA)

    with pytest.raises(
        SettingsStructureError,
        match="Extra key 'extra_key' found at .export.draw_particles",
    ):
        validate_settings(config_extra_key_lvl3, DEFAULT_SETTINGS_SCHEMA)


def test_valid_config_file_loading(tests_dir, expected_valid_config_processed):
    settings = load_manager_settings(
        tests_dir / "input/valid_config_only/settings.yml"
    )
    assert settings == expected_valid_config_processed


def test_find_images(input_multi_raw_wdir):
    images = find_all_images(input_multi_raw_wdir)
    expected = [
        input_multi_raw_wdir / Path("D7-017.tif"),
        input_multi_raw_wdir / Path("D7-019.tif"),
        input_multi_raw_wdir / Path("D7-021.tif"),
    ]
    assert images == expected


def test_create_settings_dict(input_multi_raw_wdir):
    images = find_all_images(input_multi_raw_wdir)
    settings = create_settings_dict(images=images)
    expected = {
        "general": {"working_dir": "."},
        "metadata": {"conditions": {}},
        "data": {
            "default": {
                "img_file": "auto",
                "HCT_file": "auto",
                # "correction_file": "auto",
                "magnification": "auto",
                "pixel_size": "auto",
                "Dmin": 3,
                "Dmax": 250,
                "Dspace": [3, 50, 140],
                "dist2R": 0.5,
                "param1": [220, 270],
                "param2": [14, 24],
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
            "PSD_space": [0, 10, "step", "auto"],
            "PSD_space_log": False,
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


def test_create_yaml_settings(monkeypatch, tests_dir, input_multi_raw_wdir):

    # Use monkeypatch to simulate user input
    monkeypatch.setattr("builtins.input", lambda _: "y")

    # Ensure the output directory exists
    output_dir = tests_dir.joinpath("output/settings")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_path = tests_dir.joinpath("expected/settings/settings.yml")
    assert expected_path.exists(), "Expected YAML settings file not found"

    # Generate the settings.yml file
    create_settings(
        dir_path=input_multi_raw_wdir,
        output_path=output_dir / "settings.yml",
    )
    output_path = output_dir / "settings.yml"
    # Compare the generated file with the expected file
    assert filecmp.cmp(
        expected_path,
        output_path,
        shallow=False,
    ), f"The generated {output_path} file does not match the expected {expected_path} file."


def test_create_yaml_settings_infer_images(tests_dir):
    # Input
    input_path = tests_dir / "input/multiple_image_raw_no_settings/"
    
    # Output
    # Ensure the output directory exists
    output_dir = tests_dir.joinpath("output/settings")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "settings_infer_images.yml"

    # Expected
    expected_path = tests_dir.joinpath(
        "expected/settings/settings_infer_images.yml"
    )
    assert expected_path.exists(), "Expected YAML settings file not found"
    
    # Generate the settings.yml file
    create_settings(
        dir_path=input_path,
        output_path=output_path,
    )
    
    # Compare the generated file with the expected file
    assert filecmp.cmp(
        expected_path,
        output_path,
        shallow=False,
    ), f"The generated {output_path} file does not match the expected {expected_path} file."


# # Clean up the output files after the test
# @pytest.fixture(autouse=True)
# def cleanup_output_settings(tests_dir):
#     yield
#     out_created_settings_path = tests_dir.joinpath(
#         "output/settings/settings.yml"
#     )
#     if out_created_settings_path.exists():
#         os.remove(out_created_settings_path)
#     out_created_infer_settings_path = tests_dir.joinpath(
#         "output/settings/settings_infer_images.yml"
#     )
#     if out_created_infer_settings_path.exists():
#         os.remove(out_created_infer_settings_path)
    

# Clean up the output files after the test
@pytest.fixture(autouse=True)
def cleanup_output_settings(request, tests_dir):
    output_dirs = [
        tests_dir / "output/settings/",
    ]

    def cleanup():
        for output_dir in output_dirs:
            if output_dir.exists():
                try:
                    shutil.rmtree(output_dir)
                except Exception as e:
                    print(f"Error removing {output_dir}: {e}")
    request.addfinalizer(cleanup)