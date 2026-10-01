"""Shared pytest fixtures, discovered automatically by pytest (no import needed)."""

import pathlib

import pytest


@pytest.fixture
def tests_dir():
    """Path to the tests/ directory."""
    return pathlib.Path(__file__).parent


@pytest.fixture
def input_multi_raw_wdir():
    """Flat dir with 3 .tif images and a settings.yml, no images/ subdirs yet.

    Not a valid Manager working dir until init_mgr_dirstruct() is run on it.
    """
    return (
        pathlib.Path(__file__).parent / "data" / "input" / "multiple_image_raw"
    )


@pytest.fixture
def input_single_wdir():
    """Single D5 image with ImageJ RoiSet and fitting CSV (legacy layout)."""
    return pathlib.Path(__file__).parent / "data" / "input" / "single_image"


@pytest.fixture
def input_multi_wdir():
    """Valid Manager working dir: images/<name>/ subdirs, settings.yml.

    D7-021 and D7-023 are in exclude_images, so only D7-017 and D7-019 are
    analysed. Uses rolling_ball: null, like the real D5 series settings.
    """
    return pathlib.Path(__file__).parent / "data" / "input" / "multiple_image"


@pytest.fixture
def expected_valid_config():
    """Raw settings dict that passes validate_settings (sentinels like "auto" kept).

    Matches data/input/valid_config_only/settings.yml as read from YAML, with
    the anchor/merge keys already expanded into each image entry.
    """
    config = {
        "general": {
            "working_dir": ".",
        },
        "metadata": {
            "conditions": {
                "ambient_temp": [21, "°C"],
                "ambient_pressure": [101.3, "kPa"],
            },
        },
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
                    "img_file": "D7-017.tif",
                    "HCT_file": "D7-017_HCT.csv",
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
                "D7-019": {
                    "img_file": "D7-019-modified.tif",
                    "HCT_file": "D7-019_HCT.csv",
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
            },
            "exclude_images": [
                "D7-017",
                "D7-021",
            ],
        },
        "analysis": {
            "PSD_space": {
                "start": 0.0,
                "end": 20.0,
                "periods": 20,
                "log": True,
                "step": False,
            },
            "collector_threshold": 0.5,
        },
        "export": {
            "draw_particles": {
                "labels": True,
                "alpha": 0.2,
            },
        },
    }
    return config


@pytest.fixture
def expected_valid_config_processed(expected_valid_config):
    """expected_valid_config after handle_defaults: "auto" resolved to None.

    This is the state returned by load_manager_settings().
    """
    config = expected_valid_config
    config["data"]["images"]["D7-017"]["magnification"] = None
    config["data"]["images"]["D7-019"]["magnification"] = None
    config["data"]["images"]["D7-017"]["pixel_size"] = None
    config["data"]["images"]["D7-019"]["pixel_size"] = None

    return config
