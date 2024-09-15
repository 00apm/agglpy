import pathlib

import pytest

from agglpy.img_ds import ImgDataSet
from agglpy.manager import Manager


@pytest.fixture
def tests_dir():
    return pathlib.Path(__file__).parent

@pytest.fixture
def input_multi_raw_wdir():
    return pathlib.Path(__file__).parent / "input" / "multiple_image_raw"


@pytest.fixture
def input_single_wdir():
    return pathlib.Path(__file__).parent / "input" / "single_image"

@pytest.fixture
def input_multi_wdir():
    return pathlib.Path(__file__).parent / "input" / "multiple_image"

@pytest.fixture
def expected_valid_config():
    """Config valid for YAML dump"""
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
                "correction_file": "auto",
                "magnification": "auto",
                "Dmin": None,
                "Dmax": None,
                "Dspace": [3, 50, 140],
                "dist2R": 0.5,
                "param1": [220, 270],
                "param2": [14, 24],
                "additional_info": None,
            },
            "images": {
                "D7-017": {
                    "img_file": "D7-017.tif",
                    "HCT_file": "D7-017_HCT.csv",
                    "correction_file": "D7-017_HCT.csv",
                    "magnification": "auto",
                    "Dmin": None,
                    "Dmax": None,
                    "Dspace": [3, 50, 140],
                    "dist2R": 0.5,
                    "param1": [220, 270],
                    "param2": [14, 24],
                    "additional_info": None,
                },
                "D7-019": {
                    "img_file": "D7-019-modified.tif",
                    "HCT_file": "D7-019_HCT.csv",
                    "correction_file": "D7-019_HCT.csv",
                    "magnification": "auto",
                    "Dmin": None,
                    "Dmax": None,
                    "Dspace": [3, 50, 140],
                    "dist2R": 0.5,
                    "param1": [220, 270],
                    "param2": [14, 24],
                    "additional_info": None,
                },
            },
            "exclude_images": [
                "D7-017",
                "D7-021",
            ],
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
    return config

@pytest.fixture
def expected_valid_config_processed(expected_valid_config):
    """Config valid after handling defaults
    State after loading to manager
    """
    config = expected_valid_config
    config["data"]["images"]["D7-017"]["magnification"] = None
    config["data"]["images"]["D7-019"]["magnification"] = None
    config["analysis"]["PSD_space"][-1] = None
    return config


@pytest.fixture
def model_IA(case_dir):
    return ImgDataSet(str(case_dir))


@pytest.fixture
def manager_obj_not_init(input_multi_raw_wdir):
    return Manager(input_multi_raw_wdir, initialize=False)

@pytest.fixture
def manager_obj(input_multi_raw_wdir):
    return Manager(input_multi_raw_wdir, settings_filepath="settings.yml")