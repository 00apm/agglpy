import pathlib

import pytest
import yaml

from agglpy.aggl import ImgAgl
from agglpy.cfg import load_manager_settings
from agglpy.manager import Manager


@pytest.fixture
def input_multi_raw_wdir():
    return pathlib.Path(__file__).parent / "input" / "multiple_image_raw"


@pytest.fixture
def input_single_wdir():
    return pathlib.Path(__file__).parent / "input" / "single_image"


@pytest.fixture
def valid_config():
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
                "img_name": "",
                "fitting_name": "",
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
                    "img_name": "",
                    "fitting_name": "",
                    "magnification": "auto",
                    "Dmin": None,
                    "Dmax": None,
                    "Dspace": [3, 50, 140],
                    "dist2R": 0.5,
                    "param1": [220, 270],
                    "param2": [14, 24],
                    "additional_info": None,
                }
            },
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
        "exclude_images": [
            "D7-017",
            "D7-021",
        ],
    }
    return config


@pytest.fixture
def model_IA(case_dir):
    return ImgAgl(str(case_dir))


@pytest.fixture
def manager_obj(input_multi_raw_wdir):
    return Manager(input_multi_raw_wdir, settings_filename="settings.yml")
