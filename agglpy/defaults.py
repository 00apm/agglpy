# Type definitions for type checking
from typing import Any, List, Literal, Tuple, Mapping, Union

import numpy as np

from agglpy.typing import (
    ImageSettingsTypedDict,
    ImageRawSettingsTypedDict,
    YamlRawSettingsTypedDict,
    PPSouceCsvType,
    PreprocessFunction,
    HctParameter,
)

# General defaults
SUPPORTED_IMG_FORMATS: set[str] = {"tif"}
DEFAULT_SETTINGS_FILENAME: str = "settings.yml"
PREPROCESS_FUNCTIONS: Tuple[PreprocessFunction] = ("median_blur",)
HCT_PARAMETERS: List[HctParameter] = [
    "d_min",
    "d_max",
    "dist2R",
    "param1",
    "param2",
]

# Settings defaults
DEFAULT_SETTINGS: YamlRawSettingsTypedDict = {
    "general": {
        "working_dir": ".",
    },
    "metadata": {"conditions": {}},
    "data": {
        "default": {
            "img_file": "auto",
            "HCT_file": "auto",
            "magnification": "auto",
            "pixel_size": "auto",
            "crop_ratio": 0.0,
            "median_blur": 3,
            "d_min": [3, 50],
            "d_max": [50, 140],
            "dist2R": 0.5,
            "param1": 200,
            "param2": 15,
            "additional_info": None,
        },
        "images": {},
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
DEFAULT_IMAGE_SETTINGS_VALUES: ImageRawSettingsTypedDict = {
    "img_file": "",
    "HCT_file": "",
    "magnification": "auto",
    "pixel_size": "auto",
    "crop_ratio": 0.0,
    "median_blur": 3,
    "d_min": [3, 50],
    "d_max": [50, 140],
    "dist2R": 0.5,
    "param1": 200,
    "param2": 15,
    "additional_info": None,
}


# Settings schema
DEFAULT_IMAGE_SETTINGS_SCHEMA: Mapping[str, Any] = {
    "img_file": (str, type(None)),
    "HCT_file": (str, type(None)),
    "magnification": (int, float, str),
    "pixel_size": (int, float, str),
    "crop_ratio": float,
    "median_blur": (int, type(None)),
    "d_min": (list, int),
    "d_max": (list, int),
    "dist2R": (list, float),
    "param1": (list, float, int),
    "param2": (list, float, int),
    "additional_info": (str, type(None)),
}

PSD_SPACE_SCHEMA: Mapping[str, Any] = {
    "start": float,
    "end": float,
    "periods": (int, float),
    "log": bool,
    "step": bool,
}


# Main settings schema used for validation
DEFAULT_SETTINGS_SCHEMA: Mapping[str, Any] = {
    # TODO: change validation system, using this schema does not check 
    #       internal dicts
    "general": {
        "working_dir": str,
    },
    "metadata": {
        "conditions": dict,
    },
    "data": {
        "default": DEFAULT_IMAGE_SETTINGS_SCHEMA,
        "images": dict,  # Dictionary with dynamic keys, each containing data like `default`
        "exclude_images": list,
    },
    "analysis": {
        "PSD_space": (type(None), dict),
        "collector_threshold": (int, float),
    },
    "export": {
        "draw_particles": {
            "labels": bool,
            "alpha": (int, float),
        },
    },
}


# Valid Particle data CSV files structure
ValidParticleCsvType = Mapping[
    PPSouceCsvType,
    # Mapping[
    #     str,
    #     Union[Tuple[type, ...], type],
    # ],
    Tuple[str, ...],
]
# VALID_PARTICLE_CSV_TYPE: set[str] = {"agglpy", "ImageJ"}
VALID_PARTICLE_CSV_DATA: ValidParticleCsvType = {
    "agglpy": (
        "ID",
        "X",  #: (np.int_, np.float_),
        "Y",  #: (np.int_, np.float_),
        "R",  #: (np.int_, np.float_),
    ),
    "ImageJ": (
        "Index",  #: np.int_,
        "Type",  #: str,
        "X",  #: (np.int_, np.float_),
        "Y",  #: (np.int_, np.float_),
        "Width",  #: (np.int_, np.float_),
        "Height",  #: (np.int_, np.float_),
    ),
    "agglpy_old": (
        "ID",
        "X (pixels)",  #: (np.int_, np.float_),
        "Y (pixels)",  #: (np.int_, np.float_),
        "Radius (pixels)",  #: (np.int_, np.float_),
    ),
}


# Other defaults
STR_DEFAULT: set[str] = {"default", "auto", "normal"}
STR_NONE: set[str] = {"none", "null", "nan"}
STR_TRUE: set[str] = {"true", "t", "y", "yes", "yeah", "yup"}
