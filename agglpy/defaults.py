# Type definitions for type checking
from typing import Any, Literal, Tuple, Mapping, Union

import numpy as np

from agglpy.typing import (
    ImageSettingsTypedDict,
    ImageRawSettingsTypedDict,
    YamlRawSettingsTypedDict,
    PPSouceCsvType
)

# General defaults
SUPPORTED_IMG_FORMATS: set[str] = {"tif"}
DEFAULT_SETTINGS_FILENAME: str = "settings.yml"


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
        "images": {},
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
DEFAULT_IMAGE_SETTINGS_VALUES: ImageRawSettingsTypedDict = {
    "img_file": "",
    "HCT_file": "",
    # "correction_file": "",
    "magnification": "auto",
    "pixel_size": "auto", 
    "Dmin": None,
    "Dmax": None,
    "Dspace": [3, 50, 140],
    "dist2R": 0.5,
    "param1": [220, 270],
    "param2": [14, 24],
    "additional_info": None,
}


# Settings schema
DEFAULT_IMAGE_SETTINGS_SCHEMA: Mapping[str, Any] = {
    "img_file": (str, type(None)),
    "HCT_file": (str, type(None)),
    # "correction_file": (str, type(None)),
    "magnification": (int, float, str),
    "pixel_size": (int, float, str), 
    "Dmin": (int, float, type(None)),
    "Dmax": (int, float, type(None)),
    "Dspace": list,
    "dist2R": (int, float),
    "param1": list,
    "param2": list,
    "additional_info": (str, type(None)),
}

DEFAULT_SETTINGS_SCHEMA: Mapping[str, Any] = {
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
        "PSD_space": list,
        "PSD_space_log": bool,
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
    Tuple[str, ...]
]
# VALID_PARTICLE_CSV_TYPE: set[str] = {"agglpy", "ImageJ"}
VALID_PARTICLE_CSV_DATA: ValidParticleCsvType = {
    "agglpy": (
        "ID",
        "X",#: (np.int_, np.float_),
        "Y",#: (np.int_, np.float_),
        "R",#: (np.int_, np.float_),
    ),
    "ImageJ": (
        "Index",#: np.int_,
        "Type",#: str,
        "X",#: (np.int_, np.float_),
        "Y",#: (np.int_, np.float_),
        "Width",#: (np.int_, np.float_),
        "Height",#: (np.int_, np.float_),
    ),
    "agglpy_old": (
        "ID",
        "X (pixels)",#: (np.int_, np.float_),
        "Y (pixels)",#: (np.int_, np.float_),
        "Radius (pixels)",#: (np.int_, np.float_),
    ),
}


# Other defaults
STR_DEFAULT: set[str] = {"default", "auto", "normal"}
STR_NONE: set[str] = {"none", "null", "nan"}
STR_TRUE: set[str] = {"true", "t", "y", "yes", "yeah", "yup"}
