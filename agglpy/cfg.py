from copy import deepcopy
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
)

import yaml

from agglpy.auxiliary import isdefault

DEFAULT_IMAGE_SETTINGS_SCHEMA = {
    "img_name": (str, type(None)),
    "fitting_name": (str, type(None)),
    "magnification": (str, int, float),
    "Dmin": (int, float, type(None)),
    "Dmax": (int, float, type(None)),
    "Dspace": list,
    "dist2R": (int, float),
    "param1": list,
    "param2": list,
    "additional_info": (str, type(None)),
}

DEFAULT_IMAGE_SETTINGS_VALUES = {
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

# Define the manager settings schema for validation
DEFAULT_SETTINGS_SCHEMA = {
    "general": {
        "working_dir": str,
    },
    "metadata": {
        "conditions": {
            "ambient_temp": list,
            "ambient_pressure": list,
        },
    },
    "data": {
        "default": DEFAULT_IMAGE_SETTINGS_SCHEMA,
        "images": dict,  # Dictionary with dynamic keys, each containing data like `default`
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
    "exclude_images": list,
}


def validate_conditions(
    conditions: Dict[str, Any], route="metadata.conditions"
):
    """Validate the structure of the 'conditions' under 'metadata'."""
    for condition_name, value in conditions.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                f"Condition '{condition_name}' at {route} must be a list"
                f" of length 2"
            )
        if not isinstance(value[0], (int, float)):
            raise ValueError(
                f"The first element of '{condition_name}' at {route}"
                f" must be a number (int or float)"
            )
        if not isinstance(value[1], str):
            raise ValueError(
                f"The second element of '{condition_name}' at {route}"
                f" must be a string representing a unit"
            )


def validate_settings(config: dict, schema: dict, route: str = "") -> None:
    """Recursively validate that config matches schema."""
    if isinstance(schema, dict):
        # Ensure that config is also a dict
        if not isinstance(config, dict):
            raise ValueError(
                f"Expected a dictionary at {route}, but got {type(config).__name__}"
            )
        # Check each key in the schema
        for key, subschema in schema.items():
            if key not in config:
                raise ValueError(f"Missing key '{key}' in config at {route}")
            if key == "conditions":  # Special handling for 'conditions'
                validate_conditions(config[key], route + f".{key}")
            if key == "images":  # Special handling for image settings
                for im in config[key]:
                    # Handle image settings (fill empty values, etc)
                    config[key][im] = handle_img_settings(im, config[key][im])
                    # Validate dict structure of image settings
                    validate_settings(
                        config[key][im],
                        DEFAULT_IMAGE_SETTINGS_SCHEMA,
                        route + f".{key}.{im}",
                    )
            else:
                validate_settings(
                    config[key],
                    subschema,
                    route + f".{key}",
                )

        # Check for extra keys
        for key in config:
            if key not in schema:
                raise ValueError(f"Extra key '{key}' found at {route}")

    elif isinstance(schema, tuple):
        # Ensure that config value matches one of the allowed types
        if not isinstance(config, schema):
            raise ValueError(
                f"Expected one of {schema} at {route}, but got {type(config).__name__}"
            )


def handle_img_settings(
    image_name: str,
    image_config: Any,
) -> dict:
    """Handle image settings for non-recognized types or missing values

    Args:
        image_name (str): string representing image name
        image_config (Any): object obtained from config for handling

    Returns:
        dict: image settings with proper structure
    """
    if image_config is None:
        image_config = DEFAULT_IMAGE_SETTINGS_VALUES
    
    if isinstance(image_config, dict):
        if isdefault(image_config["img_name"]):
            image_config["img_name"] = image_name
        if isdefault(image_config["fitting_name"]):
            image_config["fitting_name"] = image_name + "_fitting.csv"
    else:
        raise ValueError(
            f"Image settings for {image_name} are not recognized."
            f" See default image settings dict."
        )
    return image_config


def load_manager_settings(path: Path) -> dict:
    with open(path, mode="rt") as settings_file:
        settings = yaml.safe_load(settings_file)
        validate_settings(
            config=settings,
            schema=DEFAULT_SETTINGS_SCHEMA,
        )
    return settings
