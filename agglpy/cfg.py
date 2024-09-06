from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from copy import deepcopy

import yaml

from agglpy.auxiliary import isdefault


YamlDataType = Union[
    Dict[str, Any], List[Any], Tuple[Any], str, int, float, None
]

SUPPORTED_IMG_FORMATS = [
    "tif",
]

STR_DEFAULT = {"default", "auto", "normal"}
STR_NONE = {"none", "null", "nan"}
STR_TRUE = {"true", "t", "y", "yes", "yeah", "yup"}


DEFAULT_SETTINGS = {
    "general": {"working_dir": "."},
    "metadata": {"conditions": {}},
    "data": {
        "default": {
            "img_file": "auto",
            "HCT_file": "auto",
            "correction_file": "auto",
            "magnification": "auto",
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


DEFAULT_IMAGE_SETTINGS_VALUES = {
    "img_file": "",
    "HCT_file": "",
    "correction_file": "",
    "magnification": "auto",
    "Dmin": None,
    "Dmax": None,
    "Dspace": [3, 50, 140],
    "dist2R": 0.5,
    "param1": [220, 270],
    "param2": [14, 24],
    "additional_info": None,
}


DEFAULT_IMAGE_SETTINGS_SCHEMA = {
    "img_file": (str, type(None)),
    "HCT_file": (str, type(None)),
    "correction_file": (str, type(None)),
    "magnification": (str, int, float),
    "Dmin": (int, float, type(None)),
    "Dmax": (int, float, type(None)),
    "Dspace": list,
    "dist2R": (int, float),
    "param1": list,
    "param2": list,
    "additional_info": (str, type(None)),
}


# Define the manager settings schema for validation
DEFAULT_SETTINGS_SCHEMA = {
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


def create_settings_dict(images: List[Union[str, Path]]) -> dict:
    """Create settings dictionary

    Create settings dict and fill default image settings based on
    file basenames provided in a list of image names.

    Args:
        images (List[str|Path]): list of image file names

    Returns:
        dict: settings dictionary with a structure of YAML config
    """
    settings = DEFAULT_SETTINGS
    for i in images:
        img = Path(i)
        settings["data"]["images"][img.stem] = "*default_img"
    return settings


def create_settings(
    dir_path: Path,
    output_path: Path = Path("settings.yml"),
    images: Optional[List[Union[str, Path]]] = None,
) -> dict:
    """Create settings YAML file

    Create settings YAML file and fill default image sattings. If image
    names list is not provided attempts to find all images in <dir_path> directory
    (not recursively)

    Args:
        dir_path (Path): working directory path (directory containing images
            for analysis)
        output_dir (Path): output settings file path. If path is relative
            <dir_path> will be treates as root. Default is Path("settings.yml")
        images (Optional[List[str|Path]]): list of image file names.
            If None finds all the images in dir_path. Default is None.

    Returns:
        dict: settings dictionary with a structure of YAML config
    """
    if images is None:
        images = find_images(dir_path)
    settings = create_settings_dict(images)

    if not dir_path.is_absolute():
        file_path = dir_path / output_path
    else:
        file_path = output_path
    if file_path.exists():
        # Prompt the user for confirmation to overwrite
        while True:
            overwrite = (
                input(
                    f"The file '{file_path}' already exists. "
                    f"Do you want to overwrite it? (y/n): "
                )
                .strip()
                .lower()
            )
            if overwrite in ["y", "n"]:
                break
            else:
                print("Please answer with 'y' for yes or 'n' for no.")

        if overwrite == "n":
            print("Operation aborted. The file was not overwritten.")
            return
        # Proceed if the user agrees to overwrite
        print(f"Overwriting the file '{file_path}'...")

    # To handle anchors and references in PyYAML, define the necessary parts
    with open(file_path, "w") as file:
        yaml.dump(
            settings,
            file,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    with open(file_path, "r+") as file:
        content = file.read()

        # Add the anchor `&default_img` to the 'default' key
        content = content.replace("default:", "default: &default_img")

        # Replace the strings with the correct anchor and merge keys
        content = content.replace("'*default_img'", "*default_img")
        content = content.replace("'<<: *default_img'", "<<: *default_img")

        # Write the modified content back to the file
        file.seek(0)
        file.write(content)
        file.truncate()

    print(f"Configuration saved to '{file_path}'.")
    return settings


def find_images(dir_path: Path) -> List[Path]:
    """Finds all the images of supported formats in a directory

    Args:
        dir_path (Path): working directory path (directory containing images
            for analysis)

    Returns:
        List[Path]: list of image file paths
    """
    images = []
    for f in SUPPORTED_IMG_FORMATS:
        images += list(dir_path.glob("*." + f))
    return images


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


def validate_settings(
    config: YamlDataType,
    schema: YamlDataType,
    route: str = "",
) -> None:
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
                    # Handle image settings (fill auto, empty values, etc)
                    # config[key][im] = handle_img_names(im, config[key][im])
                    # Validate dict structure of image settings
                    if config[key][im] is None:
                        config[key][im] = deepcopy(DEFAULT_IMAGE_SETTINGS_VALUES) 
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
    else:
        # Ensure that config value matches schema value
        if not isinstance(config, schema):
            raise ValueError(
                f"Expected {schema} at {route}, but got {type(config).__name__}"
            )


def handle_defaults(
    config: YamlDataType,
    route: str = "",
) -> YamlDataType:
    """Recursively handle default values in settings retrieved from YAML

    1.  Changes str values like: 'auto', 'default', 'null'
        to None python type.
    2.  Handles image names and its relevant HCT and corrected .csv
        data files

    Args:
        config (dict): _description_

    Returns:
        dict: _description_
    """
    valid_default = STR_DEFAULT.union(STR_NONE).union([""])
    if isinstance(config, dict):
        # Create a new dictionary to store the processed result
        processed_dict = {}

        for key, value in config.items():
            if key == "default":
                # Skip processing for the 'default' branch
                processed_dict[key] = value
            elif key == "images":
                # Special handling for 'images' key
                # First handle the automatic image naming
                # and then handle other default values
                processed_dict[key] = {
                    im: handle_defaults(
                        handle_img_names(im, value[im]),
                        route + f".{key}",
                    )
                    for im in value
                }
            else:
                # General handling for other dictionary entries
                processed_dict[key] = handle_defaults(value, route + f".{key}")
        return processed_dict
    elif isinstance(config, list):
        return [handle_defaults(item, route + f".{item}") for item in config]
    elif isinstance(config, tuple):
        return tuple(
            handle_defaults(item, route + f".{item}") for item in config
        )
    # If the data is a string and matches one of the default-like values, convert it to None
    elif isinstance(config, str) and config.lower() in valid_default:
        return None
    # If it's any other type, return it as is
    return config


def handle_img_names(
    image_name: str,
    image_config: Any,
) -> dict:
    """Handle image names

    Handle image names in settings for defaults, non-recognized types

    Args:
        image_name (str): string representing image name
        image_config (Any): object obtained from config for handling

    Returns:
        dict: image settings with proper structure
    """
    processed_config = deepcopy(image_config)

    if isinstance(processed_config, dict):
        if isdefault(processed_config["img_file"]):
            processed_config["img_file"] = image_name
        if isdefault(processed_config["HCT_file"]):
            processed_config["HCT_file"] = image_name + "_HCT.csv"
        if isdefault(processed_config["correction_file"]):
            processed_config["correction_file"] = image_name + "_HCT.csv"
    else:
        raise ValueError(
            f"Image settings for {image_name} are not recognized."
            f" See default image settings dict."
        )
    return processed_config


def load_manager_settings(path: Path) -> dict:
    """Loads settings for agglpy.Manager objects

    _extended_summary_

    Args:
        path (Path): _description_

    Returns:
        dict: _description_
    """
    with open(path, mode="rt", encoding="utf-8") as settings_file:
        settings = yaml.safe_load(settings_file)
        validate_settings(
            config=settings,
            schema=DEFAULT_SETTINGS_SCHEMA,
        )
        settings = handle_defaults(config=settings)
    return settings
