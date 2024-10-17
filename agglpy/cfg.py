import os
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union, cast

import yaml

from agglpy.auxiliary import (
    txt_is_default_or_none,
    txt_is_none_plus,
    txt_is_default,
)
from agglpy.defaults import (
    DEFAULT_IMAGE_SETTINGS_SCHEMA,
    DEFAULT_IMAGE_SETTINGS_VALUES,
    DEFAULT_SETTINGS,
    DEFAULT_SETTINGS_FILENAME,
    DEFAULT_SETTINGS_SCHEMA,
    STR_DEFAULT,
    STR_NONE,
    SUPPORTED_IMG_FORMATS,
)
from agglpy.errors import SettingsStructureError
from agglpy.logger import logger
from agglpy.typing import (
    ImageSettingsTypedDict,
    YamlSettingsTypedDict,
    YamlRawSettingsTypedDict,
)


def create_settings_dict(images: List[Path]) -> YamlRawSettingsTypedDict:
    """Create settings dictionary

    Create settings dict and fill default image settings based on
    file basenames provided in a list of image names.

    Args:
        images (List[Pathlike]): list of image file names

    Returns:
        dict: settings dictionary with a structure of YAML config
    """
    settings: YamlRawSettingsTypedDict = DEFAULT_SETTINGS
    for i in images:
        img = Path(i)
        settings["data"]["images"][img.stem] = {
            "<<": "*default_img",
            "img_file": img.name,
        }
    return settings


def create_settings(
    dir_path: Path,
    output_path: Path = Path(DEFAULT_SETTINGS_FILENAME),
    images: List[Path] | None = None,
):
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
        images = find_all_images(dir_path)
    settings = create_settings_dict(images)

    if not output_path.is_absolute():
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
        content: str = file.read()

        # Add the anchor `&default_img` to the 'default' key
        content = content.replace("default:", "default: &default_img")

        # Replace the strings with the correct anchor and merge keys
        content = content.replace("'*default_img'", "*default_img")
        content = content.replace("'<<: *default_img'", "<<: *default_img")
        content = content.replace("'<<':", "<<:")

        # Write the modified content back to the file
        file.seek(0)
        file.write(content)
        file.truncate()
    print(f"Configuration saved to '{file_path}'.")


# Function may be used after future implementation of pydantic validation
def fill_empty_image_settings(
    settings: YamlSettingsTypedDict,
) -> None:
    default = settings["data"]["default"]
    def_str = "yaml file"
    if default is None:
        default = DEFAULT_IMAGE_SETTINGS_VALUES
        def_str = "agglpy lib"
    for key, val in settings["data"]["images"].items():
        if val is None: # finds defined images without settings dict
            val = default
            logger.debug(
                f"Settings for image: {key} were empty. Filling with "
                f"{def_str} defaults."
            )


def find_all_images(dir_path: os.PathLike) -> List[Path]:
    """Finds all the images of supported formats in a directory

    Args:
        dir_path (Path): working directory path (directory containing images
            for analysis)

    Returns:
        List[Path]: list of image file paths
    """
    images: List[Path] = []
    dir_path = Path(dir_path)
    for f in SUPPORTED_IMG_FORMATS:
        images += list(dir_path.glob("*." + f))
    return images


def validate_conditions(
    conditions: Dict[str, Any], route="metadata.conditions"
):
    """Validate the structure of the 'conditions' under 'metadata'."""
    for condition_name, value in conditions.items():
        if not isinstance(value, list) or len(value) != 2:
            raise SettingsStructureError(
                f"Condition '{condition_name}' at {route} must be a list"
                f" of length 2"
            )
        if not isinstance(value[0], (int, float)):
            raise SettingsStructureError(
                f"The first element of '{condition_name}' at {route}"
                f" must be a number (int or float)"
            )
        if not isinstance(value[1], str):
            raise SettingsStructureError(
                f"The second element of '{condition_name}' at {route}"
                f" must be a string representing a unit"
            )


def validate_settings(
    config: Mapping[str, Any],
    schema: Union[Mapping[str, Any], Tuple[type, ...], type],
    route: str = "",
) -> None:
    """Recursively validate that config matches schema.

    Args:
        config (Mapping[str, Any]): Dictionary created by pyyaml lib
            for validation
        schema (Union[Mapping[str, Any], Tuple[type, ...], type]): Schema dict,
            tuple of types, or a single type.
        route (str, optional): Route in settings dict. Used for debugging.
            Defaults to "".

    Raises:
        SettingsStructureError: Raised when config structure does not match schema.
    """
    if isinstance(schema, Mapping):
        # If schema is a dictionary, config must also be a dictionary
        if not isinstance(config, Mapping):
            raise SettingsStructureError(
                f"Expected a dictionary at {route}, but got "
                f"{type(config).__name__}"
            )
        # Validate each key in the schema
        for key, subschema in schema.items():
            if key not in config:
                raise SettingsStructureError(
                    f"Missing key '{key}' in config at {route}"
                )
            if key == "conditions":  # Special handling for 'conditions'
                validate_conditions(config[key], route + f".{key}")
            elif key == "images":  # Special handling for image settings
                for im in config[key]:
                    if config[key][im] is None:
                        # filling image settings if it was not provided
                        # TODO: this should not be there, deprecete when 
                        # updating to pydantic
                        config[key][im] = deepcopy(
                            DEFAULT_IMAGE_SETTINGS_VALUES
                        )
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
                raise SettingsStructureError(
                    f"Extra key '{key}' found at {route}"
                )
    elif isinstance(schema, tuple):
        # Ensure that config value matches one of the allowed types
        if not isinstance(config, schema):
            raise SettingsStructureError(
                f"Expected one of {schema} at {route}, but got "
                f"{type(config).__name__}"
            )
    elif isinstance(schema, type):
        # Ensure that config value matches the specific type
        if not isinstance(config, schema):
            raise SettingsStructureError(
                f"Expected {schema} at {route}, but got {type(config).__name__}"
            )
    else:
        raise TypeError(
            f"Invalid schema type at {route}: {type(schema).__name__}"
        )


def handle_defaults(
    config: Mapping[str, Any],
    route: str = "",
) -> Mapping[str, Any]:
    """Recursively handle default values in settings retrieved from YAML

    1.  Changes str values like: 'auto', 'default', 'null'
        to None python type.
    2.  Handles implicit settings at .data.images: image file names and its
        HCT and corrected .csv data file names

    Args:
        config (dict): recursively called config parts

    Returns:
        dict: processed config
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
    image_config: ImageSettingsTypedDict,
) -> ImageSettingsTypedDict:
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
        if txt_is_default_or_none(processed_config["img_file"]):
            # Needs change: for now .tif is hardcoded; warning is displayed
            processed_config["img_file"] = image_name + ".tif"
            warning_msg = (
                "img_file for image: image_name was set to default. File "
                "extension is assumed to be .tif; if it needs to be change "
                "please specify file name with extension in yaml settings"
            )
            warnings.warn(warning_msg, RuntimeWarning)
            logger.warning(warning_msg)
        if isinstance(processed_config["HCT_file"], str):
            if txt_is_default(processed_config["HCT_file"]):
                processed_config["HCT_file"] = image_name + "_HCT.csv"
            elif txt_is_none_plus(processed_config["HCT_file"]):
                processed_config["HCT_file"] = None
            else:
                pass
    else:
        raise SettingsStructureError(
            f"Image settings for {image_name} are not recognized."
            f" See default image settings dict."
        )
    return processed_config


def is_valid_settings(config: Mapping[str, Any]) -> bool:
    """Check if settings are valid for analysis

    Check if settings loaded via pyyaml are valid for Agglomerate analysis

    Args:
        settings (YamlDataType): dict containing settings. Loaded via pyyaml

    Returns:
        bool: True if settings are valid
    """
    try:
        validate_settings(
            config=config,
            schema=DEFAULT_SETTINGS_SCHEMA,
        )
    except SettingsStructureError:
        return False
    else:
        return True


def is_valid_settings_file(path: os.PathLike) -> bool:
    """Load a settings file at <path> and check its content

    Args:
        path (str): Settings file path

    Returns:
        bool: True if settings are valid
    """
    fpath: Path = Path(path)
    with open(fpath, mode="rt", encoding="utf-8") as settings_file:
        settings = yaml.safe_load(settings_file)
        if is_valid_settings(config=settings):
            return True
        else:
            return False


def load_manager_settings(
    path: os.PathLike,
    handle_def: bool = True,
) -> YamlSettingsTypedDict:
    """Loads settings for agglpy.Manager objects and validate them

    1. Load settings at given <path>
    2. Validate settings structure
    3. (Optiopnal) Handle default values and implicit settings at .data.images

    Args:
        path (Path): yaml settings path
        handle_defaults (bool): If true handle default values

    Returns:
        dict: loaded settings dict
    """
    fpath: Path = Path(path)
    with open(path, mode="rt", encoding="utf-8") as settings_file:
        settings: Mapping[str, Any] = yaml.safe_load(settings_file)
        try:
            validate_settings(
                config=settings,
                schema=DEFAULT_SETTINGS_SCHEMA,
            )
        except SettingsStructureError:
            raise
    logger.debug(f"Agglpy analysis settings loaded from: {str(path)}")
    if handle_def:
        settings = handle_defaults(config=settings)
        logger.debug("Default values in settings dict handled.")
    # Casting to YamlSettingsTypedDict as settings structure were validated
    settings = cast(YamlSettingsTypedDict, settings)
    return settings


def find_valid_settings(path: os.PathLike) -> List[Path]:
    """Detect valid settings in directory <path>

    Args:
        path (str): Directory path for Agglmerate analysis

    Returns:
        List[Path]: List of valid settings paths.
    """
    # Check if settings file is in directory
    wdir: Path = Path(path)
    valid_files: List[Path] = []
    for p in wdir.iterdir():
        a = re.search(".*\.(ya?ml)", str(p))  # match .yml and .yaml files
        if a is not None:
            p_valid = Path(a.group())
            try:
                # load and validate settings
                _ = load_manager_settings(
                    path=p_valid,
                    handle_def=False,
                )
            except SettingsStructureError:
                # if settings are not valid continue
                continue
            except:
                raise
            else:
                # if settings are valid append
                valid_files.append(p_valid)
    logger.debug(f"Valid settings found: {valid_files}")
    return valid_files

