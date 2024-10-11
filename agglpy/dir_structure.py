import os
import shutil
from pathlib import Path
from typing import List, Optional

from agglpy.cfg import (
    create_settings,
    find_valid_settings,
    load_manager_settings,
)
from agglpy.defaults import DEFAULT_SETTINGS_FILENAME, SUPPORTED_IMG_FORMATS
from agglpy.errors import DirectoryStructureError, SettingsStructureError
from agglpy.logger import logger
from agglpy.typing import YamlSettingsTypedDict


def init_mgr_dirstruct(
    path: os.PathLike,
    settings_path: Optional[os.PathLike] = None,
    move: bool = False,
) -> None:
    """Initialize directory structure for Agglomerate analysis.

    Initialize directory structure in <path> for Agglomerate analysis
    via agglpy.manager Manager object.

    ## Settings file
    Create directory structure based on a settings file.
    If <settings_path> is not provided, attempt to find default name
    setting file (settings.yml). If default file does not exist, find
    all images inside <path> append them in new config file and then
    initialize the directory structure.

    ## Directory initialization
    Create separate directory for each image specified in settings.
    Copy all images specified in settings to designated dir.

    Args:
        path (str): Directory path for Agglomerate analysis.
        settings_path (str, optional): Settings yaml file path.
            Defaults to None
        move (bool): If true image files are moved to designated
            directories instead of copying. Defaults to False.
    """
    wdir = Path(path)
    logger.debug(f"Initializing Manager directory structure at: {wdir}")
    if is_mgr_dirstruct(wdir):
        logger.debug(
            f"Manager directory structure is already prepared for analysis."
        )
        return
    if settings_path is None:

        # find valid settings path, settings conte

        sett_paths = find_valid_settings(wdir)

        if len(sett_paths) == 0:
            logger.debug(f"Creating settings yaml file at: {wdir}")
            create_settings(wdir)
            settings_path = wdir / DEFAULT_SETTINGS_FILENAME
        elif len(sett_paths) > 1:
            raise DirectoryStructureError(
                f"Multiple valid settings files {[i.name for i in sett_paths]} "
                f"were found at {wdir}."
            )
        else:
            settings_path = sett_paths[0]
    try:
        settings = load_manager_settings(settings_path)
    except SettingsStructureError as err:
        raise DirectoryStructureError(
            "Settings structure incorrect. Initialization failed."
        ) from err
    images = settings["data"]["images"]
    images_dir = wdir / "images"
    # ensure that source image exists
    for img in images:
        img_file: str = images[img]["img_file"]
        if not (wdir / img_file).exists():
            raise DirectoryStructureError(
                f"Image file: {img_file} declared in .data.images settings "
                f"for {img} was not found in directory: {wdir}"
            ) from FileNotFoundError
    for img in images:
        img_src = wdir / img_file
        img_dst = images_dir / str(img) / str(images[img]["img_file"])
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(img_src, img_dst)
            logger.debug(f"Image file: {img_file} moved to {img_dst.parent}")
        else:
            shutil.copy2(img_src, img_dst)
            logger.debug(f"Image file: {img_file} copied to {img_dst.parent}")


def is_mgr_dirstruct(path: os.PathLike) -> bool:
    """Check if <path> has valid directory structure

    Valid directory structure comprise:
    1.  A valid YAML settings file
    2.  ./images/ directory with separate image subdirecotries for each
        image dataset listed in the settings file.

    Raises DirectoryStructureError if directory structure is incorrect

    Args:
        path (str): Directory path for Agglomerate analysis

    Returns:
        bool: True if directory structure is correct.
    """
    try:
        validate_mgr_dirstruct(path)
    except DirectoryStructureError:
        return False
    else:
        return True


def validate_mgr_dirstruct(path: os.PathLike):
    """Check if <path> has valid directory structure

    Valid directory structure comprise:
    1.  A valid YAML settings file
    2.  ./images/ directory with separate image subdirecotries for each
        image dataset listed in the settings file.

    Raises DirectoryStructureError if directory structure is incorrect

    Args:
        path (str): Directory path for Agglomerate analysis

    Returns:
        bool: True if directory structure is correct.
    """
    # Check if settings file is in directory
    wdir: Path = Path(path)
    valid_files: List[Path] = find_valid_settings(wdir)
    if len(valid_files) == 0:
        raise DirectoryStructureError(
            f"Valid settings file was not fount at {path}. To check "
            f"if settings are valid try to load it manually using: "
            f"agglpy.cfg.load_manager_settings() and check the error message."
        ) from FileNotFoundError
    elif len(valid_files) > 1:
        raise DirectoryStructureError(
            f"Multiple valid settings files {[i.name for i in valid_files]} "
            f"were found at {path}."
        )
    else:
        valid_path = valid_files[0]

    settings = load_manager_settings(valid_path)

    # Check if image for each Image Data Set exists
    DSpaths = find_datasets_paths(
        path=wdir,
        settings=settings,
        ignore=True,
    )
    logger.debug(f"Manager directory structure positively validated.")


def find_datasets_paths(
    path: os.PathLike,
    settings: YamlSettingsTypedDict,
    ignore: bool = True,
) -> List[Path]:
    """Detect Image Data Sets in directory structure

    Detect Data Sets in working directory. Data Set is an image that:
    1)  is included in settings yaml at `.data.images`
    2)  is placed in a separate directory at
        working_directory/images/<image_name>

    Args:
        path (str): Directory path for Agglmerate analysis
        settings (dict): Settings dict retrieved from
            agglpy.cfg.load_manager_settings
        ignore (bool, optional): If True images listed in settings
            at `.data.exclude_images` are ignored. Defaults to True.
    Returns:
        List[Path]: List of paths to valid images found in directory
            structure.
    """

    wdir: Path = Path(path)
    DSpaths: List[Path] = []
    ignore_list = []
    if ignore:
        ignore_list = settings["data"]["exclude_images"]

    DS_main_dir = wdir / "images"

    images = settings["data"]["images"]
    for i in images:
        if i not in ignore_list:
            img_dir = DS_main_dir / Path(images[i]["img_file"]).stem
            if not img_dir.exists():
                raise DirectoryStructureError(
                    f"Image Data Set directory {repr(img_dir)} not found"
                )
            else:
                img_file = img_dir / Path(images[i]["img_file"])

                # If img_file in settings is without extension
                # find images with supported extensions
                if img_file.suffix == "":
                    img_list: List[Path] = []
                    # Iterate over the supported extensions
                    for ext in SUPPORTED_IMG_FORMATS:
                        # Search for files with the specified name and extension
                        img_list.extend(img_dir.glob(f"{img_file.name}.{ext}"))
                    if len(img_list) > 1:
                        raise DirectoryStructureError(
                            f"Multiple image files {img_list} found inside "
                            f"Image Data Set directory {img_dir}. Please "
                            f"verify .data.images.<img_name>.img_file settings."
                        )
                    elif len(img_list) == 0:
                        raise DirectoryStructureError(
                            f"File {img_file} not found in {img_dir}"
                        )
                    else:
                        img_file = img_list[0]
                        # update img_file in settings
                        settings["data"]["images"][i][
                            "img_file"
                        ] = img_file.name
                DSpaths.append(img_file)

    return DSpaths
