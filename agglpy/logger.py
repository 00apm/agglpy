import logging.config
from pathlib import Path

import yaml

# Create a default logger for the package (API)
logger: logging.Logger = logging.getLogger("agglpy")
logger.addHandler(logging.NullHandler())


# Config used only for endpoint interface (currently jupyter notebook)
ENDPOINT_DEFAULT_LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(levelname)s: %(message)s",
        },
        "detailed": {
            "format": "(%(asctime)s) [%(levelname)s|%(name)s|%(module)s]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "WARNING",
            "formatter": "simple",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": None, # created inside setup functions
            "maxBytes": 10000000, # max 10MB
            "backupCount": 3,
        },
    },
    "loggers": {
        "root": {
            "level": "DEBUG",
            "handlers": [
                "stderr",
                "file",
            ],
        },
        "matplotlib": {  # Suppress matplotlib logs
            "level": "WARNING",
            "handlers": ["stderr"],
            "propagate": False,
        },
    },
}

def setup_notebook_logger(
    log_cfg_path: Path = Path("./.agglpy/log/logger_config.yml"),
    logs_path: Path = Path("./.agglpy/log/agglpy.log"),
) -> None:
    """
    Setup a logger for the jupyter notebook environment.

    It checks if the logger config yaml file exists at log_cfg_path.
    If not, it creates the file with a default configuration and applies it.

    Args:
        log_cfg_path (Path): Path to the logger config yaml file.
        logs_path (Path): Path where the log file should be created.
    """

    # If logger_config.yml exists, load it; otherwise, create it with the default config
    if log_cfg_path.exists():
        # Load configuration from existing YAML file
        with open(log_cfg_path, "r") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Error loading YAML configuration file: {exc}")
                return
    else:
        # Create the YAML file with the default configuration
        log_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        config = ENDPOINT_DEFAULT_LOGGER_CONFIG

        # Set the log filename
        if logs_path.is_dir():
            logs_path = logs_path / "agglpy.log"
           
        config["handlers"]["file"]["filename"] = str(logs_path)

        with open(log_cfg_path, "w") as file:
            yaml.dump(
                config, 
                file, 
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )       

    logs_path.parent.mkdir(parents=True, exist_ok=True) 
    # Apply the configuration
    try:
        logging.config.dictConfig(config)
        print(f"Logger configured using config at: {log_cfg_path}")
    except Exception as e:
        print(f"Failed to configure logger: {e}")

