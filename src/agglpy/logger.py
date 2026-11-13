"""The agglpy logger and a logging setup for notebooks.

`logger` (``"agglpy"``) has only a ``NullHandler``, so agglpy writes no
log output until the application configures logging;
`setup_notebook_logger` does that for a notebook session.
"""

import copy
import logging.config
import warnings
from pathlib import Path
from typing import Any

import yaml

logger: logging.Logger = logging.getLogger("agglpy")
"""The package logger, silent until logging is configured."""
logger.addHandler(logging.NullHandler())


# Config used only for endpoint interface (currently jupyter notebook).
# Never mutated: setup functions work on a deep copy.
ENDPOINT_DEFAULT_LOGGER_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(levelname)s: %(message)s",
        },
        "detailed": {
            "format": (
                "(%(asctime)s) [%(levelname)s|%(name)s|%(module)s]: "
                "%(message)s"
            ),
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
            "filename": None,  # created inside setup functions
            "maxBytes": 10000000,  # max 10MB
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
    log_path: Path = Path("./.agglpy/log/agglpy.log"),
) -> None:
    """Configure logging for a Jupyter notebook session.

    Applies the logging config file at `log_cfg_path`. When the file
    does not exist, writes the default config there first: warnings to
    stderr, everything from DEBUG on to a rotating log file.

    Parameters
    ----------
    log_cfg_path : Path, default "./.agglpy/log/logger_config.yml"
        The logging config YAML (``logging.config.dictConfig`` schema).
    log_path : Path, default "./.agglpy/log/agglpy.log"
        The log file named in a newly written config; a folder gets
        ``agglpy.log`` inside it.

    Warns
    -----
    UserWarning
        If the config file cannot be read or applied; logging is
        then not configured.
    """
    config: dict[str, Any]
    if log_cfg_path.exists():
        with open(log_cfg_path, encoding="utf-8") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                warnings.warn(
                    f"Could not read the logger config {log_cfg_path}: "
                    f"{exc}. Logging is not configured.",
                    stacklevel=2,
                )
                return
    else:
        log_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(ENDPOINT_DEFAULT_LOGGER_CONFIG)

        if log_path.is_dir():
            log_path = log_path / "agglpy.log"

        config["handlers"]["file"]["filename"] = str(log_path)

        with open(log_cfg_path, "w", encoding="utf-8") as file:
            yaml.dump(
                config,
                file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logging.config.dictConfig(config)
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        warnings.warn(
            f"Could not apply the logger config {log_cfg_path}: {exc}",
            stacklevel=2,
        )
        return
    logger.info(f"Logger configured using config at: {log_cfg_path}")
