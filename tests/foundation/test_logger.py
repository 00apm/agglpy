"""Notebook logger setup: no shared-state mutation, no print."""

import copy
import importlib
import logging.config
from pathlib import Path
from typing import Any

import pytest

# agglpy/__init__.py re-exports the Logger object under the name
# `logger`, which hides the submodule from `from agglpy import logger`.
logger_module = importlib.import_module("agglpy.logger")


@pytest.fixture
def applied(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record configs passed to dictConfig instead of applying them.

    Applying a real config would attach a file handler to the root
    logger for the rest of the test session.
    """
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(logging.config, "dictConfig", calls.append)
    return calls


def test_setup_leaves_the_default_config_unchanged(
    tmp_path: Path, applied: list[dict[str, Any]]
) -> None:
    before = copy.deepcopy(logger_module.ENDPOINT_DEFAULT_LOGGER_CONFIG)

    logger_module.setup_notebook_logger(
        log_cfg_path=tmp_path / "cfg.yml", log_path=tmp_path
    )

    assert logger_module.ENDPOINT_DEFAULT_LOGGER_CONFIG == before
    filename = applied[0]["handlers"]["file"]["filename"]
    assert filename == str(tmp_path / "agglpy.log")


def test_setup_does_not_print(
    tmp_path: Path,
    applied: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger_module.setup_notebook_logger(
        log_cfg_path=tmp_path / "cfg.yml", log_path=tmp_path
    )

    assert capsys.readouterr().out == ""


def test_broken_config_file_warns_instead_of_printing(
    tmp_path: Path,
    applied: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = tmp_path / "cfg.yml"
    cfg.write_text("handlers: [unclosed", encoding="utf-8")

    with pytest.warns(UserWarning, match="logger config"):
        logger_module.setup_notebook_logger(
            log_cfg_path=cfg, log_path=tmp_path
        )

    assert applied == []
    assert capsys.readouterr().out == ""
