"""The agglpy logger: silent until logging is configured."""

import logging

import pytest

from agglpy.logger import logger


def test_logger_is_silent_until_configured(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert logger is logging.getLogger("agglpy")
    # A NullHandler keeps Python's last-resort handler from printing
    # agglpy's warnings to stderr.
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    logger.warning("not shown")
    assert capsys.readouterr() == ("", "")
