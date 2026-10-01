"""Fixtures for the core tests: every test runs once per implementation."""

import pytest

from support.synthetic.adapters import ADAPTERS


@pytest.fixture(params=list(ADAPTERS))
def adapter(request: pytest.FixtureRequest) -> str:
    """Name of the implementation under test (a key of ``ADAPTERS``)."""
    return request.param
