"""Exception hierarchy of the new code."""

import pytest

from agglpy.errors import (
    AgglpyError,
    DuplicateParticlesWarning,
    ParamsError,
    ParticleTableError,
)


@pytest.mark.parametrize("error", [ParamsError, ParticleTableError])
def test_errors_share_one_base_and_are_value_errors(
    error: type[Exception],
) -> None:
    # One base lets a frontend catch everything agglpy raises on purpose;
    # ValueError keeps `except ValueError` in user code working.
    assert issubclass(error, AgglpyError)
    assert issubclass(error, ValueError)


def test_duplicate_warning_is_a_user_warning() -> None:
    assert issubclass(DuplicateParticlesWarning, UserWarning)
