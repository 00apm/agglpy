"""Exception hierarchy of the new code."""

import pytest

from agglpy.errors import (
    AgglpyError,
    DuplicateParticlesWarning,
    MissingImagesWarning,
    ParamsError,
    ParticleTableError,
    TableError,
    ValuesNotCountedWarning,
)


@pytest.mark.parametrize(
    "error", [ParamsError, TableError, ParticleTableError]
)
def test_errors_share_one_base_and_are_value_errors(
    error: type[Exception],
) -> None:
    # One base lets a frontend catch everything agglpy raises on purpose;
    # ValueError keeps `except ValueError` in user code working.
    assert issubclass(error, AgglpyError)
    assert issubclass(error, ValueError)


def test_particle_table_error_is_a_table_error() -> None:
    # `except TableError` catches every table problem, particle tables
    # included.
    assert issubclass(ParticleTableError, TableError)


@pytest.mark.parametrize(
    "warning",
    [DuplicateParticlesWarning, ValuesNotCountedWarning, MissingImagesWarning],
)
def test_warnings_are_user_warnings(warning: type[Warning]) -> None:
    assert issubclass(warning, UserWarning)
