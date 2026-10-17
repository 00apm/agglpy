"""Exception hierarchy of the new code."""

import pytest

from agglpy.errors import (
    AgglpyError,
    DuplicateParticlesWarning,
    FileFormatError,
    ImageError,
    MissingImagesWarning,
    NonCircularParticlesWarning,
    NoParticlesWarning,
    OutputExistsError,
    OutputLockedError,
    ParamsError,
    ParticlesOutsideImageWarning,
    ParticleTableError,
    PixelSizeWarning,
    TableError,
    ValuesNotCountedWarning,
)


@pytest.mark.parametrize(
    "error",
    [ParamsError, TableError, ParticleTableError, FileFormatError, ImageError],
)
def test_errors_share_one_base_and_are_value_errors(
    error: type[Exception],
) -> None:
    # One base lets a frontend catch everything agglpy raises on purpose;
    # ValueError keeps `except ValueError` in user code working.
    assert issubclass(error, AgglpyError)
    assert issubclass(error, ValueError)


def test_file_format_error_is_not_a_table_error() -> None:
    # A bad file and a bad table built in memory are told apart; the
    # file error chains the table error as its cause instead.
    assert not issubclass(FileFormatError, TableError)


def test_particle_table_error_is_a_table_error() -> None:
    # `except TableError` catches every table problem, particle tables
    # included.
    assert issubclass(ParticleTableError, TableError)


@pytest.mark.parametrize(
    "warning",
    [
        DuplicateParticlesWarning,
        NoParticlesWarning,
        ValuesNotCountedWarning,
        MissingImagesWarning,
        PixelSizeWarning,
        NonCircularParticlesWarning,
        ParticlesOutsideImageWarning,
    ],
)
def test_warnings_are_user_warnings(warning: type[Warning]) -> None:
    assert issubclass(warning, UserWarning)


def test_image_error_is_not_a_table_error() -> None:
    # A bad image array and a bad table are told apart.
    assert not issubclass(ImageError, TableError)


@pytest.mark.parametrize(
    ("error", "os_error"),
    [
        (OutputExistsError, FileExistsError),
        (OutputLockedError, PermissionError),
    ],
)
def test_output_errors_are_agglpy_and_os_errors(
    error: type[Exception], os_error: type[OSError]
) -> None:
    # `except FileExistsError` / `except PermissionError` in user code
    # keeps working; `except AgglpyError` catches them too.
    assert issubclass(error, AgglpyError)
    assert issubclass(error, os_error)
    assert not issubclass(error, ValueError)
