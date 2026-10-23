"""Replace a file safely: write a temporary file, then swap it in."""

import contextlib
import os
from collections.abc import Callable
from pathlib import Path

from agglpy.errors import OutputLockedError

__all__ = ["replace_file"]


def replace_file(path: Path, write: Callable[[Path], object]) -> None:
    """Write ``path`` through ``write(temporary_path)``.

    The temporary file sits in the same folder, so the final swap is
    one rename: a failed write or a file locked by another program
    (Excel locks an open workbook) leaves the original as it was.

    Raises:
        OutputLockedError: If ``path`` can't be replaced.
    """
    tmp = path.with_name(f".{path.name}.agglpy-tmp")
    try:
        write(tmp)
        try:
            os.replace(tmp, path)
        except PermissionError:
            raise OutputLockedError(
                f"{path} can't be replaced: close {path.name} and run again"
            ) from None
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
