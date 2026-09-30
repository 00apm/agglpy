import re
import shutil
from pathlib import Path

import pytest

from agglpy.dir_structure import (
    init_mgr_dirstruct,
    is_mgr_dirstruct,
    validate_mgr_dirstruct,
)
from agglpy.errors import DirectoryStructureError


def test_validate_mgr_dirstruct_valid_layout_passes(input_multi_wdir: Path):
    """A valid layout passes silently (no exception)."""
    validate_mgr_dirstruct(input_multi_wdir)


def test_validate_mgr_dirstruct_missing_image_dir_raises(input_multi_raw_wdir: Path):
    """The error names the first images/<name>/ dir that is missing."""
    missing_dir = input_multi_raw_wdir / "images" / "D7-019"
    expected_err_msg = f"Image Data Set directory {missing_dir!r} not found"
    with pytest.raises(DirectoryStructureError, match=re.escape(expected_err_msg)):
        validate_mgr_dirstruct(input_multi_raw_wdir)


@pytest.mark.parametrize(
    ("input_dir", "expected"),
    [
        pytest.param("multiple_image", True, id="valid-layout"),
        pytest.param("multiple_image_raw", False, id="flat-dir"),
    ],
)
def test_is_mgr_dirstruct(tests_dir: Path, input_dir, expected):
    """Boolean counterpart of validate_mgr_dirstruct: returns instead of raising."""
    assert is_mgr_dirstruct(tests_dir / "data/input" / input_dir) is expected


@pytest.mark.parametrize(
    "input_dir",
    [
        pytest.param("multiple_image_raw", id="with-settings"),
        pytest.param("multiple_image_raw_no_settings", id="no-settings"),
    ],
)
def test_init_mgr_dirstruct_creates_valid_layout(
    tests_dir: Path, tmp_path: Path, input_dir
):
    """A flat dir of images is reorganised into images/<name>/ subdirs.

    Without a settings.yml, one is created from the found images first.
    """
    wdir = tmp_path / "wdir"
    shutil.copytree(tests_dir / "data/input" / input_dir, wdir)

    init_mgr_dirstruct(wdir)

    assert is_mgr_dirstruct(wdir)
