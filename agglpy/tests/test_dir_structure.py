import shutil
from pathlib import Path
import re

import pytest

from agglpy.dir_structure import (
    init_mgr_dirstruct,
    validate_mgr_dirstruct,
    is_mgr_dirstruct,
)
from agglpy.errors import DirectoryStructureError
from agglpy.tests.fixtures import (
    input_multi_raw_wdir,
    input_multi_wdir,
    tests_dir,
)


def test_validate_mgr_dirstruct(
    input_multi_wdir: Path,
    input_multi_raw_wdir: Path,
):
    # Case 1: valid directory structure
    validate_mgr_dirstruct(input_multi_wdir)

    # Case 2: invalid directory structure
    expected_err_msg = (
        r"Image Data Set directory WindowsPath('d:/DATA/library/"
        r"programming libraries/agglpy/agglpy/tests/input/multiple_image_raw/"
        r"images/D7-019') not found"
    )
    escaped_err_msg = re.escape(expected_err_msg)
    with pytest.raises(
        DirectoryStructureError,
        match=rf"{escaped_err_msg}"
    ):
        validate_mgr_dirstruct(input_multi_raw_wdir)

def test_is_mgr_dirstruct(
    input_multi_wdir: Path,
    input_multi_raw_wdir: Path,
):
    # Case 1: valid directory structure
    assert is_mgr_dirstruct(input_multi_wdir)

    # Case 2: invalid directory structure
    assert not is_mgr_dirstruct(input_multi_raw_wdir)


def test_init_mgr_dirstruct(input_multi_raw_wdir: Path, tests_dir: Path):
    output_dir = tests_dir / "output/init_dirstruct"
    # output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(input_multi_raw_wdir, output_dir)
    init_mgr_dirstruct(output_dir)
    assert is_mgr_dirstruct(output_dir)


def test_init_mgr_dirstruct_no_settings(tests_dir: Path):
    input_dir = tests_dir / "input/multiple_image_raw_no_settings" 
    output_dir = tests_dir / "output/init_dirstruct_no_settings"
    # output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(input_dir, output_dir)
    init_mgr_dirstruct(output_dir)
    assert is_mgr_dirstruct(output_dir)

# Clean up the output files after the test
@pytest.fixture(autouse=True)
def cleanup_output_init_dirstruct(request, tests_dir):
    output_dirs = [
        tests_dir / "output/init_dirstruct",
        tests_dir / "output/init_dirstruct_no_settings",

    ]

    def cleanup():
        for output_dir in output_dirs:
            if output_dir.exists():
                try:
                    shutil.rmtree(output_dir)
                except Exception as e:
                    print(f"Error removing {output_dir}: {e}")
    request.addfinalizer(cleanup)
