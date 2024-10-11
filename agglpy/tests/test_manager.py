from pathlib import Path

import pytest

from agglpy.manager import Manager
from agglpy.tests.fixtures import (
    input_multi_raw_wdir,
    input_multi_wdir,
)


def test_manager_constr_wo_init(input_multi_raw_wdir: Path):
    """Test checks Manager object without initialization"""
    M = Manager(working_dir=input_multi_raw_wdir, init_data_sets=False)
    # Check attributes
    assert list(M.__dict__.keys()) == [
        "_workdir",
        "_settings_path",
        "_settings",
    ]


# def test_find_dataset_paths(input_multi_wdir: Path):
#     M = Manager(working_dir=input_multi_wdir, init_data_sets=False)
#     paths = M._find_datasets_paths(ignore=True)

#     expected = [
#         input_multi_wdir / "images/D7-017/D7-017.tif",
#         input_multi_wdir / "images/D7-019/D7-019.tif",
#     ]
#     assert paths == expected

def test_manager_constr(input_multi_wdir: Path):
    M = Manager(working_dir=input_multi_wdir, init_data_sets=True)
    assert False 
