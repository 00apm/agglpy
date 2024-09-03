import pathlib

import pytest

from agglpy.aggl import ImgAgl
from agglpy.tests.fixtures import (
    input_multi_raw_wdir,
    manager_obj,
)


def test_manager_construct(manager_obj):
    M = manager_obj
    assert 0


def test_csv_fitting_file_exists(input_multi_raw_wdir):
    csv_file_path = input_multi_raw_wdir / "test_fitting.csv"
    assert csv_file_path.exists()
