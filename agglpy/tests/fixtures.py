import pathlib

import pytest

from agglpy.aggl import ImgAgl


@pytest.fixture
def case_dir():
    return pathlib.Path(__file__).parent / "input" / "single_image"


@pytest.fixture
def model_IA(case_dir):
    return ImgAgl(str(case_dir))
