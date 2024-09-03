import pathlib

import pytest

from ..aggl import ImgAgl


def test_generateEmptyImgAgl(model_IA: ImgAgl):
    print(model_IA._all_P_DF)
    assert 0
