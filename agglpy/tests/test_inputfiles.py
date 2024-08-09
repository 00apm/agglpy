import pathlib

import pytest

from ..aggl import ImgAgl
from .fixtures import case_dir, model_IA


def test_hasSettingsFile(case_dir):
    assert (case_dir / "names.csv").exists()


def test_csvFittingFileExists(case_dir):
    csv_file_path = case_dir / "test_fitting.csv"
    assert csv_file_path.exists()
