from aggl import ImgAgl
import pytest
import pathlib


@pytest.fixture
def case_dir():
    return pathlib.Path(__file__).parent / "input" / "single_image"

@pytest.fixture
def model_IA(case_dir):
    return ImgAgl(str(case_dir))


def test_hasSettingsFile(case_dir):
    assert (case_dir / "names.csv").exists()

def test_csvFittingFileExists(case_dir):
    csv_file_path = case_dir / "test_fitting.csv"
    assert csv_file_path.exists()

def test_generateEmptyImgAgl(model_IA):
    print(model_IA._all_P_DF)
    assert 0
    