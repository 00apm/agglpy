from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agglpy.manager import Manager, PSD_space


def test_manager_init_without_data_sets(input_multi_wdir: Path):
    """With init_data_sets=False, settings are loaded but no ImgDataSets are built."""
    M = Manager(working_dir=input_multi_wdir, init_data_sets=False)

    assert M.working_dir == input_multi_wdir
    assert M.collector_threshold == 0.5
    assert M._settings["data"]["exclude_images"] == ["D7-021", "D7-023"]
    assert M._DS_paths == []
    assert M._DS == []


# def test_find_dataset_paths(input_multi_wdir: Path):
#     M = Manager(working_dir=input_multi_wdir, init_data_sets=False)
#     paths = M._find_datasets_paths(ignore=True)

#     expected = [
#         input_multi_wdir / "images/D7-017/D7-017.tif",
#         input_multi_wdir / "images/D7-019/D7-019.tif",
#     ]
#     assert paths == expected

def test_manager_init_creates_data_sets(input_multi_wdir: Path):
    """Full init builds one ImgDataSet per image, skipping exclude_images."""
    M = Manager(working_dir=input_multi_wdir, init_data_sets=True)

    # D7-021 and D7-023 are listed in exclude_images of the fixture settings
    assert M._DS_paths == [
        input_multi_wdir / "images/D7-017/D7-017.tif",
        input_multi_wdir / "images/D7-019/D7-019.tif",
    ]
    assert [ds.name for ds in M._DS] == ["D7-017", "D7-019"]


# ----------- PSD / summary bookkeeping (roadmap 1.2)
# These tests build the Manager without ImgDataSets and fill the batch tables
# by hand, so they exercise only the Manager's own logic, not detection.

PARTICLE_D = [1.2, 2.5, 3.1, 7.8]
AGL_D = [2.0, 4.5, 9.0]


@pytest.fixture
def mgr(input_multi_wdir: Path) -> Manager:
    """Manager with settings loaded but no ImgDataSets.

    The fixture settings define PSD_space: start 0, end 10, periods 20, linear.
    """
    return Manager(working_dir=input_multi_wdir, init_data_sets=False)


def test_set_PSD_space_stores_bins(mgr: Manager):
    """set_PSD_space() must store the bins, not only return them."""
    expected = PSD_space(start=0, end=10, periods=20, log=False, step=False)

    space = mgr.set_PSD_space()

    np.testing.assert_array_equal(space, expected)
    np.testing.assert_array_equal(mgr._PSD_space, expected)


def test_generate_PSD_uses_settings_space(mgr: Manager):
    """Without an explicit PSD_space, bin by settings.analysis.PSD_space."""
    mgr.batch_res_pDF = pd.DataFrame({"D": PARTICLE_D})

    mgr.generate_PSD(plot=False)

    assert len(mgr.batch_res_PSD) == 20
    assert mgr.batch_res_PSD["counts"].sum() == len(PARTICLE_D)
