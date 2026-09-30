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


def test_generate_aglPSD_before_generate_PSD(mgr: Manager):
    """generate_aglPSD must not rely on generate_PSD having set _PSD_space."""
    mgr.batch_res_aglDF = pd.DataFrame({"D": AGL_D})

    mgr.generate_aglPSD(plot=False)

    assert len(mgr.batch_res_aglPSD) == 20
    assert mgr.batch_res_aglPSD["counts"].sum() == len(AGL_D)


def test_generate_aglPSD_builds_missing_agl_table(mgr: Manager, monkeypatch):
    """If the agglomerate table was never generated (None), build it first."""
    calls = []

    def fake_generate_aglTable():
        calls.append(True)
        mgr.batch_res_aglDF = pd.DataFrame({"D": AGL_D})
        return mgr.batch_res_aglDF

    monkeypatch.setattr(mgr, "generate_aglTable", fake_generate_aglTable)

    mgr.generate_aglPSD(plot=False)

    assert calls == [True]
    assert mgr.batch_res_aglPSD["counts"].sum() == len(AGL_D)


def test_generate_aglPCD_builds_missing_agl_table(mgr: Manager, monkeypatch):
    """Same as above for the primary-particle count distribution."""
    calls = []

    def fake_generate_aglTable():
        calls.append(True)
        mgr.batch_res_aglDF = pd.DataFrame({"members_count": [2, 3, 3, 8]})
        return mgr.batch_res_aglDF

    monkeypatch.setattr(mgr, "generate_aglTable", fake_generate_aglTable)

    mgr.generate_aglPCD(PCD_space=np.array([0, 2, 5, 10]), plot=False)

    assert calls == [True]
    assert mgr.batch_res_aglPCD["counts"].tolist() == [1, 2, 1]


def test_get_PSD_first_call(mgr: Manager):
    """get_PSD() works before any PSD table exists (batch_res_PSD is None)."""
    mgr.batch_res_pDF = pd.DataFrame({"D": PARTICLE_D})

    out = mgr.get_PSD()

    assert list(out.columns) == ["left", "counts"]
    assert out["counts"].sum() == len(PARTICLE_D)


def test_generate_summary_builds_missing_DSsummary(mgr: Manager, monkeypatch):
    """If the per-DataSet summary was never generated (None), build it first."""
    ds_summary = pd.DataFrame(
        {
            "DS ID": ["A", "B"],
            "N_primary_particle": [10, 30],
            "N_aerosol_particle": [5, 15],
            "N_pp1": [3, 7],
            "N_ppA": [7, 23],
            "N_agl": [2, 8],
            "N_collector_agl": [1, 3],
            "N_similar_agl": [1, 5],
            "N_pp1_separate": [3, 7],
        }
    )

    def fake_generate_DSsummary():
        mgr.batch_res_DSsummary = ds_summary

    monkeypatch.setattr(mgr, "generate_DSsummary", fake_generate_DSsummary)
    mgr.batch_res_pDF = pd.DataFrame({"D": PARTICLE_D})
    mgr.batch_res_aglDF = pd.DataFrame(
        {"D": AGL_D, "members_count": [2, 3, 8]}
    )

    mgr.generate_summary()

    summary = mgr.batch_res_summary[0]
    assert summary["N_primary_particle"] == 40
    assert summary["N_agl"] == 10
    assert summary["n_ppP"] == pytest.approx(40 / 20)
