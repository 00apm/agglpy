from pathlib import Path

from agglpy.manager import Manager


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
