"""ImgDataSet behaviour that stays stable through the Phase 2 refactor."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agglpy._legacy.cfg import load_manager_settings
from agglpy._legacy.img_ds import ImgDataSet


@pytest.fixture
def d7_017(input_multi_wdir: Path) -> ImgDataSet:
    """D7-017 test image with cropping and one preprocessing step enabled."""
    settings = load_manager_settings(input_multi_wdir / "settings.yml")
    img_settings = settings["data"]["images"]["D7-017"]
    img_settings["HCT_file"] = None  # detect instead of loading a CSV
    img_settings["crop_ratio"] = 0.1
    img_settings["median_blur"] = 3
    img_settings["rolling_ball"] = None  # keep the test fast
    return ImgDataSet(
        input_multi_wdir / "images" / "D7-017",
        settings=img_settings,
        auto_load=False,
    )


def test_detect_primary_particles_is_repeatable(
    d7_017: ImgDataSet, monkeypatch
):
    """A second detection must start from the raw image again.

    It used to overwrite the stored image with the cropped and preprocessed
    one, so every call cropped and preprocessed an already processed image.
    """
    received = []

    def fake_HCT_multi(image, **kwargs):
        # record what detection sees instead of running Hough circles
        received.append(image.copy())
        return pd.DataFrame(
            {"X": [10.0], "Y": [10.0], "R": [5.0]},
            index=pd.Index([0], name="ID"),
        )

    monkeypatch.setattr("agglpy._legacy.img_ds.HCT_multi", fake_HCT_multi)

    d7_017.detect_primary_particles(export_csv=False, export_img=False)
    d7_017.detect_primary_particles(export_csv=False, export_img=False)

    first, second = received
    assert first.shape == second.shape
    np.testing.assert_array_equal(first, second)
