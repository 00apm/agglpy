"""col[name]: the Image handle."""

from pathlib import Path

import pandas as pd
import pytest

import agglpy.app
from agglpy.app import Collection, Image
from agglpy.errors import NoParticleSetError, NoParticlesWarning
from agglpy.io import read_image_metadata, read_particles

from support.collection import DISCS, HCT, work, write_disc_image
from support.tiffs import write_zeiss


def test_handle_reads_files_on_access(particles_col: Collection) -> None:
    img = particles_col["D7-017"]
    assert isinstance(img, Image)
    assert img.name == "D7-017"
    assert img.particle_set is None
    assert repr(img) == "Image('D7-017', particle_set=None)"
    with pytest.raises(
        NoParticleSetError,
        match=r"D7-017 has no particles: run detect\(\) or import_particles",
    ):
        img.particles
    assert img.detection is None
    particles_col.detect(progress=False)
    # the same object sees the new files
    assert img.particle_set == "hct"
    assert repr(img) == "Image('D7-017', particle_set='hct')"
    assert len(img.particles) == 2
    assert img.detection is not None
    assert img.detection["kind"] == "detection"
    assert img.detection["particles"]["count"] == 2
    assert type(img.detection["particles"]["file"]) is str
    assert img.correction is None


def test_corrected_set_wins(particles_col: Collection, fiji_csv: Path) -> None:
    particles_col.detect(progress=False)
    particles_col.import_particles("D7-017", fiji_csv)
    img = particles_col["D7-017"]
    assert img.particle_set == "corrected"
    pd.testing.assert_frame_equal(
        img.particles,
        read_particles(
            work(particles_col, "D7-017") / "particles_corrected.csv"
        ),
    )
    assert img.correction is not None
    assert img.correction["source"]["format"] == "imagej"


def test_record_without_csv_or_unreadable_is_none(
    particles_col: Collection,
) -> None:
    particles_col.detect(progress=False)
    folder = work(particles_col, "D7-017")
    (folder / "particles_hct_record.yml").write_text("format: 2\n")
    with pytest.warns(UserWarning):
        assert particles_col["D7-017"].detection is None
    folder = work(particles_col, "D7-018")
    (folder / "particles_hct.csv").unlink()
    assert particles_col["D7-018"].detection is None


def test_unknown_name_is_key_error(particles_col: Collection) -> None:
    with pytest.raises(KeyError):
        particles_col["X"]


def test_excluded_image_still_gives_a_handle(
    particles_col: Collection,
) -> None:
    sheet = particles_col.read_sheet()
    sheet.loc[0, "include"] = False
    particles_col.write_sheet(sheet)
    assert particles_col["D7-017"].particle_set is None


def test_pixel_size_in_metres(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    write_disc_image(folder / "A.tif", DISCS["D7-017"])
    # Zeiss: 1 nm at 1024 px is 16 nm at 64 px
    zeiss = write_zeiss(folder / "B.tif", 64, 48, raw=1e-9)
    write_disc_image(folder / "C.tif", DISCS["D7-018"])
    col = Collection.create(tmp_path / "col", images=folder)
    sheet = col.read_sheet()
    sheet.loc[0, "pixel_size_nm"] = 2.5
    col.write_sheet(sheet)
    from_tag = read_image_metadata(zeiss).pixel_size
    assert col["A"].pixel_size == pytest.approx(2.5e-9)
    assert col["B"].pixel_size == from_tag  # exactly as read
    assert col["C"].pixel_size is None
    # the image gone: the copy in the particle record
    col.set_params(None, hct=HCT)
    with pytest.warns(NoParticlesWarning):
        col.detect(["B"], progress=False)
    zeiss.unlink()
    assert col["B"].pixel_size == pytest.approx(from_tag)


def test_params(particles_col: Collection) -> None:
    assert particles_col["D7-017"].params == particles_col.params("D7-017")


def test_exported_names() -> None:
    assert "Image" in agglpy.app.__all__
