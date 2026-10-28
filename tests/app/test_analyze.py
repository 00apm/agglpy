"""col.analyze(): readiness, the results tables and their warnings."""

import logging
import os
import warnings
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from agglpy.app import Collection, _analyze
from agglpy.app._analyze import summary_text
from agglpy.app._sheet import IMAGES_TABLE_COLUMNS, RESERVED_COLUMNS
from agglpy.core.properties import to_physical
from agglpy.errors import (
    NotReadyError,
    OutdatedParticlesWarning,
    ParticlesOutsideRegionWarning,
    PixelSizeWarning,
)
from agglpy.io import read_image_metadata, read_particles, write_particles
from agglpy.pipeline import analyze

from support.collection import DISCS, work, write_disc_image
from support.tiffs import write_zeiss

NAMES = ["D7-017", "D7-018", "D7-019"]


def image_file(col: Collection, name: str) -> Path:
    return col.folder.parent / "images" / f"{name}.tif"


def used_csv(col: Collection, name: str) -> Path:
    folder = work(col, name)
    corrected = folder / "particles_corrected.csv"
    return corrected if corrected.is_file() else folder / "particles_hct.csv"


def test_tables_hold_each_image_in_metres(analysis_col: Collection) -> None:
    res = analysis_col.analyze(progress=False)
    assert res.images.columns.tolist() == [
        *IMAGES_TABLE_COLUMNS,
        "series",
        "flow_rate",
        "note",
    ]
    assert res.images["image"].tolist() == NAMES  # D7-020 excluded
    assert res.images["particle_set"].tolist() == ["hct", "corrected", "hct"]
    assert res.images["pixel_size_source"].tolist() == [
        "images.xlsx",
        "image file",
        "images.xlsx",
    ]
    tag = read_image_metadata(image_file(analysis_col, "D7-018")).pixel_size
    assert tag is not None
    sizes = [2.5e-9, tag, 2.5e-9]
    assert res.images.loc[1, "pixel_size"] == tag  # exactly as read
    assert res.images["pixel_size"].tolist() == pytest.approx(sizes)
    assert res.images["fov_area"].tolist() == pytest.approx(
        [200 * 160 * p**2 for p in sizes]
    )
    assert res.images[["width", "height"]].values.tolist() == [[200, 160]] * 3
    assert res.images["outside_region"].tolist() == [0, 0, 0]
    for name, p in zip(NAMES, sizes, strict=True):
        found = analyze(
            read_particles(used_csv(analysis_col, name)), width=200, height=160
        )
        for table, expected in [
            (res.particles, found.particles),
            (res.agglomerates, found.agglomerates),
        ]:
            got = table[table["image"] == name].drop(columns="image")
            pd.testing.assert_frame_equal(
                got.reset_index(drop=True), to_physical(expected, p)
            )
    for table in (res.images, res.particles, res.agglomerates):
        assert table.index.equals(pd.RangeIndex(len(table)))
    assert res.particles.columns[0] == res.agglomerates.columns[0] == "image"
    # the blank image keeps its row
    assert (res.agglomerates["image"] == "D7-019").sum() == 0
    assert set(IMAGES_TABLE_COLUMNS) <= RESERVED_COLUMNS
    agglpys = [c for c in res.images if c not in ("series", "flow_rate")]
    assert set(agglpys) <= RESERVED_COLUMNS
    assert repr(res) == (
        "Results(3 images, 5 particles, 4 agglomerates; 0 subset steps)"
    )


def test_not_ready_lists_every_image_and_computes_nothing(
    analysis_col: Collection,
) -> None:
    sheet = analysis_col.read_sheet()
    sheet.loc[0, "pixel_size_nm"] = None  # D7-017's TIFF has no tag
    analysis_col.write_sheet(sheet)
    for csv in ("particles_hct.csv", "particles_corrected.csv"):
        (work(analysis_col, "D7-018") / csv).unlink()
    image_file(analysis_col, "D7-019").unlink()
    with pytest.raises(NotReadyError) as info:
        analysis_col.analyze(progress=False)
    lines = str(info.value).splitlines()
    stored = analysis_col.read_sheet().loc[2, "file"]
    assert lines[:4] == [
        "3 of 3 images are not ready for analyze():",
        "  D7-017: no pixel size",
        "  D7-018: no particles",
        f"  D7-019: image file not found: {stored}",
    ]
    assert " ".join(lines[4:]) == (
        "Fill pixel_size_nm in images.xlsx; run detect() or "
        "import_particles(); fix the image file or its path in "
        "images.xlsx; set include to false to leave an image out. "
        "col.status() shows every image."
    )
    status = analysis_col.status().set_index("image")
    assert status.loc["D7-017", "pixel_size_source"] == "none"
    assert status.loc["D7-018", "particle_set"] == "none"
    assert status.loc["D7-019", "file"] == "missing"


def test_unreadable_files_are_not_ready(analysis_col: Collection) -> None:
    image_file(analysis_col, "D7-017").write_bytes(b"junk")
    (work(analysis_col, "D7-018") / "particles_corrected.csv").write_text(
        "junk\n"
    )
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(NotReadyError) as info:
            analysis_col.analyze(progress=False)
    # "particles_corrected.csv does not read" is in the error already
    assert seen == []
    text = str(info.value)
    assert text.startswith("2 of 3 images are not ready")
    stored = analysis_col.read_sheet().loc[0, "file"]
    assert f"D7-017: image file does not read: {stored}\n" in text
    assert "D7-018: particles_corrected.csv does not read" in text
    assert "import the particles again or delete the file" in text


def test_no_included_image(analysis_col: Collection) -> None:
    sheet = analysis_col.read_sheet()
    sheet["include"] = False
    analysis_col.write_sheet(sheet)
    with pytest.raises(
        NotReadyError,
        match=r"no image to analyse: every row of images\.xlsx has include "
        "false",
    ):
        analysis_col.analyze(progress=False)


def test_analyze_writes_nothing(
    analysis_col: Collection, tmp_path: Path
) -> None:
    def files() -> dict[Path, tuple[bytes, int]]:
        return {
            p: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in sorted(tmp_path.rglob("*"))
            if p.is_file()
        }

    before = files()
    analysis_col.analyze(progress=False)
    assert files() == before


def test_exception_names_the_image(
    analysis_col: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(particles: pd.DataFrame, **options: object) -> object:
        if len(particles) == 3:  # D7-018's correction
            raise RuntimeError("boom")
        return analyze(particles, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(_analyze, "analyze_image", broken)
    with pytest.raises(RuntimeError, match="boom") as info:
        analysis_col.analyze(progress=False)
    assert "while analysing image D7-018" in info.value.__notes__


def test_user_columns_typed_over_analysed_rows(
    analysis_col: Collection,
) -> None:
    # D7-020's empty cells (include false) don't make flow_rate float
    res = analysis_col.analyze(progress=False)
    assert res.images["flow_rate"].dtype == "int64"
    assert res.images["flow_rate"].tolist() == [2500, 2500, 4500]
    assert res.images["series"].tolist() == ["a", "a", "b"]
    assert res.images["note"].tolist() == ["", "", ""]


# ---------------------------------------------------------------------
# Warnings, progress, logging
# ---------------------------------------------------------------------


def test_outside_region_warned_once_at_the_callers_line(
    analysis_col: Collection, tmp_path: Path
) -> None:
    # the region keeps rows 0-79: D7-017's disc at y 110.7 lies outside
    analysis_col.set_params(None, region={"bottom_fraction": 0.5})
    four = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "x": [40.0, 58.0, 120.0, 100.0],
            "y": [40.0, 40.0, 60.0, 140.0],
            "r": [10.0, 10.0, 8.0, 8.0],
        }
    )
    source = tmp_path / "fiji" / "D7-018_v2.csv"
    write_particles(four, source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis_col.import_particles("D7-018", source)
    with pytest.warns(UserWarning) as ws:
        res = analysis_col.analyze(progress=False)
    outside = [w for w in ws if w.category is ParticlesOutsideRegionWarning]
    assert len(outside) == 1
    assert os.path.normcase(outside[0].filename) == os.path.normcase(__file__)
    text = str(outside[0].message)
    assert text.startswith("2 images: ParticlesOutsideRegionWarning")
    assert "\n  D7-017: " in text and "\n  D7-018: " in text
    # the region change made the detections outdated: that comes first
    assert ws[0].category is OutdatedParticlesWarning
    assert res.images["outside_region"].tolist() == [1, 1, 0]
    (stored,) = res._origin.images["D7-017"]["warnings"]
    assert stored.startswith("ParticlesOutsideRegionWarning: ")
    assert f"\n  D7-017: {stored.split(': ', 1)[1]}\n" in text


def test_outdated_detection_warns_first_and_still_returns(
    analysis_col: Collection,
) -> None:
    analysis_col.set_params(
        "D7-017", hct={"ranges": [{"d_min": 10, "d_max": 40}]}
    )
    write_disc_image(
        analysis_col.folder.parent / "images" / "D7-019.tif", DISCS["D7-018"]
    )
    # a changed detection under a correction is not named
    analysis_col.set_params(
        "D7-018", hct={"ranges": [{"d_min": 10, "d_max": 40}]}
    )
    with pytest.warns(UserWarning) as ws:
        res = analysis_col.analyze(progress=False)
    assert ws[0].category is OutdatedParticlesWarning
    assert os.path.normcase(ws[0].filename) == os.path.normcase(__file__)
    assert str(ws[0].message) == (
        "2 images were analysed with particles from an outdated detection:\n"
        "  D7-017: params changed since detect()\n"
        "  D7-019: image file changed since detect()\n"
        "These results do not reflect the current params. Run col.detect() "
        "to update them, then analyze() again."
    )
    assert len(res.images) == 3
    assert res._origin.images["D7-019"]["detection"] == "image changed"


def test_switched_off_detection_reason(analysis_col: Collection) -> None:
    analysis_col.set_params("D7-017", hct={"ranges": []})
    with pytest.warns(OutdatedParticlesWarning) as ws:
        analysis_col.analyze(progress=False)
    assert str(ws[0].message).startswith(
        "1 image was analysed with particles from an outdated detection:\n"
        "  D7-017: detection was switched off in agglpy.yml after detect(); "
        "set hct again or import_particles()\n"
    )


def test_pixel_size_warning_only_when_the_metadata_decides(
    analysis_col: Collection,
) -> None:
    # the Zeiss text field (6 nm) disagrees with the raw value (5.12 nm)
    write_zeiss(
        analysis_col.folder.parent / "images" / "D7-018.tif",
        200,
        160,
        raw=1e-9,
        image_pixel_size=(6.0, "nm"),
    )
    with pytest.warns(PixelSizeWarning) as ws:
        res = analysis_col.analyze(progress=False)
    (w,) = [w for w in ws if w.category is PixelSizeWarning]
    assert str(w.message).startswith("1 image: PixelSizeWarning")
    assert "\n  D7-018: " in str(w.message)
    assert res._origin.images["D7-018"]["warnings"][0].startswith(
        "PixelSizeWarning: "
    )
    sheet = analysis_col.read_sheet()
    sheet.loc[1, "pixel_size_nm"] = 6.0
    analysis_col.write_sheet(sheet)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        res = analysis_col.analyze(progress=False)
    assert [w.category for w in seen] == []
    assert res._origin.images["D7-018"]["warnings"] == []


class _Bar:
    """Stands in for tqdm: records what analyze shows."""

    made: ClassVar[list["_Bar"]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.descriptions: list[str] = []
        self.updates = 0
        self.closed = False
        _Bar.made.append(self)

    def set_description(self, text: str) -> None:
        self.descriptions.append(text)

    def update(self, n: int) -> None:
        self.updates += n

    def close(self) -> None:
        self.closed = True


def test_progress_and_log_lines(
    analysis_col: Collection,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis_col.analyze(progress=False)
    assert capsys.readouterr().err == ""
    _Bar.made = []
    monkeypatch.setattr(_analyze, "tqdm", _Bar)
    caplog.set_level(logging.INFO, logger="agglpy")
    analysis_col.analyze()
    (bar,) = _Bar.made
    assert bar.kwargs == {"total": 3, "disable": False, "unit": "image"}
    assert bar.descriptions == NAMES
    assert bar.updates == 3 and bar.closed
    tag = read_image_metadata(
        analysis_col.folder.parent / "images" / "D7-018.tif"
    ).pixel_size
    assert tag is not None
    assert [r.getMessage() for r in caplog.records] == [
        "D7-017: 2 particles (hct), 2 agglomerates, 0 outside the region; "
        "pixel size 2.5 nm from images.xlsx",
        "D7-018: 3 particles (corrected), 2 agglomerates, 0 outside the "
        f"region; pixel size {tag * 1e9:.12g} nm from image file",
        "D7-019: 0 particles (hct), 0 agglomerates, 0 outside the region; "
        "pixel size 2.5 nm from images.xlsx",
        "analysed 3 images",
    ]


def test_summary_text_lists_ten_then_more() -> None:
    text = summary_text(
        ParticlesOutsideRegionWarning, [(f"I{i}", "m") for i in range(12)]
    )
    lines = text.splitlines()
    assert lines[0] == (
        "12 images: ParticlesOutsideRegionWarning (each message is in "
        "res.export's record.yml)"
    )
    assert lines[1:3] == ["  I0: m", "  I1: m"]
    assert lines[-1] == "  and 2 more"
    assert "  I9: m" in lines and "  I10: m" not in lines
