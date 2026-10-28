"""col.analyze(): readiness, the results tables and their warnings."""

from pathlib import Path

import pandas as pd
import pytest

from agglpy.app import Collection, _analyze
from agglpy.app._sheet import IMAGES_TABLE_COLUMNS, RESERVED_COLUMNS
from agglpy.core.properties import to_physical
from agglpy.errors import NotReadyError
from agglpy.io import read_image_metadata, read_particles
from agglpy.pipeline import analyze

from support.collection import work

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
    with pytest.warns(UserWarning), pytest.raises(NotReadyError) as info:
        analysis_col.analyze(progress=False)
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
