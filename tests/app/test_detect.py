"""Collection.detect: decisions per image, files written, the report."""

import logging
import os
import stat
import warnings
from pathlib import Path
from typing import ClassVar

import cv2
import pandas as pd
import pytest
import tifffile

from agglpy.app import Collection, _files
from agglpy.app import _detect as detect_module
from agglpy.app._records import DetectionRecord, read_record, sha256
from agglpy.errors import (
    DuplicateParticlesWarning,
    IncompleteRunWarning,
    LossyImageWarning,
    NoParticlesWarning,
    ParamsError,
)

from support.collection import (
    DISCS,
    HCT,
    work,
    write_blank_image,
    write_disc_image,
)
from support.discs import draw_discs

COLUMNS = ["image", "outcome", "particle_count", "message", "warnings"]
NAMES = ["D7-017", "D7-018", "D7-019"]
OTHER_DISCS = [(100.2, 80.3, 14.2)]
# the batch summary has its own tests below
NOT_DONE = pytest.mark.filterwarnings(
    "ignore::agglpy.errors.IncompleteRunWarning"
)


def csv(col: Collection, name: str) -> Path:
    return work(col, name) / "particles_hct.csv"


def record_path(col: Collection, name: str) -> Path:
    return work(col, name) / "particles_hct_record.yml"


def image(col: Collection, name: str) -> Path:
    return col.folder.parent / "images" / f"{name}.tif"


def test_first_detect_writes_csv_and_record_per_image(
    particles_col: Collection,
) -> None:
    rep = particles_col.detect(progress=False)
    assert rep.columns.tolist() == COLUMNS
    assert rep["image"].tolist() == NAMES
    assert rep["outcome"].tolist() == ["detected"] * 3
    assert rep["particle_count"].tolist() == [2, 2, 2]
    assert rep["message"].tolist() == ["", "", ""]
    assert rep["warnings"].tolist() == ["", "", ""]
    assert str(rep["particle_count"].dtype) == "Int64"
    assert sorted(p.name for p in work(particles_col, "D7-017").iterdir()) == [
        "particles_hct.csv",
        "particles_hct_record.yml",
    ]
    rec = read_record(record_path(particles_col, "D7-017"), DetectionRecord)
    assert rec is not None
    assert rec.params == particles_col.params("D7-017")
    assert rec.particles.count == 2
    assert rec.image.file == "../images/D7-017.tif"
    assert rec.particles.sha256 == sha256(csv(particles_col, "D7-017"))


def test_second_detect_skips_up_to_date(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    times = [csv(particles_col, n).stat().st_mtime_ns for n in NAMES]
    rep = particles_col.detect(progress=False)
    assert rep["outcome"].tolist() == ["skipped"] * 3
    assert rep["message"].tolist() == ["up to date"] * 3
    assert rep["particle_count"].isna().all()
    assert [csv(particles_col, n).stat().st_mtime_ns for n in NAMES] == times


def test_force_redetects_up_to_date(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    rep = particles_col.detect(force=True, progress=False)
    assert rep["outcome"].tolist() == ["detected"] * 3
    assert rep["message"].tolist() == ["was up to date"] * 3


def test_params_changed_redetects(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    particles_col.set_params(
        "D7-018", hct={"ranges": [{"d_min": 12, "d_max": 40}]}
    )
    rep = particles_col.detect(progress=False)
    assert rep["outcome"].tolist() == ["skipped", "detected", "skipped"]
    rec = read_record(record_path(particles_col, "D7-018"), DetectionRecord)
    assert rec is not None
    assert rec.params == particles_col.params("D7-018")


def test_image_changed_redetects(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    write_disc_image(image(particles_col, "D7-019"), OTHER_DISCS)
    rep = particles_col.detect(progress=False)
    assert rep["outcome"].tolist() == ["skipped", "skipped", "detected"]
    assert rep["particle_count"].tolist()[2] == 1


@NOT_DONE
def test_hct_not_set_is_skipped(tmp_path: Path, images: Path) -> None:
    col = Collection.create(tmp_path / "col", images=images)
    rep = col.detect(progress=False)
    assert rep["outcome"].tolist() == ["skipped"] * 3
    assert (
        rep["message"].tolist()
        == ["detection not set: hct.ranges is empty (set_params)"] * 3
    )
    assert list((col.folder / "agglpy").iterdir()) == []


@NOT_DONE
def test_missing_image_is_an_error_row(particles_col: Collection) -> None:
    image(particles_col, "D7-018").unlink()
    rep = particles_col.detect(progress=False)
    assert rep["outcome"].tolist() == ["detected", "error", "detected"]
    assert rep["message"][1] == "image file not found: ../images/D7-018.tif"
    assert not work(particles_col, "D7-018").exists()


@NOT_DONE
def test_unreadable_image_is_an_error_row(particles_col: Collection) -> None:
    image(particles_col, "D7-018").write_bytes(b"no image")
    rep = particles_col.detect(progress=False)
    assert rep["outcome"].tolist() == ["detected", "error", "detected"]
    assert "D7-018.tif" in rep["message"][1]
    assert not work(particles_col, "D7-018").exists()


@NOT_DONE
def test_correction_skips_and_force_keeps_it(
    particles_col: Collection,
) -> None:
    particles_col.detect(progress=False)
    corrected = work(particles_col, "D7-017") / "particles_corrected.csv"
    corrected.write_bytes(b"id,x,y,r\n1,10,10,5\n")
    rep = particles_col.detect(progress=False)
    assert rep["outcome"].tolist() == ["skipped"] * 3
    assert rep["message"][0] == (
        "has a correction; force=True re-detects (the correction is kept)"
    )
    rep = particles_col.detect(force=True, progress=False)
    assert rep["outcome"][0] == "detected"
    assert rep["message"][0] == "the correction is kept and stays in use"
    assert corrected.read_bytes() == b"id,x,y,r\n1,10,10,5\n"


def _edit_csv(col: Collection) -> None:
    with csv(col, "D7-017").open("a", encoding="utf-8") as fh:
        fh.write("3,10.5,10.5,5.0\n")


def _delete_record(col: Collection) -> None:
    record_path(col, "D7-017").unlink()


def _semicolon_csv(col: Collection) -> None:
    csv(col, "D7-017").write_text("id;x;y;r\n1;10;10;5\n", encoding="utf-8")


@NOT_DONE
@pytest.mark.filterwarnings("ignore:UserWarning in D7-017")
@pytest.mark.parametrize(
    ("damage", "word", "past"),
    [
        (_edit_csv, "was edited by hand", "was edited by hand"),
        (_delete_record, "has no record", "had no record"),
        (_semicolon_csv, "does not read", "did not read"),
    ],
    ids=["edited", "no-record", "unreadable"],
)
def test_persons_file_is_kept_then_replaced_with_force(
    particles_col: Collection, damage, word: str, past: str
) -> None:
    particles_col.detect(progress=False)
    damage(particles_col)
    before = csv(particles_col, "D7-017").read_bytes()
    rep = particles_col.detect(progress=False)
    assert rep["outcome"].tolist() == ["kept", "skipped", "skipped"]
    assert rep["message"][0] == (
        f"particles_hct.csv {word}; force=True replaces it"
    )
    assert csv(particles_col, "D7-017").read_bytes() == before
    rep = particles_col.detect(force=True, progress=False)
    assert rep["outcome"][0] == "replaced"
    assert rep["message"][0] == f"replaced particles_hct.csv, which {past}"
    rec = read_record(record_path(particles_col, "D7-017"), DetectionRecord)
    assert rec is not None
    assert rec.particles.sha256 == sha256(csv(particles_col, "D7-017"))


def test_zero_particles_is_a_result(particles_col: Collection) -> None:
    write_blank_image(image(particles_col, "D7-017"))
    with pytest.warns(NoParticlesWarning):
        rep = particles_col.detect(progress=False)
    assert rep["outcome"][0] == "detected"
    assert rep["particle_count"][0] == 0
    assert csv(particles_col, "D7-017").read_bytes() == b"id,x,y,r\n"


def test_selection_by_name_and_include(particles_col: Collection) -> None:
    table = particles_col.read_sheet()
    table.loc[1, "include"] = False
    particles_col.write_sheet(table)
    rep = particles_col.detect(progress=False)
    assert rep["image"].tolist() == ["D7-017", "D7-019"]
    rep = particles_col.detect("D7-018", progress=False)
    assert rep["image"].tolist() == ["D7-018"]
    assert rep["outcome"].tolist() == ["detected"]
    rep = particles_col.detect(
        ["D7-019", "D7-017", "D7-019"], force=True, progress=False
    )
    assert rep["image"].tolist() == ["D7-017", "D7-019"]


def test_unknown_name_raises_before_running(
    particles_col: Collection,
) -> None:
    message = r"no image 'X', 'Y' in images\.xlsx"
    with pytest.raises(ParamsError, match=message):
        particles_col.detect(["D7-017", "X", "Y"], progress=False)
    assert list((particles_col.folder / "agglpy").iterdir()) == []


def test_broken_yml_raises(particles_col: Collection) -> None:
    path = particles_col.folder / "agglpy.yml"
    text = path.read_text(encoding="utf-8")
    assert "d_max: 44" in text
    path.write_text(text.replace("d_max: 44", "d_max: many"), "utf-8")
    with pytest.raises(ParamsError, match="d_max"):
        particles_col.detect(progress=False)
    assert list((particles_col.folder / "agglpy").iterdir()) == []


def test_sources_untouched(particles_col: Collection) -> None:
    paths = [image(particles_col, n) for n in NAMES]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    for p in paths:
        os.chmod(p, stat.S_IREAD)
    try:
        rep = particles_col.detect(progress=False)
    finally:
        for p in paths:
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
    assert rep["outcome"].tolist() == ["detected"] * 3
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == before


@NOT_DONE
def test_interrupted_write_shows_as_edited_then_force_redetects(
    particles_col: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    particles_col.detect(progress=False)
    # a new image gives a new CSV; the record write then fails
    write_disc_image(image(particles_col, "D7-017"), OTHER_DISCS)

    def fail(path: Path, record: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(detect_module, "write_record", fail)
    rep = particles_col.detect(progress=False)
    assert rep["outcome"][0] == "error"
    assert rep["message"][0] == "OSError: disk full"
    monkeypatch.undo()
    rep = particles_col.detect(progress=False)
    assert rep["outcome"][0] == "kept"
    assert rep["message"][0] == (
        "particles_hct.csv was edited by hand; force=True replaces it"
    )
    rep = particles_col.detect(force=True, progress=False)
    assert rep["outcome"][0] == "replaced"


@NOT_DONE
def test_resume_after_an_error(
    particles_col: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = detect_module.detect_particles
    calls = []

    def second_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(detect_module, "detect_particles", second_fails)
    rep = particles_col.detect(progress=False)
    assert rep["outcome"].tolist() == ["detected", "error", "detected"]
    assert rep["message"][1] == "RuntimeError: boom"
    assert pd.isna(rep["particle_count"][1])
    monkeypatch.undo()
    rep = particles_col.detect(progress=False)
    assert rep["outcome"].tolist() == ["skipped", "detected", "skipped"]
    assert rep["message"].tolist() == ["up to date", "", "up to date"]


def test_detect_returns_empty_report_for_empty_selection(
    particles_col: Collection,
) -> None:
    table = particles_col.read_sheet()
    table["include"] = False
    particles_col.write_sheet(table)
    rep = particles_col.detect(progress=False)
    assert rep.columns.tolist() == COLUMNS
    assert len(rep) == 0
    assert str(rep["particle_count"].dtype) == "Int64"


# ---------------------------------------------------------------------
# Review focus
# ---------------------------------------------------------------------


@NOT_DONE
def test_locked_csv_is_an_error_row_and_the_batch_continues(
    particles_col: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    particles_col.detect(progress=False)
    locked = csv(particles_col, "D7-018")
    before = (locked.read_bytes(), record_path(particles_col, "D7-018"))
    before_record = before[1].read_bytes()
    real = _files.os.replace

    def replace(src, dst):
        if Path(dst) == locked:
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr(_files.os, "replace", replace)
    rep = particles_col.detect(force=True, progress=False)
    assert rep["outcome"].tolist() == ["detected", "error", "detected"]
    assert "particles_hct.csv" in rep["message"][1]
    assert locked.read_bytes() == before[0]
    assert before[1].read_bytes() == before_record


def test_non_ascii_name_and_path(tmp_path: Path) -> None:
    folder = tmp_path / "próbki 2026"
    write_disc_image(folder / "D7-019 ł.tif", DISCS["D7-019"])
    col = Collection.create(tmp_path / "col", images=folder)
    col.set_params(None, hct=HCT)
    rep = col.detect(progress=False)
    assert rep["image"].tolist() == ["D7-019 ł"]
    assert rep["outcome"].tolist() == ["detected"]
    rec = read_record(
        work(col, "D7-019 ł") / "particles_hct_record.yml", DetectionRecord
    )
    assert rec is not None
    assert rec.image.file == col.read_sheet()["file"][0]
    assert col.detect(progress=False)["message"].tolist() == ["up to date"]


# ---------------------------------------------------------------------
# Warnings, progress, logging
# ---------------------------------------------------------------------

TWO_RANGES = {
    "ranges": [{"d_min": 12, "d_max": 44}, {"d_min": 20, "d_max": 40}]
}


def _is_here(w: warnings.WarningMessage) -> bool:
    return os.path.normcase(w.filename) == os.path.normcase(__file__)


def test_duplicates_go_into_the_record_and_are_reemitted_once(
    particles_col: Collection,
) -> None:
    # two overlapping ranges find the same discs
    particles_col.set_params(None, hct=TWO_RANGES)
    with pytest.warns(DuplicateParticlesWarning) as ws:
        rep = particles_col.detect(progress=False)
    dup = [w for w in ws if w.category is DuplicateParticlesWarning]
    assert len(dup) == 1
    assert str(dup[0].message) == (
        "DuplicateParticlesWarning in D7-017, D7-018, D7-019 (messages in "
        "the returned table and the records)"
    )
    assert _is_here(dup[0])
    assert rep["warnings"].tolist() == ["DuplicateParticlesWarning"] * 3
    rec = read_record(record_path(particles_col, "D7-017"), DetectionRecord)
    assert rec is not None
    assert rec.warnings[0].startswith("DuplicateParticlesWarning: ")


def _jpeg(path: Path, discs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".jpg", draw_discs(160, 200, discs))
    assert ok
    data.tofile(path)
    return path


def test_jpeg_warns_lossy(tmp_path: Path) -> None:
    _jpeg(tmp_path / "images" / "D7-017.jpg", DISCS["D7-017"])
    col = Collection.create(tmp_path / "col", images=tmp_path / "images")
    col.set_params(None, hct=HCT)
    with pytest.warns(LossyImageWarning, match="D7-017") as ws:
        rep = col.detect(progress=False)
    assert _is_here(next(w for w in ws if w.category is LossyImageWarning))
    assert rep["warnings"].tolist() == ["LossyImageWarning"]
    rec = read_record(
        work(col, "D7-017") / "particles_hct_record.yml", DetectionRecord
    )
    assert rec is not None
    assert rec.warnings == (
        "LossyImageWarning: ../images/D7-017.jpg is JPEG-compressed; edges "
        "may shift by the compression artefacts",
    )


def test_jpeg_recognised_by_bytes_not_suffix(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    _jpeg(folder / "D7-017.tif", DISCS["D7-017"])
    tifffile.imwrite(
        folder / "D7-018.jpg", draw_discs(160, 200, DISCS["D7-018"])
    )
    col = Collection.create(tmp_path / "col", images=folder)
    col.set_params(None, hct=HCT)
    with pytest.warns(LossyImageWarning) as ws:
        rep = col.detect(progress=False)
    lossy = [w for w in ws if w.category is LossyImageWarning]
    assert [str(w.message) for w in lossy] == [
        "LossyImageWarning in D7-017 (messages in the returned table and "
        "the records)"
    ]
    assert rep["warnings"].tolist() == ["LossyImageWarning", ""]


def test_incomplete_run_warns_with_counts(particles_col: Collection) -> None:
    image(particles_col, "D7-018").unlink()
    particles_col.set_params("D7-019", hct={"ranges": []})
    with pytest.warns(IncompleteRunWarning) as ws:
        particles_col.detect(progress=False)
    (w,) = [w for w in ws if w.category is IncompleteRunWarning]
    assert str(w.message) == "1 skipped, 1 error; see the returned table"
    assert _is_here(w)


def test_kept_counts_as_incomplete(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    _delete_record(particles_col)
    with pytest.warns(IncompleteRunWarning, match="^1 kept; see"):
        particles_col.detect(progress=False)


def test_up_to_date_skips_alone_stay_quiet(particles_col: Collection) -> None:
    particles_col.detect(progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rep = particles_col.detect(progress=False)
    assert rep["message"].tolist() == ["up to date"] * 3


class _Bar:
    """Stands in for tqdm: records what detect shows."""

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


def test_progress_bar_shows_image_and_step(
    particles_col: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Bar.made = []
    monkeypatch.setattr(detect_module, "tqdm", _Bar)
    particles_col.set_params("D7-018", preprocess={"median_blur": 3})
    particles_col.detect()
    (bar,) = _Bar.made
    assert bar.kwargs == {"total": 3, "disable": False, "unit": "image"}
    assert bar.descriptions == [
        "D7-017 · reading",
        "D7-017 · hct",
        "D7-018 · reading",
        "D7-018 · median",
        "D7-018 · hct",
        "D7-019 · reading",
        "D7-019 · hct",
    ]
    assert bar.updates == 3
    assert bar.closed


def test_progress_counts_skipped_images(
    particles_col: Collection, monkeypatch: pytest.MonkeyPatch
) -> None:
    particles_col.detect(progress=False)
    _Bar.made = []
    monkeypatch.setattr(detect_module, "tqdm", _Bar)
    particles_col.detect()
    assert _Bar.made[0].updates == 3


def test_progress_false_shows_nothing(
    particles_col: Collection, capsys: pytest.CaptureFixture[str]
) -> None:
    particles_col.detect(progress=False)
    assert capsys.readouterr().err == ""


def test_log_lines_per_image(
    particles_col: Collection, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="agglpy")
    particles_col.detect(progress=False)
    lines = [r.getMessage() for r in caplog.records]
    assert lines == [f"{n}: detected " for n in NAMES]


def test_error_traceback_goes_to_the_log(
    particles_col: Collection,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(detect_module, "detect_particles", boom)
    caplog.set_level(logging.INFO, logger="agglpy")
    with pytest.warns(IncompleteRunWarning):
        particles_col.detect("D7-017", progress=False)
    errors = [r for r in caplog.records if r.exc_info]
    assert len(errors) == 1
    assert errors[0].levelno == logging.ERROR
    assert "D7-017" in errors[0].getMessage()


# Review focus
def test_warnings_as_errors_raise_after_the_batch(
    particles_col: Collection,
) -> None:
    particles_col.set_params(None, hct=TWO_RANGES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DuplicateParticlesWarning):
            particles_col.detect(progress=False)
    for name in NAMES:
        assert csv(particles_col, name).is_file()
        assert record_path(particles_col, name).is_file()


def test_csv_with_crlf_line_endings_still_matches(
    particles_col: Collection,
) -> None:
    # git on Windows checks text files out with CRLF line endings
    particles_col.detect(progress=False)
    for name in NAMES:
        path = csv(particles_col, name)
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    rep = particles_col.detect(progress=False)
    assert rep["message"].tolist() == ["up to date"] * 3
    assert particles_col.status()["detection"].tolist() == ["up to date"] * 3
