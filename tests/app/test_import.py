"""Collection.import_particles and the matching of a correction."""

import os
import shutil
import stat
from datetime import date
from pathlib import Path

import pytest

from agglpy.app import Collection
from agglpy.app._correction import count_kept
from agglpy.app._records import (
    CorrectionRecord,
    DetectionRecord,
    read_record,
    sha256,
)
from agglpy.errors import (
    DuplicateParticlesWarning,
    FileFormatError,
    NonCircularParticlesWarning,
    OutputExistsError,
    ParamsError,
    UnreadableRecordWarning,
)
from agglpy.io import read_particles
from agglpy.tables import make_particles

from support.collection import work
from support.paths import DATA_DIR

FITTING = (
    DATA_DIR
    / "input"
    / "multiple_image"
    / "images"
    / "D7-017"
    / "D7-017_fitting.csv"
)


def corrected(col: Collection) -> Path:
    return work(col, "D7-017") / "particles_corrected.csv"


def corrected_record(col: Collection) -> Path:
    return work(col, "D7-017") / "particles_corrected_record.yml"


def test_count_kept_tolerances() -> None:
    det = make_particles(x=[10, 50, 90], y=[10, 10, 10], r=[5, 5, 5])
    # 1.9 px and r 5.4 match; r 6 vs 5 differs by 1/6 of 6 > 0.1
    cor = make_particles(x=[11.9, 50, 200], y=[10, 10, 10], r=[5.4, 6, 5])
    assert count_kept(det, cor) == 1
    assert count_kept(det, det.iloc[:0]) == 0
    assert count_kept(det.iloc[:0], cor) == 0


def test_count_kept_counts_each_detected_circle_once() -> None:
    det = make_particles(x=[10], y=[10], r=[5])
    with pytest.warns(DuplicateParticlesWarning):
        cor = make_particles(x=[10, 10.5], y=[10, 10], r=[5, 5])
    assert count_kept(det, cor) == 1


def test_import_writes_corrected_csv_and_record(
    particles_col: Collection, fiji_csv: Path
) -> None:
    particles_col.detect(progress=False)
    rep = particles_col.import_particles("D7-017", fiji_csv)
    assert rep.columns.tolist() == [
        "image",
        "outcome",
        "particle_count",
        "message",
        "warnings",
    ]
    assert rep.loc[0, ["image", "outcome", "message"]].tolist() == [
        "D7-017",
        "imported",
        "",
    ]
    assert rep.loc[0, "particle_count"] == 2
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.source.format == "imagej"
    assert rec.source.path == fiji_csv.resolve().as_posix()
    assert rec.source.sha256 == sha256(fiji_csv)
    assert rec.image.file == "../images/D7-017.tif"
    assert rec.particles.sha256 == sha256(corrected(particles_col))
    hct = read_record(
        work(particles_col, "D7-017") / "particles_hct_record.yml",
        DetectionRecord,
    )
    assert hct is not None
    assert rec.started_from is not None
    assert rec.started_from.kept == 2
    assert rec.started_from.detection_count == 2
    assert rec.started_from.detected == hct.created
    assert rec.started_from.sha256 == hct.particles.sha256
    table = read_particles(corrected(particles_col))
    assert table.columns.tolist() == ["id", "x", "y", "r"]
    assert len(table) == 2


def test_import_without_detection_or_image(
    particles_col: Collection, fiji_csv: Path
) -> None:
    (particles_col.folder.parent / "images" / "D7-017.tif").unlink()
    rep = particles_col.import_particles("D7-017", fiji_csv)
    assert rep["outcome"].tolist() == ["imported"]
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.started_from is None
    assert rec.image.file == "../images/D7-017.tif"
    assert rec.image.sha256 is None


def test_reimport_replaces_and_says_what(
    particles_col: Collection, fiji_csv: Path
) -> None:
    particles_col.import_particles("D7-017", fiji_csv)
    rep = particles_col.import_particles("D7-017", fiji_csv)
    assert rep["outcome"].tolist() == ["replaced"]
    assert rep["message"][0] == (
        f"replaced correction from {fiji_csv.resolve().as_posix()} (2 "
        "particles, imported "
        f"{date.today().isoformat()})"
    )


def test_hand_edited_correction_refuses_then_force(
    particles_col: Collection, fiji_csv: Path
) -> None:
    particles_col.import_particles("D7-017", fiji_csv)
    with corrected(particles_col).open("a", encoding="utf-8") as fh:
        fh.write("3,10.5,10.5,5.0\n")
    before = corrected(particles_col).read_bytes()
    with pytest.raises(
        OutputExistsError,
        match=r"particles_corrected\.csv of D7-017 was edited by hand; "
        r"force=True replaces it",
    ):
        particles_col.import_particles("D7-017", fiji_csv)
    assert corrected(particles_col).read_bytes() == before
    rep = particles_col.import_particles("D7-017", fiji_csv, force=True)
    assert rep["outcome"].tolist() == ["replaced"]


def test_correction_without_record_refuses(
    particles_col: Collection, fiji_csv: Path
) -> None:
    particles_col.import_particles("D7-017", fiji_csv)
    corrected_record(particles_col).unlink()
    with pytest.raises(OutputExistsError, match="has no record"):
        particles_col.import_particles("D7-017", fiji_csv)


def test_correction_from_scratch_warns(
    particles_col: Collection, tmp_path: Path
) -> None:
    particles_col.detect(progress=False)
    source = tmp_path / "mine.csv"
    source.write_text("id,x,y,r\n1,10.5,10.5,5.0\n", encoding="utf-8")
    with pytest.warns(UserWarning) as ws:
        particles_col.import_particles("D7-017", source)
    (w,) = [w for w in ws if "keeps only" in str(w.message)]
    assert str(w.message) == (
        "correction keeps only 0 of 2 detected circles; shown as not based "
        "on the detection"
    )
    assert os.path.normcase(w.filename) == os.path.normcase(__file__)
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.source.format == "agglpy"
    assert rec.started_from is not None
    assert rec.started_from.kept == 0


def test_unknown_name(particles_col: Collection, fiji_csv: Path) -> None:
    with pytest.raises(ParamsError, match="no image 'X'"):
        particles_col.import_particles("X", fiji_csv)


def test_bad_source_propagates(
    particles_col: Collection, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        particles_col.import_particles("D7-017", tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(FileFormatError):
        particles_col.import_particles("D7-017", bad)
    assert not work(particles_col, "D7-017").exists()


def test_non_circular_rois_warning_reaches_the_caller(
    particles_col: Collection, fiji_csv: Path
) -> None:
    with fiji_csv.open("a", encoding="utf-8") as fh:
        fh.write("3,2,roi-2,Rectangle,none,5,5,10,20\n")
    with pytest.warns(NonCircularParticlesWarning) as ws:
        rep = particles_col.import_particles("D7-017", fiji_csv)
    (w,) = [w for w in ws if w.category is NonCircularParticlesWarning]
    assert os.path.normcase(w.filename) == os.path.normcase(__file__)
    assert rep["warnings"].tolist() == ["NonCircularParticlesWarning"]
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.warnings[0].startswith("NonCircularParticlesWarning: ")


def test_source_untouched(particles_col: Collection, fiji_csv: Path) -> None:
    before = (fiji_csv.read_bytes(), fiji_csv.stat().st_mtime_ns)
    os.chmod(fiji_csv, stat.S_IREAD)
    try:
        particles_col.import_particles("D7-017", fiji_csv)
    finally:
        os.chmod(fiji_csv, stat.S_IREAD | stat.S_IWRITE)
    assert (fiji_csv.read_bytes(), fiji_csv.stat().st_mtime_ns) == before


@pytest.mark.filterwarnings("ignore:correction keeps only")
def test_old_fitting_csv_reads_as_imagej(
    particles_col: Collection, tmp_path: Path
) -> None:
    source = tmp_path / "D7-017_fitting.csv"
    shutil.copy(FITTING, source)
    particles_col.detect(progress=False)
    particles_col.import_particles("D7-017", source)
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.source.format == "imagej"


def test_relative_source_is_stored_absolute(
    particles_col: Collection,
    fiji_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(fiji_csv.parent.parent)
    particles_col.import_particles("D7-017", Path("fiji") / fiji_csv.name)
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.source.path == fiji_csv.resolve().as_posix()


def test_replaced_record_keeps_only_its_own_warnings(
    particles_col: Collection, fiji_csv: Path
) -> None:
    particles_col.import_particles("D7-017", fiji_csv)
    corrected_record(particles_col).write_text("format: [", encoding="utf-8")
    with pytest.warns(UnreadableRecordWarning):
        particles_col.import_particles("D7-017", fiji_csv, force=True)
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.warnings == ()
