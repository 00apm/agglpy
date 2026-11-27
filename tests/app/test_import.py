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
    CorrectionMismatchWarning,
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
    with pytest.warns(
        DuplicateParticlesWarning, match="1 pairs of near-identical circles"
    ):
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
    # outside the collection: absolute, it stays put when the
    # collection moves
    assert rec.source.path == fiji_csv.resolve().as_posix()
    assert rec.source.sha256 == sha256(fiji_csv)
    assert rec.image.file == "../images/D7-017.tif"
    assert rec.particles.sha256 == sha256(corrected(particles_col))
    hct = read_record(
        work(particles_col, "D7-017") / "particles_hct_record.yml",
        DetectionRecord,
    )
    assert hct is not None
    assert rec.base is not None
    assert rec.base.kept == 2
    assert rec.base.detection_count == 2
    assert rec.base.detected == hct.created
    assert rec.base.sha256 == hct.particles.sha256
    table = read_particles(corrected(particles_col))
    assert table.columns.tolist() == ["id", "x", "y", "r"]
    assert len(table) == 2


def test_source_inside_the_collection_is_stored_relative(
    particles_col: Collection, fiji_csv: Path
) -> None:
    inside = particles_col.folder / "raw" / "corrections" / "D7-017.csv"
    inside.parent.mkdir(parents=True)
    shutil.copy(fiji_csv, inside)
    particles_col.import_particles("D7-017", inside)
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.source.path == "raw/corrections/D7-017.csv"


def test_import_without_detection_or_image(
    particles_col: Collection, fiji_csv: Path
) -> None:
    (particles_col.folder.parent / "images" / "D7-017.tif").unlink()
    rep = particles_col.import_particles("D7-017", fiji_csv)
    assert rep["outcome"].tolist() == ["imported"]
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.base is None
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
        match=r"particles_corrected\.csv of D7-017 was edited outside agglpy; "
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


FEW_KEPT = (
    "it keeps only 0 of 2 detected circles (status() shows "
    "correction_base: no detection)"
)


@pytest.mark.parametrize(
    ("detected", "extra", "signs"),
    [
        (False, ["1,10.5,10.5,5.0"], [FEW_KEPT]),
        (
            True,
            ["9,500.0,80.0,5.0", "10,-30.0,80.0,5.0"],
            ["2 circles lie entirely outside the image"],
        ),
        (
            False,
            ["1,10.5,10.5,5.0", "2,500.0,80.0,5.0"],
            [FEW_KEPT, "1 circle lies entirely outside the image"],
        ),
    ],
    ids=["few-kept", "outside", "both"],
)
def test_correction_that_may_not_fit_its_image_warns(
    particles_col: Collection,
    tmp_path: Path,
    detected: bool,
    extra: list[str],
    signs: list[str],
) -> None:
    particles_col.detect(progress=False)
    rows = []
    if detected:
        # the detected circles, so only the extra rows stand out
        hct = read_particles(
            work(particles_col, "D7-017") / "particles_hct.csv"
        )
        rows = [f"{r.id},{r.x},{r.y},{r.r}" for r in hct.itertuples()]
    source = tmp_path / "mine.csv"
    source.write_text(
        "\n".join(["id,x,y,r", *rows, *extra]) + "\n", encoding="utf-8"
    )
    with pytest.warns(CorrectionMismatchWarning) as ws:
        particles_col.import_particles("D7-017", source)
    (w,) = [w for w in ws if w.category is CorrectionMismatchWarning]
    assert str(w.message) == (
        "check that this is the correction of D7-017: " + "; ".join(signs)
    )
    assert os.path.normcase(w.filename) == os.path.normcase(__file__)
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.source.format == "agglpy"
    assert rec.base is not None
    assert rec.base.kept == (2 if detected else 0)
    # kept in the record, so status() still shows it later
    st = particles_col.status().set_index("image")
    assert st.loc["D7-017", "particle_set_warnings"] == (
        "CorrectionMismatchWarning"
    )


def test_unknown_name(particles_col: Collection, fiji_csv: Path) -> None:
    with pytest.raises(ParamsError, match="no image 'X'"):
        particles_col.import_particles("X", fiji_csv)


def test_bad_source_propagates(
    particles_col: Collection, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match=r"missing\.csv"):
        particles_col.import_particles("D7-017", tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(FileFormatError, match="not a particle file"):
        particles_col.import_particles("D7-017", bad)
    assert not work(particles_col, "D7-017").exists()


def test_non_circular_rois_warning_reaches_the_caller(
    particles_col: Collection, fiji_csv: Path
) -> None:
    with fiji_csv.open("a", encoding="utf-8") as fh:
        fh.write("3,2,roi-2,Rectangle,none,5,5,10,20\n")
    with pytest.warns(
        NonCircularParticlesWarning, match="non-circular ROIs dropped"
    ) as ws:
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


@pytest.mark.filterwarnings("ignore::agglpy.errors.CorrectionMismatchWarning")
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
    with pytest.warns(
        UnreadableRecordWarning,
        match=r"particles_corrected_record\.yml in D7-017/ unreadable",
    ):
        particles_col.import_particles("D7-017", fiji_csv, force=True)
    rec = read_record(corrected_record(particles_col), CorrectionRecord)
    assert rec is not None
    assert rec.warnings == ()
